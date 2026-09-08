"""Read what a binary says about the platform it was built for.

A cross-built wheel has exactly one interesting failure mode: it looks right and
carries the wrong machine code. Nothing in `pip`, `setuptools` or `wheel` checks
that, because on the host they never have to -- the compiler and the tag come
from the same `sysconfig`. Once the two are decoupled the tag becomes a claim,
and this module is what the claim is checked against.

`file(1)` would answer most of these questions, but only in prose, and only for
a path on disk; the bytes here come out of a zip archive. Both formats are read
directly instead:

  Mach-O   architecture, `LC_BUILD_VERSION` (which platform, which minimum OS)
           and the `LC_LOAD_DYLIB` list. The platform field is the load-bearing
           one -- an arm64 `iphoneos` dylib and an arm64 `macos` dylib differ in
           nothing else that a size or an architecture check would notice.
  ELF      class, endianness, machine and type. Android's `.so` files carry no
           API level, so that half of the tag cannot be verified from the
           artefact and is taken from the interpreter it is built against.
           Linux is the other way round -- `elf_dynamic` below reads `DT_NEEDED`
           and the `.gnu.version_r` symbol-version requirements, and the highest
           `GLIBC_x.y` in there *is* the manylinux floor (docs/platform/LINUX.md §5.2).
           There is no Mach-O analogue for that, and no Android one either.
  wasm     no symbol table at all. The import and export sections *are* the
           answer -- every name the host must resolve and every name offered,
           spelled out in the module. That makes linkage easier to read here
           than in ELF and makes the platform unreadable: a wasm module records
           no version of anything, so nothing in `pyemscripten_2026_0_wasm32`
           except `wasm32` can be checked against the bytes. See the WASM
           section below, which says which questions it declines.

Everything returns `None` rather than raising when the bytes are not of that
format, so a caller can ask both questions and use whichever answered.
"""

from __future__ import annotations

import struct

# ------------------------------------------------------------------- Mach-O

_FAT_MAGICS = {0xCAFEBABE, 0xCAFEBABF}
_THIN_MAGICS = {0xFEEDFACE, 0xFEEDFACF}

# cputype values from <mach/machine.h>; the 0x01000000 bit is CPU_ARCH_ABI64.
CPU_NAMES = {0x0100000C: "arm64", 0x01000007: "x86_64", 0x00000007: "i386"}

LC_ID_DYLIB = 0x0D
LC_LOAD_DYLIB = 0x0C
LC_BUILD_VERSION = 0x32

# The pre-LC_BUILD_VERSION spelling, one command per platform. Still emitted:
# the iOS `_C.dylib` built here carries LC_VERSION_MIN_IPHONEOS rather than
# LC_BUILD_VERSION, because Rust's default deployment target for
# `aarch64-apple-ios` (10.0) predates the newer command. Reading only
# LC_BUILD_VERSION would report `platform: None` for it and quietly weaken every
# check downstream.
LC_VERSION_MIN = {
    0x24: "macos", 0x25: "ios", 0x2F: "tvos", 0x30: "watchos",
}
LC_VERSION_MIN_IPHONEOS = 0x25

# <mach-o/loader.h> PLATFORM_*. The pair that matters here is 2 vs 7: a wheel
# tagged `..._iphoneos` holding a `iossimulator` binary installs and then fails
# on a device only, which is the slowest possible place to find out.
MACHO_PLATFORMS = {
    1: "macos", 2: "ios", 3: "tvos", 4: "watchos", 5: "bridgeos",
    6: "maccatalyst", 7: "iossimulator", 8: "tvossimulator",
    9: "watchossimulator", 10: "driverkit",
}


def _ver(packed: int) -> tuple[int, int, int]:
    """Apple's xxxx.yy.zz packed into 32 bits."""
    return (packed >> 16, (packed >> 8) & 0xFF, packed & 0xFF)


def macho_arches(data: bytes) -> list[str]:
    """Architectures actually present in a Mach-O image (fat or thin)."""
    if len(data) < 8:
        return []
    magic_be = struct.unpack_from(">I", data)[0]
    if magic_be in _FAT_MAGICS:
        # `fat_arch` is 20 bytes, `fat_arch_64` (magic ...BF) is 32; both start
        # with cputype, which is the only field wanted here.
        stride = 20 if magic_be == 0xCAFEBABE else 32
        count = struct.unpack_from(">I", data, 4)[0]
        return [
            CPU_NAMES.get(c, f"cpu{c:#x}")
            for c in (
                struct.unpack_from(">I", data, 8 + i * stride)[0]
                for i in range(count)
            )
        ]
    for endian in ("<", ">"):
        magic = struct.unpack_from(endian + "I", data)[0]
        if magic in _THIN_MAGICS:
            cputype = struct.unpack_from(endian + "I", data, 4)[0]
            return [CPU_NAMES.get(cputype, f"cpu{cputype:#x}")]
    return []


def macho_info(data: bytes) -> dict | None:
    """Architecture, target platform, minimum OS and dylib list of a thin
    64-bit Mach-O. `None` if `data` is not one (fat images included -- nothing
    this project ships is fat, and pretending to summarise one would hide which
    slice the answer came from).
    """
    if len(data) < 32:
        return None
    magic = struct.unpack_from("<I", data)[0]
    if magic != 0xFEEDFACF:  # thin, 64-bit, little-endian
        return None
    cputype, _sub, _ftype, ncmds, _sizeofcmds, _flags, _res = struct.unpack_from(
        "<IIIIIII", data, 4
    )
    out: dict = {
        "format": "macho",
        "arch": CPU_NAMES.get(cputype, f"cpu{cputype:#x}"),
        "platform": None,
        "minos": None,
        "id": None,
        "dylibs": [],
    }
    off = 32
    for _ in range(ncmds):
        if off + 8 > len(data):
            break
        cmd, cmdsize = struct.unpack_from("<II", data, off)
        if cmdsize < 8 or off + cmdsize > len(data):
            break
        if cmd == LC_BUILD_VERSION and cmdsize >= 24:
            plat, minos, _sdk, _ntools = struct.unpack_from("<IIII", data, off + 8)
            out["platform"] = MACHO_PLATFORMS.get(plat, f"platform{plat}")
            out["minos"] = _ver(minos)
        elif cmd in LC_VERSION_MIN and cmdsize >= 16 and not out["platform"]:
            version, _sdk = struct.unpack_from("<II", data, off + 8)
            out["platform"] = LC_VERSION_MIN[cmd]
            out["minos"] = _ver(version)
        elif cmd in (LC_ID_DYLIB, LC_LOAD_DYLIB) and cmdsize >= 24:
            name_off = struct.unpack_from("<I", data, off + 8)[0]
            start = off + name_off
            end = data.find(b"\0", start, off + cmdsize)
            name = data[start: end if end >= 0 else off + cmdsize].decode(
                "utf-8", "replace"
            )
            if cmd == LC_ID_DYLIB:
                out["id"] = name
            else:
                out["dylibs"].append(name)
        off += cmdsize
    return out


# ---------------------------------------------------------------------- ELF

# e_machine values from <elf.h>.
ELF_MACHINES = {0xB7: "aarch64", 0x3E: "x86_64", 0x28: "arm", 0x03: "i386"}
ELF_TYPES = {1: "rel", 2: "exec", 3: "dyn", 4: "core"}


def elf_info(data: bytes) -> dict | None:
    """Class, endianness, machine and type of an ELF image. `None` otherwise."""
    if len(data) < 20 or data[:4] != b"\x7fELF":
        return None
    ei_class, ei_data = data[4], data[5]
    endian = "<" if ei_data == 1 else ">"
    e_type, e_machine = struct.unpack_from(endian + "HH", data, 16)
    return {
        "format": "elf",
        "bits": {1: 32, 2: 64}.get(ei_class, ei_class),
        "endian": {1: "little", 2: "big"}.get(ei_data, ei_data),
        "machine": ELF_MACHINES.get(e_machine, f"machine{e_machine:#x}"),
        "type": ELF_TYPES.get(e_type, f"type{e_type}"),
    }


# Section types from <elf.h>. `SHT_GNU_verneed` is the one with no Mach-O
# counterpart: it records, per needed library, which *symbol versions* the image
# requires. For a glibc target the highest `GLIBC_x.y` in it is what auditwheel
# calls the policy floor, and it is the only place that number exists -- CPython's
# `_sysconfigdata_*.py` has no field for it (docs/platform/LINUX.md §3), unlike
# `ANDROID_API_LEVEL` and `IPHONEOS_DEPLOYMENT_TARGET`.
SHT_DYNSYM = 11
SHT_DYNAMIC = 6
SHT_GNU_VERNEED = 0x6FFFFFFE
SHT_GNU_VERSYM = 0x6FFFFFFF

DT_NULL, DT_NEEDED, DT_SONAME = 0, 1, 14

_EHDR64_SHOFF = 0x28
_EHDR64_SHENTSIZE = 0x3A
_SHDR64_SIZE = 64
_DYN64_SIZE = 16
_VERNEED64_SIZE = 16
_VERNAUX64_SIZE = 16


def _elf_sections(data: bytes):
    """`(endian, [section], cstr)` for a 64-bit ELF, or `None`.

    Section headers rather than `PT_DYNAMIC`, because they need no
    address-to-offset mapping and every `.so` a compiler emits has them. An image
    that has been stripped of them answers `None`, which callers report as "could
    not be read" -- never as "requires nothing", which is the direction that
    would turn a missing section into a passing check.
    """
    if len(data) < 0x40 or data[:4] != b"\x7fELF" or data[4] != 2:
        return None
    endian = "<" if data[5] == 1 else ">"
    try:
        shoff, = struct.unpack_from(endian + "Q", data, _EHDR64_SHOFF)
        shentsize, shnum, shstrndx = struct.unpack_from(
            endian + "HHH", data, _EHDR64_SHENTSIZE)
    except struct.error:
        return None
    if not shoff or not shnum or shentsize < _SHDR64_SIZE:
        return None
    if shoff + shnum * shentsize > len(data) or shstrndx >= shnum:
        return None

    sections = []
    for i in range(shnum):
        try:
            name, stype, _flags, addr, off, size, link, _info, _al, entsize = \
                struct.unpack_from(endian + "IIQQQQIIQQ", data,
                                   shoff + i * shentsize)
        except struct.error:
            return None
        sections.append({"name": name, "type": stype, "addr": addr, "off": off,
                         "size": size, "link": link, "entsize": entsize})

    def cstr(base: int, offset: int) -> str:
        start = base + offset
        if not 0 <= start < len(data):
            return ""
        end = data.find(b"\0", start)
        if end < 0:
            end = len(data)
        return data[start:end].decode("utf-8", "replace")

    shstr = sections[shstrndx]["off"]
    for section in sections:
        section["sname"] = cstr(shstr, section["name"])
    return endian, sections, cstr


def elf_dynamic(data: bytes) -> dict | None:
    """`DT_SONAME`, `DT_NEEDED` and the per-library version requirements.

        {"soname": "libpython3.13.so.1.0",
         "needed": ["libm.so.6", ..., "libc.so.6"],
         "versions": {"libc.so.6": {"GLIBC_2.2.5", ..., "GLIBC_2.17"}, ...}}

    `None` when the bytes are not a readable 64-bit ELF with section headers.
    An ELF that *is* readable but has no dynamic section answers with empty
    fields, which is a different thing and is reported differently by callers:
    the first is the check failing to run, the second is a finding.
    """
    parsed = _elf_sections(data)
    if parsed is None:
        return None
    endian, sections, cstr = parsed

    soname: str | None = None
    needed: list[str] = []
    versions: dict[str, set[str]] = {}
    by_index: dict[int, tuple[str, str]] = {}

    for section in sections:
        if section["type"] == SHT_DYNAMIC:
            if section["link"] >= len(sections):
                continue
            strtab = sections[section["link"]]["off"]
            for i in range(section["size"] // _DYN64_SIZE):
                try:
                    tag, val = struct.unpack_from(
                        endian + "qQ", data, section["off"] + i * _DYN64_SIZE)
                except struct.error:
                    break
                if tag == DT_NULL:
                    break
                if tag == DT_NEEDED:
                    needed.append(cstr(strtab, val))
                elif tag == DT_SONAME:
                    soname = cstr(strtab, val)

        elif section["type"] == SHT_GNU_VERNEED:
            if section["link"] >= len(sections):
                continue
            strtab = sections[section["link"]]["off"]
            off = section["off"]
            end = section["off"] + section["size"]
            seen = 0
            while off + _VERNEED64_SIZE <= end and seen < 4096:
                seen += 1
                try:
                    _v, cnt, vfile, vaux, vnext = struct.unpack_from(
                        endian + "HHIII", data, off)
                except struct.error:
                    break
                library = cstr(strtab, vfile)
                aux = off + vaux
                for _ in range(cnt):
                    if aux + _VERNAUX64_SIZE > end:
                        break
                    try:
                        _h, _fl, _o, name, anext = struct.unpack_from(
                            endian + "IHHII", data, aux)
                    except struct.error:
                        break
                    versions.setdefault(library, set()).add(cstr(strtab, name))
                    # `vna_other` is the index `.gnu.version` uses to point an
                    # undefined symbol at this exact (library, version) pair.
                    # It is the only thing in an ELF that binds a symbol to a
                    # library the way a Mach-O two-level namespace does.
                    by_index[_o & 0x7FFF] = (library, cstr(strtab, name))
                    if not anext:
                        break
                    aux += anext
                if not vnext:
                    break
                off += vnext

    return {"soname": soname, "needed": needed, "versions": versions,
            "version_index": by_index}


def elf_symbols(data: bytes) -> dict | None:
    """`.dynsym`, split into what the image defines and what it needs.

        {"defined":   {"PyList_New", ...},
         "undefined": [("memcpy", "libc.so.6", "GLIBC_2.14", False),
                       ("PyList_New", None, None, False),
                       ("__gmon_start__", None, None, True), ...]}

    The fourth element is `STB_WEAK`. It has to be carried, not filtered here,
    because a weak undefined symbol is *allowed* to stay unresolved -- every
    shared object gcc or clang emits carries `__gmon_start__`,
    `_ITM_registerTMCloneTable` and `_ITM_deregisterTMCloneTable`, none of which
    exists anywhere on an ordinary system. A resolver that counted those as
    failures would report three every time and teach its reader to ignore it.

    The second element of each undefined triple is **the library the symbol is
    bound to, when the ELF says so** -- and the whole difficulty of checking a
    Linux extension is that it usually does not.

    ELF resolves undefined symbols by a flat search across everything loaded, so
    unlike a two-level-namespace Mach-O (`tools/wheel/verify_ios_device.py`) an
    import carries no library name. The exception is symbol *versioning*:
    `.gnu.version` gives each `.dynsym` entry an index, and for an undefined
    symbol that index points into `.gnu.version_r`, which is grouped by library.
    glibc versions all of its exports, so every libc import does name its
    library; CPython versions none of its own, so no `Py*` import does.

    That asymmetry is the honest limit of ELF symbol checking, and it is why the
    Linux check is weaker than the iOS one. `None` means unversioned, not
    unbound.

    Returns `None` when the bytes are not a readable 64-bit ELF with section
    headers -- the same "could not run" answer `elf_dynamic` gives. Unlike
    `elf_dynamic`, a *readable* ELF with no `SHT_DYNSYM` section also answers
    `None` rather than the empty `{"defined": set(), "undefined": []}`: a
    `DT_DYNAMIC`-less ELF is a legitimate thing (a static executable), so
    `elf_dynamic` reporting empty fields for one is a finding. A `.dynsym`-less
    *dynamic* shared object is not a legitimate thing -- `ld.so` could not
    resolve a single symbol in or out of it -- so its absence here means the
    section table this read could not be trusted, not that the image needs
    nothing. Confirmed live: zeroing one section's `sh_type` from `SHT_DYNSYM`
    to `SHT_NULL` in an otherwise-untouched `libtorch_global_deps.so` --
    `elf_info` still reports it as a normal 64-bit x86-64 `dyn` ELF -- used to
    make this return `{"defined": 0 symbols, "undefined": 0 symbols}`, which
    `verify_linux.py`'s `resolve()` reports as "0 undefined (0 exported)" and
    "0 unresolved": a clean PASS on a file whose symbol table was never read.
    """
    parsed = _elf_sections(data)
    if parsed is None:
        return None
    endian, sections, cstr = parsed

    dynsym_sections = [s for s in sections if s["type"] == SHT_DYNSYM]
    if not dynsym_sections:
        # See the docstring: for a dynamic shared object this is not "it
        # defines and needs nothing", it is this read having failed to find
        # the one section the whole answer depends on.
        return None

    dynamic = elf_dynamic(data) or {"version_index": {}}
    by_index = dynamic["version_index"]

    versym: list[int] = []
    for section in sections:
        if section["type"] == SHT_GNU_VERSYM and section["entsize"] == 2:
            count = section["size"] // 2
            versym = list(struct.unpack_from(
                endian + f"{count}H", data, section["off"]))
            break

    defined: set[str] = set()
    undefined: list[tuple[str, str | None, str | None, bool]] = []
    for section in dynsym_sections:
        if section["entsize"] != 24:
            continue
        if section["link"] >= len(sections):
            continue
        strtab = sections[section["link"]]["off"]
        for i in range(section["size"] // 24):
            try:
                name, info, _other, shndx, _value, _size = struct.unpack_from(
                    endian + "IBBHQQ", data, section["off"] + i * 24)
            except struct.error:
                break
            symbol = cstr(strtab, name)
            if not symbol:
                continue
            binding = info >> 4
            if shndx == 0:                        # SHN_UNDEF
                index = versym[i] & 0x7FFF if i < len(versym) else 1
                library, version = by_index.get(index, (None, None))
                undefined.append((symbol, library, version, binding == 2))
            elif binding in (1, 2):               # STB_GLOBAL, STB_WEAK
                defined.add(symbol)
    return {"defined": defined, "undefined": undefined}


# ----------------------------------------------------------------------- PE
#
# PE is the format where the "which library does this symbol come from" question
# is *easiest*, which is the opposite of ELF. An import is not a name the loader
# searches for: it is an entry in a table that is indexed by DLL, so every
# undefined symbol arrives already attached to the file that has to provide it.
# That is the same guarantee Mach-O's two-level namespace gives and the one
# `.gnu.version_r` gives only for versioned symbols (docs/platform/LINUX.md §6.1).
#
# The layout read here, from PE/COFF:
#
#   "MZ" .. e_lfanew(0x3C) -> "PE\0\0" + COFF header + optional header
#   optional header magic 0x20B = PE32+ (64-bit), 0x10B = PE32
#   data directory 1 = import table, 12 = delay import, 0 = export table
#   each IMAGE_IMPORT_DESCRIPTOR: OriginalFirstThunk, .., Name(RVA), FirstThunk
#   the thunk array is IMAGE_THUNK_DATA: high bit set = import by ordinal,
#   otherwise an RVA to IMAGE_IMPORT_BY_NAME { WORD Hint; char Name[] }

PE_MACHINES = {0x8664: "x86_64", 0xAA64: "aarch64", 0x14C: "i386",
               0x1C0: "arm", 0x1C4: "armnt"}

IMAGE_FILE_DLL = 0x2000


def _pe_headers(data: bytes):
    """(coff_offset, machine, magic, characteristics, sections) or None."""
    if len(data) < 0x40 or data[:2] != b"MZ":
        return None
    (lfanew,) = struct.unpack_from("<I", data, 0x3C)
    if lfanew + 24 > len(data) or data[lfanew:lfanew + 4] != b"PE\0\0":
        return None
    coff = lfanew + 4
    machine, nsections = struct.unpack_from("<HH", data, coff)
    opt_size, characteristics = struct.unpack_from("<HH", data, coff + 16)
    if opt_size < 2:
        return None
    (magic,) = struct.unpack_from("<H", data, coff + 20)
    sections = []
    base = coff + 20 + opt_size
    for i in range(nsections):
        off = base + i * 40
        if off + 40 > len(data):
            break
        name = data[off:off + 8].rstrip(b"\0").decode("ascii", "replace")
        vsize, vaddr, rawsize, rawptr = struct.unpack_from("<IIII", data, off + 8)
        sections.append((name, vaddr, vsize, rawptr, rawsize))
    return coff, machine, magic, characteristics, sections


def pe_info(data: bytes) -> dict | None:
    """Machine, bitness and DLL-ness, or None if these bytes are not PE."""
    parsed = _pe_headers(data)
    if parsed is None:
        return None
    _, machine, magic, characteristics, _ = parsed
    if magic not in (0x10B, 0x20B):
        return None
    return {
        "machine": PE_MACHINES.get(machine, f"0x{machine:x}"),
        "bits": 64 if magic == 0x20B else 32,
        "dll": bool(characteristics & IMAGE_FILE_DLL),
    }


def _rva_to_offset(sections, rva: int) -> int | None:
    for _, vaddr, vsize, rawptr, rawsize in sections:
        if vaddr <= rva < vaddr + max(vsize, rawsize):
            delta = rva - vaddr
            if delta < rawsize:
                return rawptr + delta
            return None
    return None


def _cstring(data: bytes, offset: int) -> str:
    end = data.find(b"\0", offset)
    return data[offset:end if end >= 0 else len(data)].decode("ascii", "replace")


def _data_directory(data: bytes, coff: int, magic: int, index: int):
    # The directory array starts after the optional header's fixed part, whose
    # size differs between PE32 (96 bytes) and PE32+ (112).
    fixed = 112 if magic == 0x20B else 96
    base = coff + 20 + fixed
    (count,) = struct.unpack_from("<I", data, base - 4)
    if index >= count:
        return 0, 0
    return struct.unpack_from("<II", data, base + index * 8)


def pe_imports(data: bytes) -> dict[str, set[str]] | None:
    """`{dll name: {imported symbol, ...}}`, straight out of the import table.

    Ordinal-only imports appear as `#<n>`, because that is all the file records:
    the name lives in the exporting DLL's ordinal table and is not recoverable
    from here. Delay-loaded imports (directory 12) are merged in -- they resolve
    at first call rather than at load, so they are still symbols this image
    requires from that DLL, and leaving them out would understate the
    dependency.
    """
    parsed = _pe_headers(data)
    if parsed is None:
        return None
    coff, _, magic, _, sections = parsed
    if magic not in (0x10B, 0x20B):
        return None
    ptr_size = 8 if magic == 0x20B else 4
    ordinal_flag = 1 << (ptr_size * 8 - 1)
    fmt = "<Q" if ptr_size == 8 else "<I"

    result: dict[str, set[str]] = {}
    for directory, descriptor_size, name_field, thunk_fields in (
            (1, 20, 12, (0, 16)), (12, 32, 4, (16, 12))):
        rva, _size = _data_directory(data, coff, magic, directory)
        if not rva:
            continue
        table = _rva_to_offset(sections, rva)
        if table is None:
            continue
        while table + descriptor_size <= len(data):
            fields = struct.unpack_from(f"<{descriptor_size // 4}I", data, table)
            if not any(fields):
                break
            name_rva = fields[name_field // 4]
            name_off = _rva_to_offset(sections, name_rva)
            if name_off is None:
                break
            # The delay-load directory used image-relative addresses in its
            # original form and RVAs since VS2015; both appear in the wild, and
            # a value that does not land in a section means the former.
            dll = _cstring(data, name_off)
            names = result.setdefault(dll, set())
            for field in thunk_fields:
                thunk_rva = fields[field // 4]
                thunk = _rva_to_offset(sections, thunk_rva) if thunk_rva else None
                if thunk is None:
                    continue
                while thunk + ptr_size <= len(data):
                    (entry,) = struct.unpack_from(fmt, data, thunk)
                    if entry == 0:
                        break
                    if entry & ordinal_flag:
                        names.add(f"#{entry & 0xFFFF}")
                    else:
                        hint = _rva_to_offset(sections, entry & 0x7FFFFFFF)
                        if hint is not None:
                            names.add(_cstring(data, hint + 2))
                    thunk += ptr_size
                break  # the first thunk array that resolves is enough
            table += descriptor_size
    return result


def pe_exports(data: bytes) -> set[str] | None:
    """Names in the export directory. Used to resolve another image's imports."""
    parsed = _pe_headers(data)
    if parsed is None:
        return None
    coff, _, magic, _, sections = parsed
    if magic not in (0x10B, 0x20B):
        return None
    rva, _size = _data_directory(data, coff, magic, 0)
    if not rva:
        return set()
    table = _rva_to_offset(sections, rva)
    if table is None or table + 40 > len(data):
        return set()
    count, names_rva = struct.unpack_from("<I", data, table + 24)[0], \
        struct.unpack_from("<I", data, table + 32)[0]
    names_off = _rva_to_offset(sections, names_rva)
    if names_off is None:
        return set()
    exported: set[str] = set()
    for i in range(count):
        if names_off + i * 4 + 4 > len(data):
            break
        (name_rva,) = struct.unpack_from("<I", data, names_off + i * 4)
        offset = _rva_to_offset(sections, name_rva)
        if offset is not None:
            exported.add(_cstring(data, offset))
    return exported


# --------------------------------------------------------------------- WASM
#
# The three readers above all answer their questions out of a *symbol table*:
# ELF has `.dynsym` and `.dynamic`, Mach-O has `LC_LOAD_DYLIB` and the symtab,
# PE has an import directory and an export directory. A WebAssembly module has
# none of those. What it has instead is two mandatory sections in the module
# itself -- `import` (id 2) and `export` (id 7) -- which name every symbol the
# host must supply and every symbol the module offers, by string, in the binary.
#
# That makes the *linkage* questions easier to answer than in ELF (no string
# table indirection, no versioning records, no address-to-offset mapping) and
# some other questions impossible. Specifically:
#
#   answerable    which symbols must be resolved by the host, under which
#                 import module name -- `wasm_imports`, the `DT_NEEDED` +
#                 undefined-symbol pair in one structure
#                 which symbols are offered, and of which kind --
#                 `wasm_exports`. `PyInit__C` has to be there, and has to be a
#                 *function*: an exported global of that name is a linker
#                 accident that no loader would call.
#                 whether the module is loadable by `dlopen` at all -- a side
#                 module imports its memory and function table, a main module
#                 defines them, and Emscripten's `dlopen` refuses a main-module
#                 link (docs/platform/WASM.md §9.2). Not `dylink.0`, which both have.
#                 wasm32 vs wasm64, from the memory type's limits flags
#
#   NOT answerable  the platform, the way `LC_BUILD_VERSION` gives it. A wasm
#                 module records no Emscripten version, no Pyodide ABI version
#                 and no minimum anything; two modules built a major release
#                 apart are byte-indistinguishable in their headers. Every
#                 version component of `pyemscripten_2026_0_wasm32` is
#                 therefore unverifiable *from the artefact*, and the caller has
#                 to say so rather than checking a nearby easier thing.
#                 the manylinux-style floor. No counterpart, same as Mach-O.
#                 which imports come from libpython rather than from the C
#                 runtime. Emscripten puts every undefined symbol under the
#                 single import module `env`, so the `python3.dll` vs
#                 `python313.dll` distinction that `pe_imports` uses to prove
#                 an abi3 binding has no wasm spelling at all.
#
# Everything here returns `None` on bytes that are not a well-formed module,
# never a plausible empty answer: an unreadable module must be reported as
# unreadable, not as one that imports nothing.

WASM_MAGIC = b"\0asm"

# Section ids from the core specification, §5.5.2.
WASM_SECTIONS = {
    0: "custom", 1: "type", 2: "import", 3: "function", 4: "table",
    5: "memory", 6: "global", 7: "export", 8: "start", 9: "element",
    10: "code", 11: "data", 12: "datacount", 13: "tag",
}

# External kinds, shared by the import and export sections (§5.5.5, §5.5.10).
# `tag` is the exception-handling proposal's, which matters here: every module
# in this process is built `-fwasm-exceptions` (docs/platform/WASM.md §7.4a), so a real
# artefact does carry kind 4 and a reader that stopped at 3 would fail on it.
WASM_KINDS = {0: "func", 1: "table", 2: "memory", 3: "global", 4: "tag"}

# `dylink.0` subsection ids, from Emscripten's dynamic-linking ABI.
_DYLINK_MEM_INFO = 1
_DYLINK_NEEDED = 2


def _uleb(data: bytes, i: int) -> tuple[int, int]:
    """LEB128, bounded. Raises `_WasmTruncated` past the end or past 5 bytes."""
    result = shift = 0
    for _ in range(5):
        if i >= len(data):
            raise _WasmTruncated
        byte = data[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, i
        shift += 7
    raise _WasmTruncated


class _WasmTruncated(Exception):
    """The bytes ran out, or a length field pointed past the end."""


def _wasm_name(data: bytes, i: int) -> tuple[str, int]:
    length, i = _uleb(data, i)
    if i + length > len(data):
        raise _WasmTruncated
    return data[i:i + length].decode("utf-8", "replace"), i + length


def _wasm_sections(data: bytes) -> list[tuple[int, bytes]] | None:
    """`[(section id, body)]` for a well-formed module, else `None`.

    Deliberately strict about the trailing bytes: a section whose declared size
    runs past the end of the file makes the whole module unreadable rather than
    yielding the sections before it. A truncated `_C.wasm` that still had a
    valid export section would otherwise pass every export check while being
    unloadable.
    """
    if len(data) < 8 or data[:4] != WASM_MAGIC:
        return None
    version = int.from_bytes(data[4:8], "little")
    if version != 1:
        return None
    out: list[tuple[int, bytes]] = []
    i = 8
    try:
        while i < len(data):
            sid = data[i]
            i += 1
            if sid not in WASM_SECTIONS:
                return None
            size, i = _uleb(data, i)
            if i + size > len(data):
                raise _WasmTruncated
            out.append((sid, data[i:i + size]))
            i += size
    except _WasmTruncated:
        return None
    return out


def _dylink(body: bytes) -> dict | None:
    """The `dylink.0` custom section's contents, or `None` if unreadable."""
    try:
        name, i = _wasm_name(body, 0)
        if name != "dylink.0":
            return None
        info: dict = {"needed": [], "mem_size": None, "table_size": None}
        while i < len(body):
            sub = body[i]
            i += 1
            size, i = _uleb(body, i)
            if i + size > len(body):
                raise _WasmTruncated
            end = i + size
            if sub == _DYLINK_MEM_INFO:
                j = i
                info["mem_size"], j = _uleb(body, j)
                _align, j = _uleb(body, j)
                info["table_size"], j = _uleb(body, j)
            elif sub == _DYLINK_NEEDED:
                count, j = _uleb(body, i)
                for _ in range(count):
                    lib, j = _wasm_name(body, j)
                    info["needed"].append(lib)
            i = end
        return info
    except _WasmTruncated:
        return None


def wasm_info(data: bytes) -> dict | None:
    """Header facts about a WebAssembly module. `None` if it is not one.

    `side_module` is the load-bearing field and it has no analogue in the other
    three formats' `info` dicts, because in those formats "is this loadable"
    is a type field in the header. A main-module link cannot be `dlopen`ed --
    and therefore cannot be a `ctypes.CDLL` or an extension module
    (docs/platform/WASM.md §9.2) -- so this is the wasm spelling of PE's `dll` bit.

    It is **not** the presence of `dylink.0`, which was the first guess here and
    is wrong: Pyodide's own `pyodide.asm.wasm` is a `-sMAIN_MODULE` link and
    carries a `dylink.0` too (with `mem_size` 0). The distinction that does hold,
    measured across all 31 wasm modules in the six Pyodide wheels on this machine
    plus that main module, is *who owns the memory and the function table*: a
    side module imports `env.memory` and `env.__indirect_function_table`, and a
    main module defines both and exports them under `memory` and `table`.
    """
    sections = _wasm_sections(data)
    if sections is None:
        return None
    bits = 32
    dylink = None
    defines_memory = False
    for sid, body in sections:
        if sid == 0 and dylink is None:
            dylink = _dylink(body)
        elif sid == 5:                                   # defined memory
            try:
                count, i = _uleb(body, 0)
                if count:
                    defines_memory = True
                    bits = 64 if body[i] & 0x04 else 32
            except _WasmTruncated:                       # pragma: no cover
                return None
    imported_memory = _wasm_memory_bits(sections)
    if imported_memory is not None:
        bits = imported_memory
    return {
        "format": "wasm",
        "version": int.from_bytes(data[4:8], "little"),
        "bits": bits,
        "machine": f"wasm{bits}",
        "dylink": dylink is not None,
        "side_module": (dylink is not None and imported_memory is not None
                        and not defines_memory),
        "needed": list(dylink["needed"]) if dylink else [],
        "mem_size": dylink["mem_size"] if dylink else None,
        "table_size": dylink["table_size"] if dylink else None,
        "sections": [WASM_SECTIONS[sid] for sid, _ in sections],
    }


def _wasm_memory_bits(sections) -> int | None:
    """32 or 64 from an *imported* memory's limits flags, or `None` if none.

    A side module imports `env.memory` rather than defining one, so the bit that
    says wasm32 lives in the import section for exactly the modules this
    repository builds -- reading only section 5 would report every one of them
    as 32-bit by default rather than by observation.
    """
    for sid, body in sections:
        if sid != 2:
            continue
        try:
            count, i = _uleb(body, 0)
            for _ in range(count):
                _module, i = _wasm_name(body, i)
                _field, i = _wasm_name(body, i)
                kind = body[i]
                i += 1
                if kind == 0:
                    _, i = _uleb(body, i)
                elif kind == 1:
                    i += 1
                    flags = body[i]
                    i += 1
                    _, i = _uleb(body, i)
                    if flags & 0x01:
                        _, i = _uleb(body, i)
                elif kind == 2:
                    flags = body[i]
                    i += 1
                    _, i = _uleb(body, i)
                    if flags & 0x01:
                        _, i = _uleb(body, i)
                    return 64 if flags & 0x04 else 32
                elif kind == 3:
                    i += 2
                elif kind == 4:
                    i += 1
                    _, i = _uleb(body, i)
                else:
                    return None
        except (_WasmTruncated, IndexError):
            return None
    return None


def wasm_imports(data: bytes) -> dict[str, set[str]] | None:
    """`{import module: {field}}` -- what the host has to supply.

    Same shape as `pe_imports` on purpose, so the two can be read side by side,
    but the resemblance stops at the shape. In PE the keys are DLL filenames and
    carry information (`python3.dll` vs `python313.dll` is the whole abi3
    check); under Emscripten the keys are `env` for real undefined symbols and
    `GOT.mem` / `GOT.func` for the dynamic-linking global-offset-table entries
    the loader fills in. There is no per-library grouping to read.

    A module with no import section imports nothing, which is a real answer;
    `None` is reserved for bytes that could not be read as a module at all.
    """
    records = wasm_import_records(data)
    if records is None:
        return None
    out: dict[str, set[str]] = {}
    for module, field, _kind in records:
        out.setdefault(module, set()).add(field)
    return out


def wasm_import_records(data: bytes) -> list[tuple[str, str, str]] | None:
    """`[(import module, field, kind)]`, kind spelled as in `WASM_KINDS`."""
    sections = _wasm_sections(data)
    if sections is None:
        return None
    out: list[tuple[str, str, str]] = []
    for sid, body in sections:
        if sid != 2:
            continue
        try:
            count, i = _uleb(body, 0)
            for _ in range(count):
                module, i = _wasm_name(body, i)
                field, i = _wasm_name(body, i)
                if i >= len(body):
                    raise _WasmTruncated
                kind = body[i]
                i += 1
                if kind == 0:                            # typeidx
                    _, i = _uleb(body, i)
                elif kind == 1:                          # reftype + limits
                    i += 1
                    flags = body[i]
                    i += 1
                    _, i = _uleb(body, i)
                    if flags & 0x01:
                        _, i = _uleb(body, i)
                elif kind == 2:                          # limits
                    flags = body[i]
                    i += 1
                    _, i = _uleb(body, i)
                    if flags & 0x01:
                        _, i = _uleb(body, i)
                elif kind == 3:                          # valtype + mutability
                    i += 2
                elif kind == 4:                          # tag: attribute + type
                    i += 1
                    _, i = _uleb(body, i)
                else:
                    return None
                out.append((module, field, WASM_KINDS.get(kind, f"kind{kind}")))
        except (_WasmTruncated, IndexError):
            return None
    return out


def wasm_exports(data: bytes) -> dict[str, str] | None:
    """`{exported name: kind}` -- what the module offers.

    `pe_exports` returns a bare set because a PE export is always callable-ish;
    here the kind is kept because it is checkable and worth checking. An
    extension module is found by `dlsym`-ing `PyInit__C` and *calling* it, so
    `PyInit__C` exported as kind `global` -- which a data symbol of that name
    would be -- is a wheel that fails at import with nothing pointing at the
    export section.
    """
    sections = _wasm_sections(data)
    if sections is None:
        return None
    out: dict[str, str] = {}
    for sid, body in sections:
        if sid != 7:
            continue
        try:
            count, i = _uleb(body, 0)
            for _ in range(count):
                field, i = _wasm_name(body, i)
                if i >= len(body):
                    raise _WasmTruncated
                kind = body[i]
                i += 1
                _index, i = _uleb(body, i)
                out[field] = WASM_KINDS.get(kind, f"kind{kind}")
        except (_WasmTruncated, IndexError):
            return None
    return out


def describe(data: bytes) -> str:
    """One line, for printing next to a filename."""
    macho = macho_info(data)
    if macho:
        bits = [f"Mach-O {macho['arch']}"]
        if macho["platform"]:
            minos = ".".join(str(n) for n in (macho["minos"] or ())[:2])
            bits.append(f"{macho['platform']} {minos}+")
        return " ".join(bits)
    elf = elf_info(data)
    if elf:
        return (f"ELF {elf['bits']}-bit {elf['endian']}-endian "
                f"{elf['machine']} {elf['type']}")
    pe = pe_info(data)
    if pe:
        return (f"PE{'32+' if pe['bits'] == 64 else '32'} "
                f"{pe['machine']} {'dll' if pe['dll'] else 'exe'}")
    wasm = wasm_info(data)
    if wasm:
        return (f"wasm {wasm['bits']}-bit v{wasm['version']} "
                f"{'side module' if wasm['side_module'] else 'main module'}")
    if macho_arches(data):
        return "Mach-O fat: " + "+".join(macho_arches(data))
    return "not a recognised binary"
