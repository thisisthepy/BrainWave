"""ExecuTorch's Qualcomm (QNN) backend, as a subgraph delegate behind an nn.Module.

docs/QNN.md is this module's design record. The two decisions it exists to
implement, both taken before any code:

**The lowering is upstream's and it happens offline.** `torch.export.export`,
`to_edge_transform_and_lower_to_qnn` and the QNN context-binary compiler all
run on a Linux x86-64 host with the Qualcomm AI Engine Direct SDK installed,
under **upstream** torch and **upstream** executorch. torchnative is not in
that process. What crosses the boundary is one file -- a `.pte` -- and this
module's runtime half only loads it. That is deliberate: `docs/EXPORT5.md`
measures torchnative's own `torch.export` at four hand-written modules and
**zero** real architectures, and a design that put it on the critical path to
an NPU would be betting the whole path on the weakest thing in the tree.

**The delegate is a subgraph, not a runtime.** The `.pte` stands behind one
`nn.Module` inside a real `transformers` model. `from_pretrained` and
`generate` are untouched, the user never holds the `.pte`, and everything the
partitioner declined stays in eager torch. `torchnative.export.npu` is the
piece that does the swapping and it knows nothing about Qualcomm.

## What this host can and cannot do

Measured, not assumed -- `probe()` re-measures it on every call:

* `torch.export`, `to_edge_transform_and_lower` and the ExecuTorch runtime all
  work on macOS arm64. `lower_cpu_reference` uses them and it runs here.
* The **QNN** half does not. Every one of ExecuTorch's 112 QNN op builders
  begins `import executorch.backends.qualcomm.python.PyQnnManagerAdaptor`, and
  that package is absent from the macOS wheel -- it is a pybind extension built
  against `$QNN_SDK_ROOT`, and the SDK ships host libraries for
  `lib/x86_64-linux-clang` only. `qnn_aot_refusal()` returns that sentence with
  the module name in it rather than letting an ImportError surface from
  eighteen frames down.

## What this module will not claim

Getting the right answer out of a `.pte` is **not** evidence that it ran on the
HTP. docs/NPU2.md is this project's record of paying for that mistake twice:
the CoreML models docs/NPU.md called "executed" had run on the CPU, and every
NNAPI driver on the emulator was software. The results were correct either way.

So the artefact reader below answers a narrower question than "did it run on
the NPU" -- it answers **"could it have"**, by decoding, from upstream's own
schema, which backend each delegated segment names and which SoC and HTP
architecture its compile spec was built for. `docs/QNN.md` §6 is what would
have to be observed on a device for the wider claim, and §6.4 is why this round
does not make it.
"""

from __future__ import annotations

import importlib
import os
import platform


__all__ = [
    "QnnRefused",
    "QNN_BACKEND_ID",
    "COMPILE_SPEC_KEY",
    "host_platform",
    "executorch_version",
    "qnn_sdk_version",
    "qnn_aot_refusal",
    "qnn_aot_available",
    "soc_targets",
    "resolve_soc",
    "probe",
    "compiler_spec",
    "lower",
    "lower_cpu_reference",
    "delegation_report",
    "PteArtefact",
    "read_artefact",
    "match_device",
    "runtime_backends",
    "ExecuTorchModule",
    "QnnModule",
]


class QnnRefused(RuntimeError):
    """This host, this artefact or this runtime cannot do what was asked.

    Always carries the name of the missing thing. A refusal that says only
    "unsupported" sends the reader to the wrong place; docs/NPU.md §4 is the
    precedent -- an op with no calling convention is *absent* and named, never
    approximated.
    """


#: The `backend_id` ExecuTorch's Qualcomm backend registers itself under, and
#: the string that appears in a `.pte`'s delegate table. Read back out of the
#: artefact by `read_artefact`; this constant is what it is compared against.
QNN_BACKEND_ID = "QnnBackend"

#: The compile-spec key the QNN backend puts its serialised options under.
#: Mirrors `executorch.backends.qualcomm.utils.constants.QCOM_QNN_COMPILE_SPEC`,
#: and `_qnn_constant` below reads that module rather than trusting this copy
#: whenever executorch is importable -- this value exists so the string can be
#: named in a refusal on a host where it is not.
COMPILE_SPEC_KEY = "qnn_compile_spec"


# --------------------------------------------------------------------------
# Host capability. Every one of these measures; none of them recalls.
# --------------------------------------------------------------------------


def host_platform():
    """`(system, machine)` lowercased, e.g. `("darwin", "arm64")`."""
    return platform.system().lower(), platform.machine().lower()


def _import(name):
    try:
        return importlib.import_module(name)
    except Exception as exc:  # ImportError, and whatever a broken install raises
        raise QnnRefused(
            f"torchnative qnn: cannot import {name!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def executorch_version():
    """The installed ExecuTorch version, or a refusal naming the package."""
    et = _import("executorch")
    version = getattr(et, "__version__", None)
    if version:
        return version
    meta = _import("importlib.metadata")
    try:
        return meta.version("executorch")
    except Exception as exc:
        raise QnnRefused(
            "torchnative qnn: executorch is importable but reports no version "
            f"({type(exc).__name__}). Refusing rather than reporting 'unknown', "
            "since every version-dependent statement in docs/QNN.md is keyed to "
            "a number."
        ) from exc


def qnn_sdk_version():
    """The QNN SDK version *this ExecuTorch build pins*, read from its own script.

    Not transcribed. `executorch/backends/qualcomm/scripts/download_qnn_sdk.py`
    resolves `QNN_VERSION` from `qnn_config.sh` if that file shipped and falls
    back to a literal otherwise; reading the resulting module attribute gets
    whichever one is true for the installed wheel. Transcribing the number here
    would go stale the first time ExecuTorch bumps it, which is the mistake
    docs/DECOMP.md §12.6 named for operator lists and this is the same mistake
    one size smaller.
    """
    mod = _import("executorch.backends.qualcomm.scripts.download_qnn_sdk")
    version = getattr(mod, "QNN_VERSION", None)
    url = getattr(mod, "QNN_ZIP_URL", None)
    if not version:
        raise QnnRefused(
            "torchnative qnn: download_qnn_sdk exposes no QNN_VERSION. The "
            "installed ExecuTorch does not say which SDK it was built against."
        )
    return version, url


def _qnn_constant(name, fallback):
    """A constant from executorch's own module if importable, else the fallback."""
    try:
        mod = importlib.import_module(
            "executorch.backends.qualcomm.utils.constants"
        )
    except Exception:
        return fallback
    return getattr(mod, name, fallback)


def qnn_aot_refusal():
    """Why the QNN ahead-of-time lowering cannot run here, or `None` if it can.

    The check is an **import**, not a platform string comparison. A platform
    test would say "macOS is not on the supported list", which is true and is
    not the same claim; this says which module is missing, and it would go
    quiet by itself on a host where somebody had built it.
    """
    try:
        importlib.import_module("executorch")
    except Exception as exc:
        return (
            f"executorch is not installed ({type(exc).__name__}). The "
            "ahead-of-time half of this path is upstream ExecuTorch and there "
            "is nothing here to substitute for it."
        )
    try:
        importlib.import_module(
            "executorch.backends.qualcomm.python.PyQnnManagerAdaptor"
        )
    except Exception as exc:
        system, machine = host_platform()
        return (
            "executorch.backends.qualcomm.python.PyQnnManagerAdaptor is not "
            f"importable on this host ({system}/{machine}): "
            f"{type(exc).__name__}: {exc}. That module is a pybind extension "
            "built against $QNN_SDK_ROOT, and the Qualcomm AI Engine Direct "
            "SDK ships host libraries under lib/x86_64-linux-clang only -- "
            "ExecuTorch's own installer gates on is_linux_x86(). Every one of "
            "the QNN op builders imports it at module scope, so the lowering "
            "cannot start. Lower on a Linux x86-64 host; docs/QNN.md §3 is the "
            "procedure."
        )
    try:
        importlib.import_module(
            "executorch.backends.qualcomm.partition.qnn_partitioner"
        )
    except Exception as exc:
        return (
            "the QNN native adaptor imports but "
            "executorch.backends.qualcomm.partition.qnn_partitioner does not: "
            f"{type(exc).__name__}: {exc}"
        )
    return None


def qnn_aot_available():
    return qnn_aot_refusal() is None


def soc_targets():
    """`{name: (soc_model, htp_arch)}` read out of ExecuTorch's own tables.

    The authority is `QcomChipset` and `_soc_info_table` in
    `executorch/backends/qualcomm/serialization/qc_schema.py`. Those are plain
    Python enums with no SDK dependency, so this works on a host where the
    lowering does not -- which is the point: a refusal that can list the SoCs
    it would have accepted is more useful than one that cannot.

    docs/DECOMP.md §12.2 is the rule being followed. Copying twenty-three
    chipset names into this file would make torchnative the authority on
    Qualcomm's part numbers, and it would be wrong the first time a part is
    added.
    """
    schema = _import("executorch.backends.qualcomm.serialization.qc_schema")
    chipset = schema.QcomChipset
    table = schema._soc_info_table
    out = {}
    for member in chipset:
        info = table.get(member)
        if info is None:
            continue
        out[member.name] = (int(member.value), int(info.htp_info.htp_arch))
    return out


def resolve_soc(name):
    """`(QcomChipset, soc_model, htp_arch)` for an SoC name, or a named refusal.

    Case-insensitive, because the value this is fed comes from the device --
    `ro.soc.model` on a Snapdragon 8 Gen 3 reads `SM8650`, but the property is
    OEM-populated and its case is not guaranteed.
    """
    schema = _import("executorch.backends.qualcomm.serialization.qc_schema")
    wanted = str(name).strip().upper()
    for member in schema.QcomChipset:
        if member.name.upper() == wanted:
            info = schema._soc_info_table.get(member)
            if info is None:
                raise QnnRefused(
                    f"torchnative qnn: {member.name} is in QcomChipset but has "
                    "no entry in _soc_info_table, so its HTP architecture is "
                    "unknown and no compile spec can be built for it."
                )
            return member, int(member.value), int(info.htp_info.htp_arch)
    known = ", ".join(sorted(soc_targets()))
    raise QnnRefused(
        f"torchnative qnn: {name!r} is not a SoC ExecuTorch's QNN backend "
        f"knows. Known: {known}. The name is read from the device's "
        "`ro.soc.model` property (docs/QNN.md §5); if a real device reports "
        "one that is not on this list, the SDK and ExecuTorch both need to "
        "grow it and no lowering here can substitute."
    )


def probe():
    """Everything this host can say about its own QNN capability, measured.

    Returns a dict rather than raising, because the useful thing when the
    answer is no is the *whole* picture -- which half is missing.
    """
    system, machine = host_platform()
    out = {
        "system": system,
        "machine": machine,
        "executorch": None,
        "executorch_error": None,
        "qnn_sdk_version": None,
        "qnn_sdk_url": None,
        "qnn_aot_refusal": None,
        "qnn_aot_available": False,
        "soc_targets": 0,
        "runtime_backends": (),
        "qnn_backend_registered": False,
    }
    try:
        out["executorch"] = executorch_version()
    except QnnRefused as exc:
        out["executorch_error"] = str(exc)
        out["qnn_aot_refusal"] = qnn_aot_refusal()
        return out
    try:
        out["qnn_sdk_version"], out["qnn_sdk_url"] = qnn_sdk_version()
    except QnnRefused:
        pass
    out["qnn_aot_refusal"] = qnn_aot_refusal()
    out["qnn_aot_available"] = out["qnn_aot_refusal"] is None
    try:
        out["soc_targets"] = len(soc_targets())
    except QnnRefused:
        pass
    try:
        out["runtime_backends"] = runtime_backends()
    except QnnRefused:
        pass
    out["qnn_backend_registered"] = QNN_BACKEND_ID in out["runtime_backends"]
    return out


# --------------------------------------------------------------------------
# Ahead of time. Upstream torch and upstream executorch; torchnative is absent.
# --------------------------------------------------------------------------


def compiler_spec(soc_model, fp16=True, profile_level=0, online_prepare=False):
    """The QNN compile spec for `soc_model`, built by ExecuTorch's own helper.

    Works on a host with no SDK: `generate_qnn_executorch_compiler_spec` only
    fills and serialises a flatbuffer, and the schema module has no native
    dependency. So the spec a lowering *would* use is inspectable here even
    though the lowering is not -- which is what makes `read_artefact` testable
    on this machine.

    `fp16` selects `kHtpFp16` over `kHtpQuantized`, and it is the single most
    consequential argument in this file. docs/QNN.md §7: the HTP is not float32
    hardware, so a float32 numerical claim and an HTP execution claim exclude
    each other exactly the way docs/NPU2.md §1.1 measured for the Neural
    Engine. `kHtpQuantized` needs a calibrated quantizer as well and is a
    different round.
    """
    schema = _import("executorch.backends.qualcomm.serialization.qc_schema")
    utils = _import("executorch.backends.qualcomm.utils.utils")
    member, _, _ = resolve_soc(soc_model) if isinstance(soc_model, str) else (
        soc_model, None, None
    )
    precision = (
        schema.QnnExecuTorchHtpPrecision.kHtpFp16
        if fp16
        else schema.QnnExecuTorchHtpPrecision.kHtpQuantized
    )
    backend_options = schema.QnnExecuTorchBackendOptions(
        backend_type=schema.QnnExecuTorchBackendType.kHtpBackend,
        htp_options=schema.QnnExecuTorchHtpBackendOptions(precision=precision),
    )
    return utils.generate_qnn_executorch_compiler_spec(
        soc_model=member,
        backend_options=backend_options,
        profile_level=profile_level,
        online_prepare=online_prepare,
    )


def lower(module, example_inputs, soc_model, out_path, fp16=True,
          profile_level=0):
    """Lower `module` to a QNN-delegated `.pte`. Offline, upstream, Linux x86-64.

    Refuses by name anywhere the QNN ahead-of-time half is not importable,
    **before** touching the module -- so the failure names the SDK rather than
    surfacing as an ImportError from inside a partitioner pass.

    `module` is an eager `nn.Module` and `example_inputs` a tuple of tensors;
    the exporter is `torch.export.export` and it is **upstream's**, not this
    project's. docs/QNN.md §2 says where that boundary falls and why.
    """
    refusal = qnn_aot_refusal()
    if refusal is not None:
        raise QnnRefused(f"torchnative qnn: cannot lower here -- {refusal}")

    # No `_import("torch")` here any more: nothing in this function touches
    # `torch.` since the export moved inside executorch's helper, and
    # executorch cannot import without torch, so the check below covers it.
    utils = _import("executorch.backends.qualcomm.utils.utils")
    specs = compiler_spec(soc_model, fp16=fp16, profile_level=profile_level)
    # `to_edge_transform_and_lower_to_qnn` takes the eager MODULE and calls
    # `torch.export.export` itself (executorch/backends/qualcomm/utils/utils.py
    # :450). Exporting first and handing it the ExportedProgram raises
    # `Expected `mod` to be an instance of `torch.nn.Module``. Found by the
    # first CI run -- it cannot be found here, because `lower()` refuses on
    # every host this project can reach (docs/QNN.md §1.3).
    module = module.eval()
    edge = utils.to_edge_transform_and_lower_to_qnn(
        module, tuple(example_inputs), specs
    )
    report = delegation_report(edge)
    program = edge.to_executorch()
    with open(out_path, "wb") as handle:
        handle.write(program.buffer)
    return out_path, report


def lower_cpu_reference(module, example_inputs, out_path):
    """Lower to an XNNPACK-delegated `.pte`. **Not a QNN artefact.**

    This exists for two reasons and neither is "a fallback".

    * It is the **control**. It runs the same `torch.export` ->
      `to_edge_transform_and_lower` -> `.to_executorch()` pipeline the QNN path
      runs, so a failure in the shared machinery shows up here, on a machine
      that has no Qualcomm anything.
    * It is what makes the loading and running half of this module *executed*
      rather than merely wired. `ExecuTorchModule` runs the result on this host
      and it is compared element-wise against eager torch; `QnnModule` is then
      required to **refuse the very same file by name**, which is the check
      that the QNN narrowing is a real narrowing.

    Nothing about it is evidence for the QNN path's numerics, and
    `read_artefact(...).is_qnn` is `False` for everything it writes.
    """
    torch = _import("torch")
    exir = _import("executorch.exir")
    part = _import(
        "executorch.backends.xnnpack.partition.xnnpack_partitioner"
    )
    exported = torch.export.export(module.eval(), tuple(example_inputs))
    edge = exir.to_edge_transform_and_lower(
        exported, partitioner=[part.XnnpackPartitioner()]
    )
    report = delegation_report(edge)
    program = edge.to_executorch()
    with open(out_path, "wb") as handle:
        handle.write(program.buffer)
    return out_path, report


def delegation_report(edge_manager):
    """How much of the graph the partitioner claimed. The AOT half of §6.

    This is the number that catches the silent fallback docs/QNN.md §6.2 is
    about. QNN does not fall back at *runtime* -- an HTP that cannot initialise
    aborts the load, loudly (§6.3). What it does silently is decline nodes at
    partition time: whatever `QnnPartitioner` did not tag stays in the program
    as portable CPU kernels, the `.pte` runs, and the answer is right. A model
    that reports two delegated nodes out of nine hundred ran on the CPU, and
    nothing about its output says so.
    """
    debug = _import("executorch.devtools.backend_debug")
    graph_module = edge_manager.exported_program().graph_module
    info = debug.get_delegation_info(graph_module)
    delegated = int(info.num_delegated_nodes)
    non_delegated = int(info.num_non_delegated_nodes)
    total = delegated + non_delegated
    return {
        "subgraphs": int(info.num_delegated_subgraphs),
        "delegated_nodes": delegated,
        "non_delegated_nodes": non_delegated,
        "total_nodes": total,
        "delegated_fraction": (delegated / total) if total else 0.0,
    }


# --------------------------------------------------------------------------
# The artefact. What crosses the boundary, and what can be read off it.
# --------------------------------------------------------------------------


def _enum_value(value):
    """The integer behind an enum field, or `None` if it does not have one.

    `match_device` compares numbers rather than rendered forms because these
    fields render three ways. Measured on the installed build:
    `QnnExecuTorchBackendType.kHtpBackend` has `str()` of `"htp"`, `repr()` of
    `"<QnnExecuTorchBackendType.kHtpBackend: 2>"` and `.name` of
    `"kHtpBackend"`. A substring test over any of those is a test over a
    rendering, and `"htp"` is a substring of two of the three renderings of a
    value that is not HTP the moment somebody adds `kHtpLpaiBackend`.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    return None


def _enum_name(value):
    """A readable name for an enum-or-int-or-string field."""
    if value is None:
        return "(unset)"
    return str(getattr(value, "name", value))


class PteArtefact:
    """An ExecuTorch `.pte`, decoded far enough to say where it would run.

    Everything here is read from the file through **upstream's own** flatbuffer
    schema -- `executorch.exir._serialize._deserialize_pte_binary` for the
    program and `qc_schema_serialize.flatbuffer_to_option` for the QNN compile
    spec. There is no second parser here that could agree with the first by
    sharing a mistake, which is the same reason docs/NPU2.md §3.1 gives for
    `nnapi_runner.c` being a replayer and not a converter.

    It answers **"could this have run on the HTP"**. It does not answer "did
    it" -- nothing in a file can. docs/QNN.md §6.
    """

    def __init__(self, path, program, methods, delegates):
        self.path = path
        self.program = program
        self.methods = tuple(methods)
        #: One `(method, backend_id, compile_specs)` per delegated segment.
        self.delegates = tuple(delegates)

    def __repr__(self):
        ids = sorted({d[1] for d in self.delegates}) or ["(none)"]
        return (
            f"<PteArtefact {os.path.basename(self.path)}: "
            f"{len(self.methods)} method(s), {len(self.delegates)} delegate(s) "
            f"[{', '.join(ids)}]>"
        )

    @property
    def backend_ids(self):
        """Sorted distinct `backend_id`s, over every delegated segment."""
        return tuple(sorted({d[1] for d in self.delegates}))

    @property
    def is_qnn(self):
        """True iff at least one delegated segment names `QnnBackend`.

        Deliberately not "all". A program can legitimately mix -- QNN for what
        the partitioner claimed and portable kernels for the rest -- and a
        reader that demanded purity would call every real model False. What it
        must not do is call a program with **zero** QNN segments True, which is
        the whole silent-fallback shape.
        """
        return QNN_BACKEND_ID in self.backend_ids

    def qnn_options(self):
        """The decoded `QnnExecuTorchOptions` of each QNN segment, in order.

        Refuses by name if there are none: an empty list would read as "this
        artefact has no opinion about the HTP", when the truth is "this
        artefact has nothing to do with QNN".
        """
        key = _qnn_constant("QCOM_QNN_COMPILE_SPEC", COMPILE_SPEC_KEY)
        serialize = _import(
            "executorch.backends.qualcomm.serialization.qc_schema_serialize"
        )
        found = []
        for method, backend_id, specs in self.delegates:
            if backend_id != QNN_BACKEND_ID:
                continue
            blob = None
            for spec in specs:
                if spec.key == key:
                    blob = spec.value
                    break
            if blob is None:
                raise QnnRefused(
                    f"torchnative qnn: {os.path.basename(self.path)} method "
                    f"{method!r} has a {QNN_BACKEND_ID} segment with no "
                    f"{key!r} compile spec. The backend cannot be configured "
                    "without one, so this file was not written by the QNN "
                    "partitioner."
                )
            found.append(serialize.flatbuffer_to_option(bytes(blob)))
        if not found:
            raise QnnRefused(
                f"torchnative qnn: {os.path.basename(self.path)} has no "
                f"{QNN_BACKEND_ID} delegate. Its backends are "
                f"{list(self.backend_ids) or '(none)'}. Nothing in this file "
                "will reach a Hexagon NPU."
            )
        return found

    def htp_plan(self):
        """`{backend_type, soc_model, htp_arch, precision}` for the first segment.

        This is the artefact-level half of docs/QNN.md §6's evidence. Three of
        the four fields are what a silent fallback moves:

        * `backend_type` other than `htp` -- the compile spec asked for QNN's
          CPU or GPU reference implementation. The `.pte` still says
          `QnnBackend`, still delegates, still gets the right answer, and never
          goes near the NPU. This is the one that looks most like success.
        * `soc_model` / `htp_arch` that do not match the device -- QNN itself
          catches this at init and says so ("Arch NN set by custom config is
          different from arch associated with SoC NN"), and the load fails.
        * `precision` -- `kHtpQuantized` means the numbers are quantised and a
          float32 comparison is meaningless against them (§7).
        """
        option = self.qnn_options()[0]
        backend = option.backend_options
        htp = getattr(backend, "htp_options", None)
        precision = getattr(htp, "precision", None)
        return {
            "backend_type": backend.backend_type,
            "backend_type_name": _enum_name(backend.backend_type),
            "soc_model": int(option.soc_info.soc_model),
            "htp_arch": int(option.soc_info.htp_info.htp_arch),
            "precision": precision,
            "precision_name": _enum_name(precision),
        }


def match_device(artefact, soc_name):
    """Does this artefact's compile spec match the SoC the device reports?

    Returns `(ok, reason)`. This is the check that runs **before** anything is
    pushed, and it is worth running because the failure it prevents is loud but
    late: QNN validates the architecture at backend init and aborts the method
    load with `Arch NN set by custom config is different from arch associated
    with SoC NN` (pytorch/executorch#16465 is the same check firing on an
    unrecognised part). Catching it on the host costs an `adb getprop`;
    catching it on the device costs a push, a run and a stack trace.

    It compares `htp_arch`, not `soc_model`. Several chipsets share an
    architecture -- SM8550, SA8255, SC8380XP, SSG2115P, SSG2125P, SXR1230P and
    QCS9100 are all V73 in ExecuTorch's own table -- and a context binary built
    for one runs on another of the same generation. Comparing the part number
    would refuse combinations that work, which is a refusal that teaches the
    reader to stop believing refusals.
    """
    _, _, want_arch = resolve_soc(soc_name)
    plan = artefact.htp_plan()
    have_arch = plan["htp_arch"]
    if have_arch != want_arch:
        return False, (
            f"artefact was built for HTP v{have_arch} (soc_model "
            f"{plan['soc_model']}) but {soc_name} is HTP v{want_arch}. QNN "
            "refuses this at backend init and the method load aborts; it does "
            "not fall back."
        )
    schema = _import("executorch.backends.qualcomm.serialization.qc_schema")
    htp = int(schema.QnnExecuTorchBackendType.kHtpBackend)
    if _enum_value(plan["backend_type"]) != htp:
        return False, (
            f"artefact's backend_type is {plan['backend_type_name']}, not HTP. "
            "It will delegate, it will compute the right answer, and it will "
            "do so on QNN's CPU or GPU reference implementation. This is the "
            "fallback that looks most like success (docs/QNN.md §6.2)."
        )
    return True, (
        f"HTP v{have_arch}, backend_type {plan['backend_type_name']}, "
        f"precision {plan['precision_name']}"
    )


def read_artefact(path):
    """Decode a `.pte` into a `PteArtefact`, or refuse by name."""
    if not os.path.isfile(path):
        raise QnnRefused(f"torchnative qnn: no such artefact: {path}")
    serialize = _import("executorch.exir._serialize")
    with open(path, "rb") as handle:
        raw = handle.read()
    try:
        pte = serialize._deserialize_pte_binary(raw)
    except Exception as exc:
        raise QnnRefused(
            f"torchnative qnn: {os.path.basename(path)} is not an ExecuTorch "
            f"program: {type(exc).__name__}: {exc}"
        ) from exc
    program = getattr(pte, "program", pte)
    methods, delegates = [], []
    for plan in program.execution_plan:
        methods.append(plan.name)
        for delegate in plan.delegates:
            delegates.append((plan.name, delegate.id, tuple(delegate.compile_specs)))
    return PteArtefact(path, program, methods, delegates)


# --------------------------------------------------------------------------
# Runtime. This half is what ships; it loads a file and calls a method.
# --------------------------------------------------------------------------


def runtime_backends():
    """Backend ids registered in *this process's* ExecuTorch runtime.

    A runtime assertion, not a build-time one. `QnnBackend` appears here only
    where the runtime was linked against the QNN backend library, so its
    absence is the first thing to check when a QNN `.pte` will not load -- and
    its presence is a precondition for any claim about where a graph ran.
    """
    runtime = _import("executorch.runtime")
    registry = runtime.Runtime.get().backend_registry
    return tuple(registry.registered_backend_names)


class _Loader:
    """Mixin: hold a `.pte` and call one of its methods. Backend-agnostic.

    Split out from the QNN class because it is the half that can be *executed*
    on a developer machine. `QnnModule` narrows it to artefacts that name
    `QnnBackend`, and that narrowing is tested by requiring it to refuse a file
    this one happily runs.
    """

    def _open(self):
        if getattr(self, "_method", None) is not None:
            return self._method
        if not os.path.isfile(self.artefact_path):
            self.refuse(f"no such artefact: {self.artefact_path}")
        runtime = _import("executorch.runtime")
        program = runtime.Runtime.get().load_program(self.artefact_path)
        available = tuple(program.method_names)
        if self.method_name not in available:
            self.refuse(
                f"{os.path.basename(self.artefact_path)} has no method "
                f"{self.method_name!r}. It has {sorted(available)}."
            )
        self._program = program
        self._method = program.load_method(self.method_name)
        return self._method

    def _execute(self, args):
        method = self._open()
        outputs = method.execute(list(args))
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


# `DelegateModule` lives in a sibling module and importing it at module scope
# would make `import torchnative.export.qnn` drag in `torch`. It does not, and
# that is deliberate: `probe()` and `qnn_aot_refusal()` are the two things a
# person debugging a host wants, and they should not need a tensor library to
# answer. The two `nn.Module` classes are built on first attribute access
# instead (PEP 562), so `qnn.QnnModule` is an ordinary class to every caller.
_CLASSES = {}


def _build_classes():
    if _CLASSES:
        return _CLASSES
    npu = _import("torchnative.export.npu")

    class ExecuTorchModule(_Loader, npu.DelegateModule):
        """Any ExecuTorch `.pte`, behind an `nn.Module`.

        **Not a QNN claim.** This will happily run an XNNPACK-delegated or a
        fully-portable program, on a laptop, on the CPU. It exists so that the
        loading and calling machinery is executed rather than merely wired, and
        so that `QnnModule` has something to be a narrowing *of*.
        """

        backend_name = "ExecuTorch"

        def __init__(self, artefact_path, method_name="forward"):
            npu.DelegateModule.__init__(self)
            self.artefact_path = artefact_path
            self.method_name = method_name
            self._method = None
            self._program = None

        def forward(self, *args):
            return self._execute(args)

        def extra_repr(self):
            return (
                f"backend={self.backend_name}, "
                f"artefact={os.path.basename(self.artefact_path)}, "
                f"method={self.method_name}"
            )

    class QnnModule(ExecuTorchModule):
        """An ExecuTorch `.pte` that must carry a `QnnBackend` delegate.

        Two refusals, and both are the reason this class exists at all:

        * the artefact does not name `QnnBackend` -- so nothing in it can reach
          a Hexagon NPU, whatever it computes;
        * `QnnBackend` is not registered in this process's runtime -- so it
          could not be executed here even if it did.

        Neither is deferred to the first `forward`. docs/QNN.md §6.2: an
        artefact that quietly ran its CPU half is the failure this whole file
        is arranged against, and checking late means the model is already built
        and the caller already believes it.

        `require_runtime=False` is for inspecting an artefact on a host that
        will never execute it -- a build machine. It relaxes the *second*
        refusal only; an artefact with no QNN delegate is refused either way.
        """

        backend_name = QNN_BACKEND_ID

        def __init__(self, artefact_path, method_name="forward",
                     require_runtime=True):
            ExecuTorchModule.__init__(self, artefact_path, method_name)
            artefact = read_artefact(artefact_path)
            if not artefact.is_qnn:
                self.refuse(
                    f"{os.path.basename(artefact_path)} has no "
                    f"{QNN_BACKEND_ID} delegate -- its backends are "
                    f"{list(artefact.backend_ids) or '(none)'}. Running it "
                    "would produce correct numbers off the CPU and no output "
                    "would say so (docs/QNN.md §6.2)."
                )
            #: What the compile spec asked the HTP for. See `PteArtefact.htp_plan`.
            self.plan = artefact.htp_plan()
            self.require_runtime = require_runtime
            if require_runtime:
                registered = runtime_backends()
                if QNN_BACKEND_ID not in registered:
                    self.refuse(
                        f"{QNN_BACKEND_ID} is not registered in this "
                        f"ExecuTorch runtime (it has {list(registered)}). The "
                        "artefact names it, so this process is the wrong place "
                        "to run it -- an Android build with the QNN backend "
                        "linked in is the right one."
                    )

        def forward(self, *args):
            if not self.require_runtime:
                self.refuse(
                    "this module was built with require_runtime=False, which "
                    "is an inspection handle for a build machine. It refuses "
                    "to execute rather than falling through to a runtime that "
                    "has no QnnBackend registered."
                )
            return self._execute(args)

        def extra_repr(self):
            return (
                f"backend={self.backend_name}, "
                f"artefact={os.path.basename(self.artefact_path)}, "
                f"soc={self.plan['soc_model']}, htp=v{self.plan['htp_arch']}, "
                f"precision={self.plan['precision']}"
            )

    _CLASSES["ExecuTorchModule"] = ExecuTorchModule
    _CLASSES["QnnModule"] = QnnModule
    return _CLASSES


def __getattr__(name):
    if name in ("ExecuTorchModule", "QnnModule"):
        return _build_classes()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
