"""GGML block formats, reimplemented from the format rather than from candle.

**This file is the verification axis.** docs/graph/QUANT2.md §2 is the argument for
it; what follows is the argument in short.

Every other judgement in this repository is bit equality against upstream.
Quantisation cannot be judged that way -- it is lossy by construction, at 7.5%
relative RMS for Q4K on random weights (docs/graph/QUANT.md §7) -- so "close to
upstream" is the only thing a comparison against upstream could ever say, and a
tolerance wide enough to admit that is wide enough to admit a wrong kernel.

The way out is to stop comparing the *result* and start comparing the
*function*. A GGML block format is a completely specified map from `float32` to
bytes: block size, scale derivation, rounding mode, packing order, all fixed.
So it can be written down twice and the two spellings compared exactly:

  * `quantize_q8_0` / `quantize_q4_0` produce the **bytes**, and the test
    compares them to `_C._quantized_blob()` byte for byte.
  * `dequantize_*` produce **float32**, and the test compares them to
    `_C._dequantize()` bit for bit, over the same blob.

Nothing here has a tolerance in it.

**What this cannot catch.** Both spellings were written by reading the same
format. If GGML's own definition were misread the same way twice, both would
agree and both would be wrong -- so this checks that candle implements the
format this file describes, and that the two do not drift apart. It does *not*
independently establish that either matches `llama.cpp`. The line references in
each function are to the upstream C this file was transcribed from (the same
ones candle's own source cites), so that claim is at least checkable by hand.
A GGUF file written by `llama.cpp` compared against `_quantized_blob` would
close it, and there is none on this host (docs/graph/QUANT2.md §8).

**Float semantics are explicit.** Every arithmetic step that Rust performs in
`f32` is rounded back to `f32` here through `struct`, because Python's floats
are `f64` and an unrounded intermediate is the one difference that would show
up as a spurious failure and get "fixed" with a tolerance. `f16` goes through
struct's `'e'` format, which is round-to-nearest-even -- the same rule
`half::f16::from_f32` uses.
"""

import struct

QK8_0 = 32
QK4_0 = 32
QK_K = 256
K_SCALE_SIZE = 12

# Bytes per block. Asserted against candle's own `type_size()` in the tests, so
# a format change on either side is a failure rather than a silent reinterpret.
TYPE_SIZE = {
    "q8_0": 2 + 32,  # f16 d, 32 x i8
    "q4_0": 2 + 16,  # f16 d, 32 x 4-bit packed two to a byte
    "q4_k": 2 + 2 + K_SCALE_SIZE + QK_K // 2,  # f16 d, f16 dmin, 12 scale bytes, 128 nibble bytes
}
BLOCK_SIZE = {"q8_0": QK8_0, "q4_0": QK4_0, "q4_k": QK_K}


# --- float semantics --------------------------------------------------------


def f32(x):
    """Round a Python float to the nearest `float32`, as every Rust `f32`
    operation does to its result."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f16_bytes(x):
    """`half::f16::from_f32`, little endian.

    struct's `'e'` is IEEE binary16 with round-to-nearest-even, which is what
    `half` does. It raises on overflow where `half` saturates to infinity, so
    that case is handled rather than left to raise -- a weight block whose
    absmax exceeds 65504 * 127 is not reachable from real weights, but the
    reference should not be the thing that falls over if one arrives.
    """
    try:
        return struct.pack("<e", x)
    except OverflowError:
        return struct.pack("<e", float("inf") if x > 0 else float("-inf"))


def f16_from_bytes(b):
    return f32(struct.unpack("<e", b)[0])


def rust_round(v):
    """`f32::round` -- half away from zero, *not* Python's banker's rounding.

    `round(0.5)` is 0 in Python and 1 in Rust, and a block of a quantised
    weight lands on a tie often enough that using the wrong one shows up as a
    handful of off-by-one quants rather than as a visible break.
    """
    if v >= 0:
        return float(int(v + 0.5)) if v - int(v) != 0.5 else float(int(v) + 1)
    return -rust_round(-v)


def as_i8(v):
    """Rust's `f32 as i8`: saturating, NaN to zero (since Rust 1.45)."""
    if v != v:
        return 0
    if v >= 127.0:
        return 127
    if v <= -128.0:
        return -128
    return int(v)


def as_u8(v):
    """Rust's `f32 as u8`: truncate toward zero, saturating at both ends.

    The saturation at 0 is load-bearing for Q4_0, whose quantiser leans on it:
    `(x0 + 8.5) as u8` is how a value below -8.5 becomes nibble 0 rather than
    wrapping.
    """
    if v != v:
        return 0
    if v >= 255.0:
        return 255
    if v <= 0.0:
        return 0
    return int(v)


# --- Q8_0 -------------------------------------------------------------------
#
# llama.cpp `quantize_row_q8_0_reference` / `dequantize_row_q8_0`
# (ggml.c L1619 in the revision candle cites).
#
# One f16 scale and 32 signed bytes per 32 elements. The scale is the block's
# absmax over 127, so the reconstruction is `q * d` and the largest-magnitude
# element of every block is representable to within one rounding step.


def quantize_q8_0(values):
    out = bytearray()
    for i in range(0, len(values), QK8_0):
        block = [f32(v) for v in values[i : i + QK8_0]]
        amax = 0.0
        for v in block:
            amax = max(amax, abs(v))
        d = f32(amax / 127.0)
        inv = f32(1.0 / d) if d != 0.0 else 0.0
        out += f16_bytes(d)
        for v in block:
            out.append(as_i8(rust_round(f32(v * inv))) & 0xFF)
    return bytes(out)


def dequantize_q8_0(blob, n):
    out = []
    off = 0
    for _ in range(n // QK8_0):
        d = f16_from_bytes(blob[off : off + 2])
        qs = struct.unpack("<32b", blob[off + 2 : off + 34])
        out.extend(f32(q * d) for q in qs)
        off += 34
    return out


# --- Q4_0 -------------------------------------------------------------------
#
# llama.cpp `quantize_row_q4_0_reference` / `dequantize_row_q4_0` (ggml.c
# L1525). One f16 scale and 16 bytes of packed nibbles per 32 elements.
#
# Two things here are easy to get wrong from the name alone and are the reason
# this is transcribed rather than inferred:
#
#   * the scale is `max / -8`, taken from the element with the largest
#     *absolute* value but keeping its **sign**, so `d` is negative for a block
#     whose extreme is positive;
#   * the packing is not adjacent pairs. Element `j` goes in the low nibble of
#     byte `j` and element `j + 16` in the high nibble, so the two halves of a
#     block are interleaved across the 16 bytes.


def quantize_q4_0(values):
    out = bytearray()
    for i in range(0, len(values), QK4_0):
        block = [f32(v) for v in values[i : i + QK4_0]]
        amax, mx = 0.0, 0.0
        for v in block:
            if amax < abs(v):
                amax = abs(v)
                mx = v
        d = f32(mx / -8.0)
        inv = f32(1.0 / d) if d != 0.0 else 0.0
        out += f16_bytes(d)
        for j in range(QK4_0 // 2):
            x0 = f32(block[j] * inv)
            x1 = f32(block[QK4_0 // 2 + j] * inv)
            xi0 = min(15, as_u8(f32(x0 + 8.5)))
            xi1 = min(15, as_u8(f32(x1 + 8.5)))
            out.append(xi0 | (xi1 << 4))
    return bytes(out)


def dequantize_q4_0(blob, n):
    out = [0.0] * n
    off = 0
    for i in range(n // QK4_0):
        d = f16_from_bytes(blob[off : off + 2])
        qs = blob[off + 2 : off + 18]
        for j in range(QK4_0 // 2):
            x0 = (qs[j] & 0x0F) - 8
            x1 = (qs[j] >> 4) - 8
            out[i * QK4_0 + j] = f32(x0 * d)
            out[i * QK4_0 + j + QK4_0 // 2] = f32(x1 * d)
        off += 18
    return out


# --- Q4_K -------------------------------------------------------------------
#
# llama.cpp `dequantize_row_q4_K` (k_quants.c L928 neighbourhood).
#
# **Only the reader is reimplemented.** The Q4K *quantiser* is an iterative
# least-squares search (`make_qkx3_quants`, weighted, with a sweep over
# candidate scale/min pairs) and transcribing an optimiser is transcribing a
# search, not a format -- a reference that disagreed would not say which side
# was wrong. So the test drives this from the blob candle produced and checks
# the reconstruction, which is the half that *is* a format.
#
# A 256-element super-block carries two f16 super-scales (`d`, `dmin`), eight
# 6-bit sub-scales and eight 6-bit sub-minima packed into 12 bytes, and 128
# bytes of nibbles. `get_scale_min_k4` is that 6-bit unpacking: the first four
# pairs sit in the low six bits of bytes 0-3 and 4-7, and the last four are
# split, taking their low four bits from bytes 8-11 and their high two from the
# top of bytes 0-7.


def get_scale_min_k4(j, scales):
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)
    m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return d, m


def dequantize_q4_k(blob, n):
    out = []
    off = 0
    for _ in range(n // QK_K):
        d = f16_from_bytes(blob[off : off + 2])
        dmin = f16_from_bytes(blob[off + 2 : off + 4])
        scales = blob[off + 4 : off + 4 + K_SCALE_SIZE]
        qs = blob[off + 4 + K_SCALE_SIZE : off + TYPE_SIZE["q4_k"]]
        is_ = 0
        for j in range(0, QK_K, 64):
            q = qs[j // 2 : j // 2 + 32]
            sc, m = get_scale_min_k4(is_, scales)
            d1, m1 = f32(d * sc), f32(dmin * m)
            sc, m = get_scale_min_k4(is_ + 1, scales)
            d2, m2 = f32(d * sc), f32(dmin * m)
            out.extend(f32(f32(d1 * (b & 0xF)) - m1) for b in q)
            out.extend(f32(f32(d2 * (b >> 4)) - m2) for b in q)
            is_ += 2
        off += TYPE_SIZE["q4_k"]
    return out


DEQUANTIZERS = {
    "q8_0": dequantize_q8_0,
    "q4_0": dequantize_q4_0,
    "q4_k": dequantize_q4_k,
}
QUANTIZERS = {"q8_0": quantize_q8_0, "q4_0": quantize_q4_0}


# --- the bound the lossy part is held to ------------------------------------


def round_trip_bound(fmt, block):
    """The largest reconstruction error the format *permits* for one block.

    **Derived, with no fudge factor.** The reconstruction is not `round(x/d)*d`
    -- the quantiser divides by the `float32` scale `d` and the dequantiser
    multiplies by the `float16` one, `d'`. So the error has two parts and both
    have to be written down:

        x - q*d'  =  (x - q*d)  +  q*(d - d')
                     |________|    |_________|
                     rounding      the f16 storage of the scale
                     <= d/2        <= |q|max * eps(d)

    `eps(d)` is half an ulp of binary16: a relative `2**-11` for a normal `d`,
    or an absolute `2**-25` if `d` fell into the subnormal range, whichever is
    larger. `|q|max` is 127 for Q8_0 and 8 for Q4_0, and it is the term that
    dominates -- it makes the Q8_0 bound `0.562*d` rather than `0.5*d`, which
    is a 12% difference and *was* measured as a violation of the naive `d/2`
    (block 384 of the Gaussian fixture, 0.0077395 against 0.0076268). The bound
    is written from the arithmetic rather than widened until it passed.

    This is the *ceiling*. The floor is in the test, and it is the half that
    can actually fail for the interesting reason -- a "quantiser" that returned
    its input would sail under any ceiling.
    """
    amax = max(abs(f32(v)) for v in block) if block else 0.0
    if fmt == "q8_0":
        # Symmetric grid: `q` spans -127..127 and `d = amax/127`, so the
        # block's extreme lands on a grid point either way and the only error
        # inside the block is rounding.
        d, qmax, steps = f32(amax / 127.0), 127, 0.5
    elif fmt == "q4_0":
        # **Asymmetric grid, and this is a property of the format rather than
        # slack in the bound.** The nibble is `q - 8` for `q` in 0..15, so it
        # spans -8..+7, and `d = max / -8` is derived from the element with the
        # largest *absolute* value keeping its sign. That element lands exactly
        # on -8 steps. An element of the opposite sign at the same magnitude
        # can only reach +7 steps, so it is off by a **whole** step, not half
        # of one -- measured, block 64 of the Gaussian fixture, 0.1550 against
        # a naive half-step bound of 0.1206.
        #
        # This is why Q4_0 is worse than Q4K at the same four bits (8.5%
        # against 7.5% relative RMS, docs/graph/QUANT.md §7): the k-quant carries a
        # per-sub-block minimum and so has no wasted end.
        d, qmax, steps = f32(amax / 8.0), 8, 1.0
    else:
        raise KeyError(f"no derived bound for {fmt}; k-quants are checked by blob, not by bound")
    # Half an ulp of binary16 at `d`: relative while `d` is normal (the
    # smallest normal is 2**-14), absolute below that.
    eps = max(d * 2.0**-11, 2.0**-25)
    return d * steps + qmax * eps
