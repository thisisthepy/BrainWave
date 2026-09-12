# The (dtype × device) matrix — what computes, what refuses, and the two names that lied

Round date 2026-09-12. Branch `work/dtypedev`, on develop `dc0bc5a`. Host Apple M1
(arm64 macOS, Metal), CPython 3.13, upstream torch 2.13.0 as the oracle,
`candle-core` 0.11.0. **No CUDA, no NPU and no Android device on this machine** —
those columns are reasoned about below and are not claimed as measured.

The ask was *"make it work on every platform, for every possible dtype, on every
possible device."* This round owns the **dtype × device** axis. The API-surface
inventory is a sibling round (`work/gaps`) and is not repeated here.

> **Answer first.**
>
> | | |
> |---|---|
> | Cells computing a **wrong** answer | **none found.** 288 (dtype, device, operator) cells were graded at *agrees* against upstream and every one of them matched. |
> | `torch.backends.mps.is_built()` | **`True`** on this artefact — `cfg!(target_vendor = "apple")`. Was a hardcoded `False`. |
> | `torch.backends.mps.is_available()` | **`True`** on this host — a live `resolve()` of an `mps` device. Was a hardcoded `False`. |
> | Cell closed this round | `float64` on `mps` now refuses by name with upstream's own sentence, on all three roads onto the device. It used to *succeed* and produce a tensor that could only be cloned. |
> | Largest cell left open | `int16` and `int32` on `mps` reach **2 of 23** operators where `int64` reaches 15. §4.2. |

---

## 1. What "works" means here, and how each cell was graded

Three grades are used throughout this repository, and this document says which
one it established for every cell it claims:

| grade | question | how it was established here |
|---|---|---|
| **builds** | can the cell be constructed at all? | `_C._tensor_from_flat(...).to(dtype)`, then `.to(device)` |
| **reaches** | does an operator on it return something rather than refuse? | 23 operators, per dtype, per device |
| **agrees** | is what it returns upstream's answer? | the same computation on upstream torch 2.13.0, compared value by value |

**A dtype that produces numbers on a device and disagrees with upstream is
worse than one that refuses, because it is silent.** That ordering is why the
matrix test grades at *agrees* and why §2's defect is treated as a defect at all:
a `False` that is merely inconvenient is still a false statement about the
machine.

Where a dtype genuinely cannot work on a device — and there are such cells,
Metal has no `double` — the requirement is that it **refuses by name**, saying
what and why, rather than silently downcasting or returning a CPU-computed
answer under a device label. That last failure mode is what
[`../graph/NPU2.md`](../graph/NPU2.md) exists about; §4.4 records where this
matrix comes closest to it.

### 1.1 What the existing harnesses already cover, and the gap this fills

`tools/golden/` diffs 304 operators against upstream across 10 dtypes and is the
authority on numeric agreement. **It has no device axis at all** — read
`tools/golden/build.py`, `cases.py` and `compare.py`: `device` appears as a
*keyword argument being exercised* (`cases.py:15267`, `kwargs={"device": "cpu"}`)
and never as a dimension of the sweep. Every case it runs is a CPU case.

So the gap is not "another comparison harness" — a second one beside the golden
harness would be a liability, and this round did not build one. The gap is the
**device column**, and `rust/torch_c/pytests/test_dtypedev.py` is 23 operators
wide rather than 304 precisely so that it stays the device test and does not
become a second opinion on operator coverage. Where the two overlap (`cpu`,
float dtypes) they agree; where they do not overlap is where this file is the
only thing looking.

`docs/numerics/DTYPE.md`, `BF16.md`, `BOOL.md`, `INT8.md`, `FLOAT8*.md`,
`PROMOTE.md` and `AGREE.md` own the per-dtype numerics on the CPU. This file
cites them rather than re-deriving them, and adds only what changes when a
device is involved.

---

## 2. `torch.backends.mps.is_available()` — the shim computed on Metal and said it could not

### 2.1 The measurement

Through the vendored tree, with this shim:

```
>>> a = torch.randn(64, 64).to('mps'); b = torch.randn(64, 64).to('mps')
>>> (a @ b).device
device(type='mps', index=0)          # it really computed on Metal
>>> torch.backends.mps.is_available()
False
>>> torch.backends.mps.is_built()
False
```

Both halves were constants:

```python
module._mps_is_available = _constant_function("torch._C._mps_is_available", False)
_BUILD_FLAGS = { "_has_mps": False, ... }
```

each justified by a comment saying candle's `metal` feature was off in
`Cargo.toml`. **That comment had gone stale, and it was verified stale before
anything was changed rather than taken on trust:** `Cargo.toml:170` reads
`features = ["metal"]` for the Apple-target entry, `PyDevice::resolve` in
`device.rs` has carried an `mps` arm under `#[cfg(target_vendor = "apple")]` for
several rounds, and the matmul above lands on `mps:0` with upstream's numbers.
[`../devices/DEVICE_NS.md`](../devices/DEVICE_NS.md) §6 had already recorded the
disagreement, measured it, and deliberately left it — on the ground that fixing
it changes behaviour for every caller and reached past that round's request.
This is that round.

**Why it is not cosmetic.** `torch.backends.mps.is_available()` is *the*
standard gate: transformers, accelerate, and essentially all user code that
picks a device branch on it. So a false `False` there does not merely misreport
— it means **nothing built on top of this shim will ever select the device the
shim is already able to use.** Every model load silently lands on the CPU on a
machine with a working GPU.

### 2.2 They are two questions and they now get two answers

Upstream's own docstring for `is_built()` says it *"doesn't necessarily mean MPS
is available; just that if this PyTorch binary were run a machine with working
MPS drivers and devices, we would be able to use it."* Conflating them would be
wrong even on a host where they agree.

| name | question | answer here | source |
|---|---|---|---|
| `torch.backends.mps.is_built()` → `torch._C._has_mps` | **Was Metal compiled into this artefact?** | `True` on this Apple build; `False` on the Android, Linux and wasm artefacts | `cfg!(target_vendor = "apple")` — a compile-time constant, not probeable, and exactly the predicate that gates both candle's `metal` feature in `Cargo.toml` and the `mps` arm of `resolve()` |
| `torch.backends.mps.is_available()` → `torch._C._mps_is_available()` | **Did a Metal device actually open, here, now?** | `True` on this host | a live `PyDevice{kind:"mps"}.resolve()` |

`built` is **necessary and not sufficient**. The case that makes the distinction
real — an Apple build where `available` is `False` — is a Mac VM with no GPU
passthrough, or any host where `MTLCreateSystemDefaultDevice` returns nil. **This
machine is not one and therefore cannot demonstrate it**; what is demonstrable
here, and is asserted, is that the two come from different places and that
`available ⇒ built`.

Both are exposed through one new Rust entry point, `torch._C._mps_probe()`,
built as `_cuda_probe()`'s twin and with the same contract: **it never raises.**
A caller asking "is there a GPU here" should not have to catch anything, so a
refusal becomes `available: false` with `reason` (`"not_built"` or
`"no_device"`) and `error` carrying the refusal text.

`is_available()` re-probes on every call rather than capturing a constant at
import — a captured constant is exactly what was wrong before — and that costs
nothing, because `metal_device` already caches the `MetalDevice` per index for
the life of the process.

### 2.3 What flipping `_has_mps` breaks, and why the fix had to include it

`torch/mps/__init__.py:67` is:

```python
def manual_seed(seed):
    if not torch._C._has_mps:
        return
    _get_default_mps_generator().manual_seed(seed)
```

and `torch.manual_seed` reaches it unconditionally through `torch/random.py`.
While `_has_mps` was `False`, **that guard was the only thing keeping every
model script's `torch.manual_seed(0)` off `_mps_get_default_generator`**, which
was an `_Unimplemented`. Measured, by setting the flag and changing nothing
else:

```
NotImplementedError: not implemented in torch._C shim: torch._C._mps_get_default_generator
```

So the honest flag is only honest with a generator behind it, and
`_install_mps_backend` installs both in one place.

**The generator it installs is `torch.default_generator` itself, and that is a
claim about this build rather than a shortcut.** There is one RNG stream here
(`rng.rs`, one process-wide `CpuGenerator`) and every `mps` tensor with random
contents is filled from it: `torch.randn(4).to("mps")` draws on the host and
moves the bytes, and `torch.randn(4, device="mps")` does not work at all yet
(§4.3). There is no second stream for a separate object to own, so returning a
distinct `Generator` would be inventing a state that nothing advances.

The divergence this leaves is named rather than hidden: **`torch.mps.manual_seed(k)`
on its own also reseeds the CPU stream**, where upstream would leave it alone.
`torch.manual_seed(k)` — the call that actually matters — passes the same seed
to both and is unaffected. `_mps_get_default_generator(index=n)` for `n != 0`
refuses by name rather than handing back device 0's stream.

### 2.4 What was deliberately *not* changed

`torchnative.device.mps.availability()` still measures by allocation and still
carries the constant alongside as `detail["declared"]` /
`detail["declared_disagrees"]`. It was **not** rewritten to read the now-correct
constant. "The declared value happens to be right today" is not a reason to stop
measuring — it is precisely how the stale comment survived for as long as it
did. `declared_disagrees` is `False` there now, which is the field doing its job.

---

## 3. The matrix

23 operators — `add sub mul div matmul sum mean max neg abs exp sqrt pow2
softmax cumsum argmax eq where cat clone index0 to_f32 transpose_contig` — over
a 2×2 of the values 1, 2, 3, 4, which every dtype in the table can hold exactly
(including `float8_e4m3fn`, whose grid carries all four, and `bool`, where all
four are `True`). An input a dtype could not represent would make a
disagreement about the *input* look like a disagreement about the operator.

### 3.1 Dtypes this build can store, by device

Every number below was measured on this host on 2026-09-12, against the artefact
built from this tree. "upstream" is torch 2.13.0 in the same interpreter.

| dtype | cpu reaches / 23 | mps reaches / 23 | upstream cpu / 23 | upstream mps / 23 |
|---|---|---|---|---|
| `float64` | 21 | **0** | 23 | **0** |
| `float32` | 21 | 16 | 23 | 23 |
| `float16` | 21 | 15 | 23 | 23 |
| `bfloat16` | 21 | 15 | 23 | 23 |
| `int64` | 18 | 15 | 21 | 21 |
| `int32` | 18 | **2** | 21 | 21 |
| `int16` | 18 | **2** | 21 | 21 |
| `uint8` | 18 | 15 | 21 | 21 |
| `uint32` | 17 | 14 | **13** | **8** |
| `bool` | 12 | 10 | 16 | 16 |
| `float8_e4m3fn` | 7 | 2 | 10 | 3 |

**Grade: `agrees`, for every cell in the "reaches" columns.** Of the 288
(dtype, device, operator) cells where both this build and upstream produce a
value, **all 288 match** within one float32 ULP applied relatively. The twelve
cells that differ in their last bits are all `exp` and `softmax` — a rounding
direction, not an operator — and all sit inside the tolerances
`tools/golden/dtypes.py` already sets for those dtypes.

`_CPU_REACHES` and `_MPS_REACHES` in `test_dtypedev.py` freeze the *exact set*
of cells that compute, rather than these counts. Both directions of change are
red: a cell that stops computing is a lost capability, and a cell that starts
computing is a capability nothing has yet compared against upstream — it gets
graded the moment the set is updated to admit it. Counts alone would hide a
swap.

Two rows are worth reading carefully:

* **`uint32` is a row where this build reaches *more* than upstream** (17 vs 13
  on cpu, 14 vs 8 on mps). Upstream refuses `uint32` arithmetic outright
  (`"add_stub" not implemented for 'UInt32'` on the CPU, `Failed to create
  function state object for: add_dense_uint_uint` on Metal); candle has the
  kernels and this build runs them. That is a *widening*, not a divergence —
  every `uint32` value it produces was checked against upstream's `int64`
  ground truth by hand and agrees — but it is the one row where "agrees with
  upstream" cannot be asserted for the whole row, because upstream has no
  answer to agree with.
* **`float64` on `mps` is `0` on both sides now.** It was `3` on this side
  before this round. §4.1.

### 3.2 Dtypes this build cannot store at all

Eleven of the dtypes `torch` publishes have no candle storage here. **Grade:
refuses by name** — `_build` on each raises naming the dtype, and that is
asserted for all eleven:

| dtype | upstream cpu | upstream mps | note |
|---|---|---|---|
| `int8` | works (22/23) | works (22/23) | the one on this list most likely to surprise. [`INT8.md`](INT8.md) / [`INT8B.md`](INT8B.md) are that ground |
| `uint16`, `uint64` | partial | partial | upstream also refuses most arithmetic on these |
| `float8_e5m2`, `float8_e4m3fnuz`, `float8_e5m2fnuz`, `float8_e8m0fnu` | partial | partial | [`FLOAT8.md`](FLOAT8.md), [`FLOAT8B.md`](FLOAT8B.md), [`FLOAT8C.md`](FLOAT8C.md) — only `float8_e4m3fn` has candle storage |
| `float4_e2m1fn_x2` | refuses | refuses | both sides refuse |
| `qint8`, `quint8`, `qint32` | refuses (needs a quantizer) | refuses | `quant.rs` is a separate road |

`complex32` / `complex64` / `complex128` are deliberately outside this table:
this shim holds a complex tensor as a *pair of real tensors* and reading its
values refuses by name on both devices. That is a different axis with its own
documents, and putting it in a device matrix would misattribute a
representation gap to a device.

### 3.3 Devices that could not be measured on this machine

Stated per row rather than left blank, because a blank in a matrix reads as a
pass:

| device | status here | what would settle it |
|---|---|---|
| `cpu` | **measured**, all three grades | — |
| `mps` | **measured**, all three grades | — |
| `cuda` | **not measured — no NVIDIA GPU on this machine.** The `cuda` arm of `resolve()` *is* live on this build (candle's dummy backend compiles on every target) and refuses with `reason: not_built`, which is exercised by `test_cuda.py`. What is unmeasurable is every cell *past* that refusal. | any host with an NVIDIA GPU and a `cuda`-feature build. [`../devices/CUDA.md`](../devices/CUDA.md) |
| `vulkan` | **not measured.** `resolve()` refuses it structurally — candle's closed `Device` enum has no Vulkan variant ([`../devices/VULKAN2.md`](../devices/VULKAN2.md) §5.1) — so there are no cells, not unmeasured cells. | a backend, not a machine |
| `npu` | **not measured — no NPU reachable here.** `torchnative.device.npu` probes and refuses. | an Intel/AMD/Qualcomm NPU host. [`../devices/NPUVENDOR.md`](../devices/NPUVENDOR.md) |
| Android (`androidNative`) | **not measured — no device, and this round was forbidden to touch one.** Note the artefact question is different there: `_has_mps` is `False` on that build *by construction*, and the `mps` arm of `resolve()` does not exist. | an arm64 Android device running the pypackpack bundle |

---

## 4. Cells: closed, and left

### 4.1 Closed — `float64` on Metal now refuses by name

**Metal has no `double`.** That is a property of the API — MSL has no 64-bit
floating type — not of candle and not of this build. Upstream refuses at the
boundary:

```
TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework
doesn't support float64. Please use float32 instead.
```

**This build did not.** `x.double().to("mps")` *succeeded*, because candle will
happily allocate an `F64` Metal buffer. What came back was a tensor that could
be cloned and nothing else:

| call | before |
|---|---|
| `a + a` | `candle: Metal error Error while loading function: badd_f64` |
| `a.sum()` | `candle: Metal contiguous reduce op Sum F64 not implemented` |
| `a @ a` | `candle: Metal error mlx matmul doesn't support F64` |
| `a.to(torch.float32)` | `candle: Metal contiguous to_dtype F64 F32 not implemented` |

The last row is the one that makes this more than an inconvenience: **the
documented escape hatch was closed too.** The object could be made and could not
be converted back, and every message named an internal candle symbol rather than
the fact. This is [`../graph/NPU2.md`](../graph/NPU2.md)'s failure mode in its
quieter form — not a wrong number, but **a capability claim made by construction
succeeding**.

Now, on all three roads onto the device, with upstream's own sentence:

| road | now |
|---|---|
| `torch.zeros(2).double().to("mps")` | `TypeError: Cannot convert a MPS Tensor to float64 dtype …` |
| `torch.zeros(2).to("mps").to(torch.float64)` | same |
| `torch.zeros(2, dtype=torch.float64, device="mps")` | same |

The gate has two halves and each covers a different road:

* `device::metal_dtype_gate` is called from **`PyTensorBase::new`**, which is
  the one constructor every dense tensor in this crate passes through (there
  are exactly two `Repr::Dense` constructors and the other, `boolean`, refuses
  anything that is not `U8`). It was put there rather than at any of the **106
  call sites** that turn a dtype into candle storage, because a gate on one of
  those is a gate with 105 ways round it.
* A second call in `aten._to_copy.default` fires *before* candle does, so
  `move_then_cast` gets the fact rather than `Metal contiguous to_dtype F32 F64
  not implemented`.

### 4.2 Left, and the largest one — `int16` / `int32` on Metal

`int64` reaches 15 of 23 operators on `mps`. `int32` and `int16` reach **2**
(`clone`, `index0`; `cat` in the wider sweep). Upstream reaches 21 for all three.
Every one of the 19 refusals is by name and names the dtype, so this is a
capability gap and not a correctness risk:

```
aten.add.Tensor: candle: Metal error Error while loading function: badd_i32
aten.sum.default: candle: Metal contiguous to_dtype I32 I64 not implemented
aten.contiguous.default: candle: Metal copy_strided I32 not implemented
aten.matmul.default: candle: Metal error mlx matmul doesn't support I32
```

**What it would take:** the missing kernels are candle's, not this crate's —
`candle-metal-kernels` ships `badd_i64`/`badd_u32` and not `badd_i16`/`badd_i32`,
and the `to_dtype` matrix has no `I16→I64` or `I32→I64` arm. Closing it means
either writing Metal shaders into `candle-metal-kernels` (upstream patch, of the
shape `vendor/int8-candle-0.11.0-cpu.patch` already is for another gap), or a
promotion rule in this crate that widens `int16`/`int32` to `int64` before
dispatch on Metal and narrows after. **The second is cheaper and is the one that
needs the argument written first**, because a silent widen-compute-narrow is
exactly the shape this document spends §1 warning about: it would have to be
provably wrap-identical to the narrow computation (it is, for `+`/`-`/`*` in
two's complement; it is not, for anything that saturates) and it would have to
be refused rather than applied wherever it is not. **Not started this round.**

### 4.3 Left — `torch.randn(..., device="mps")`

```
>>> torch.randn(4, device='mps')
RuntimeError: aten.normal_.default: candle: Metal contiguous to_dtype F64 F32 not implemented
```

`torch.randn(4).to("mps")` works. The fill path draws in `float64` on the host
and converts, and Metal has no `F64→F32`. Fixing it means giving `normal_` a
`float32` path when the target device is Metal, which is small — but it is an
RNG change and `RNG.md` §1.1's bit-for-bit agreement with upstream's stream is
the thing that must not move, so it is listed rather than done. **Not started
this round.**

### 4.4 Left — refusal messages that name `_cpu` on an `mps` tensor

The nearest thing to a misleading result in the whole matrix, and it is a
message rather than a value:

```
>>> torch.zeros(2, dtype=torch.float8_e4m3fn).to('mps').neg()
NotImplementedError: "neg_cpu" not implemented for 'Float8_e4m3fn'
```

`"sum_cpu"`, `"exp_vml_cpu"`, `"argmax_cpu"`, `"add_stub"` do the same. These are
**dtype-level refusals raised before device dispatch**, and the strings are
upstream's own CPU messages transcribed verbatim — so they are accurate about
what upstream would say and misleading about where the tensor is. Nothing
computes a wrong value; a reader is sent to the wrong place. Renaming them
touches every dtype refusal in `aten.rs` and belongs to whoever owns that
family, not to this round. **Not started this round.**

---

## 5. Wrong answers

**None found.** This is the result this round was most looking for, and it is a
negative one, so it is worth saying exactly what was and was not looked at.

What was checked: 288 (dtype, device, operator) cells across 11 storable dtypes,
2 devices and 23 operators, each compared value-by-value against upstream torch
2.13.0 computing the same thing. Upstream was asked on the **cpu** for every
cell including the `mps` ones, deliberately — upstream refuses several of these
on Metal that it computes on the CPU, and the question here is whether this
build's number is *right*, not whether upstream would have produced it on that
device.

What that cannot see:

* **Any dtype × device pair outside this machine.** §3.3.
* **Operators outside the 23.** `tools/golden/` covers 304 on the CPU; **no
  harness covers them on `mps`**, and extending the golden harness with a device
  axis is the obvious next move and is larger than this round.
* **Values outside 1, 2, 3, 4.** A kernel wrong only at a boundary — overflow,
  subnormal, NaN propagation, the `i64::MIN` wrap `neg_default` is careful about
  — passes everything here. The per-dtype documents in this directory are where
  that ground is covered on the CPU.

---

## 6. Nullification — what was broken, and whether the tests noticed

Each guarantee was nullified in the source, rebuilt with
`vendor/install_shim.sh`, and the suite re-run. Recorded because a check that
cannot fail is not a check.

| # | nullification | caught by | what was seen |
|---|---|---|---|
| A | `_mps_is_available` back to `return False` | `test_mps_is_available_is_a_probe_and_not_a_constant` | *"torch.backends.mps.is_available() is False but torch.empty(2, 2, device='mps') succeeded"* — the original defect, reproduced and caught |
| B | `_has_mps = False` instead of `probe["built"]` | `test_mps_is_built_is_the_artefacts_own_answer` | *"is_built() is False on sys.platform='darwin' … A False here is the constant coming back"* |
| C | `metal_dtype_gate` short-circuited to `Ok(())` | `test_float64_refuses_by_name_on_every_road_onto_metal`, `test_no_float64_tensor_can_exist_on_a_metal_device`, **and** `test_the_mps_column_reaches_what_it_is_recorded_as_reaching` | *"cast_then_move produced 'made a torch.float64 tensor on mps'"*; the frozen column caught it independently, which is the point of freezing it |
| D | integral `neg_default` computes `1 - x` instead of `0 - x` | `test_the_dtype_device_matrix_agrees_with_upstream` | *"6 (dtype, device, operator) cells compute a value that is not upstream's"* — the wrong-value detector fires, on both devices |

**A and B are separate nullifications on purpose.** Doing them together would
have shown two red tests and proved neither: `is_built` and `is_available` are
different questions (§2.2), and a test suite that cannot tell which one broke
has collapsed them again.

Note what B *did not* break: `is_available()` stayed `True`, because it does not
read `_has_mps`. That is the two-questions property holding under a fault, which
is a stronger statement than the two of them agreeing when nothing is wrong.

---

## 7. What changed

| kind | what |
|---|---|
| **defect fixed** | `torch.backends.mps.is_available()` and `is_built()` answered `False` on a machine computing on Metal. Both now answer from the artefact. |
| **defect fixed** | `float64` on `mps` constructed successfully and produced an unusable tensor whose only escape hatch was also closed. Now refuses by name on all three roads. |
| **feature added** | `torch._C._mps_probe()`, `torch._C._mps_get_default_generator()`. |
| **tests added** | `rust/torch_c/pytests/test_dtypedev.py` — 11 tests, 4 verified nullifications. |
| **documents corrected** | [`../devices/DEVICE_NS.md`](../devices/DEVICE_NS.md) §6 and the `torchnative/device/__init__.py` module docstring both described the constant in the present tense; both now say what closed it and neither had its *measurement* rewritten. |
| **deleted** | nothing. |

No operator kernels were added this round. The `reaches` counts in §3.1 are
unchanged from before it except for `float64`/`mps`, which went from 3 to 0 —
**a capability was removed, on purpose**, because those three were `construct`,
`clone` and `index0` on a tensor nothing could compute with.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/device.rs metal_dtype_gate present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/device.rs mps_probe present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _install_mps_backend present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tensor.rs metal_dtype_gate present -->
