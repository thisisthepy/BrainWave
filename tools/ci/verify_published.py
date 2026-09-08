"""Does the published wheel compute, on a platform this project cannot run?

The README's `computes` row is ⚠️ for Linux and Windows. The wheels are on PyPI
and `tools/wheel/verify_linux.py` / `verify_windows.py` confirm every symbol
resolves, but resolving is not running: this project's only machine is an arm64
Mac with no Linux, no Windows and no container runtime. A hosted runner is the
missing machine, and this is what it runs.

**Every expected value here was produced on macOS arm64 from the same source.**
That is the point of hardcoding them rather than computing a tolerance: a
mismatch localises to the platform, because the arithmetic is identical.

What is deliberately **not** here: the `mps` and `vulkan` devices (Apple-only,
and these runners are Linux and Windows) and the `world_size >= 3` transport
(it spawns processes and binds loopback sockets, which is a different kind of
check from "does this wheel compute"). Neither is verified on Linux or Windows,
and this file staying quiet about them is the honest state rather than an
oversight.

Exits non-zero on the first disagreement, so the workflow fails loudly rather
than leaving a wrong number in a log nobody reads.
"""

import re
import sys


FAILURES = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name:<34} {got!r}")
    if not ok:
        print(f"      {'':<34} expected {want!r}")
        FAILURES.append(name)
    return ok


def provenance():
    """Which `torch` is this, before anything is asked of it.

    `_aten_implemented` exists only on the shim. A run that reaches upstream's
    `torch` instead -- because a dependency pulled one in and pip replaced ours
    -- would otherwise pass every check below while proving nothing, which is a
    mistake this project has already made once on its own machine.
    """
    import torch

    print("torch          :", torch.__version__)
    print("torch.__file__ :", torch.__file__)
    is_shim = hasattr(torch._C, "_aten_implemented")
    check("this is the shim, not upstream", is_shim, True)
    if not is_shim:
        print("\nSTOP: upstream torch is installed here. Nothing below would mean anything.")
        raise SystemExit(1)
    print("aten ops       :", len(torch._C._aten_implemented()))
    return torch


def kernels(torch):
    a = torch.ones(2, 3)
    b = torch.ones(3, 4)
    check("mm shape", tuple((a @ b).shape), (2, 4))
    check("mm sum", (a @ b).sum().item(), 24.0)

    lin = torch.nn.Linear(3, 4, bias=False)
    with torch.no_grad():
        lin.weight.fill_(1.0)
        out = lin(torch.ones(1, 3))
    check("nn.Linear sum", out.sum().item(), 12.0)


def promotion(torch):
    """0.0.9a0's mixed-dtype promotion, chosen for where it is easy to get wrong.

    Upstream casts each operand to the common dtype *first* and only then to the
    accumulator. Both routes label the result `float16`; only one of them says
    2047. The comparison is worse -- the result is `bool` either way, so nothing
    about the type betrays a wrong answer.
    """
    x = torch.tensor([2049], dtype=torch.int64)
    y = torch.tensor([1.0], dtype=torch.float16)
    check("sub(int64,float16) value", torch.sub(x, y).tolist(), [2047.0])
    check("sub(int64,float16) dtype", str(torch.sub(x, y).dtype), "torch.float16")

    check(
        "eq(int64,float32) at 2**24+1",
        torch.eq(torch.tensor([16777217]), torch.tensor([16777216.0])).tolist(),
        [True],
    )

    f32 = torch.tensor([0.1])
    f64 = torch.tensor([0.1], dtype=torch.float64)
    check("cat promotes to float64", str(torch.cat([f32, f64]).dtype), "torch.float64")
    check(
        "sub(f32,f64) value",
        torch.sub(f32, f64).tolist(),
        [1.4901161138336505e-09],
    )

    # And a pair upstream itself refuses, so "promotes everything" would fail here.
    try:
        torch.mm(torch.ones(2, 2), torch.ones(2, 2, dtype=torch.float64))
        check("mm refuses a mixed pair", "computed", "raised")
    except Exception as exc:
        check("mm refuses a mixed pair", type(exc).__name__ != "", True)
        print(f"      {'':<34} {str(exc).splitlines()[0][:80]}")


def model(torch):
    """A real checkpoint through real transformers -- the premise of the project.

    Greedy and short so the text is deterministic and comparable. The string is
    what macOS arm64 produced; a platform that imports and computes small
    kernels correctly can still diverge here, which is why this is separate from
    the arithmetic above.
    """
    import time

    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "HuggingFaceTB/SmolLM2-135M"
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    m.eval()
    tok = AutoTokenizer.from_pretrained(name)
    inp = tok("On-device inference is", return_tensors="pt")

    with torch.no_grad():
        m.generate(**inp, max_new_tokens=4, do_sample=False, use_cache=True)
        t0 = time.perf_counter()
        out = m.generate(**inp, max_new_tokens=24, do_sample=False, use_cache=True)
        dt = time.perf_counter() - t0

    text = tok.decode(out[0], skip_special_tokens=True)
    print(f"generated : {text!r}")
    print(f"speed     : {dt * 1000:.1f} ms for 24 tokens -> {24 / dt:.1f} tok/s")
    check(
        "generated text matches macOS arm64",
        text.startswith("On-device inference is a very powerful technique for learning from data."),
        True,
    )


def quantised(torch):
    """Load-time quantisation, which is the on-device path this exists for.

    Skipped, loudly and by name, when the installed release predates the
    feature. This runs against **published** wheels, so a check written today
    will outrun the version on PyPI until the next release -- and reporting that
    as a platform failure would be a lie about the platform. The version is
    printed so the skip cannot quietly become permanent.
    """
    import importlib.metadata as md

    from transformers import AutoModelForCausalLM

    try:
        from torchnative.quant import TorchnativeConfig
    except ImportError:
        print(
            f"SKIP  load-time quantisation -- torchnative "
            f"{md.version('torchnative')} predates TorchnativeConfig. "
            f"Not a platform result."
        )
        return

    m = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M",
        dtype=torch.float32,
        quantization_config=TorchnativeConfig("q8_0"),
    )
    kinds = {}
    for mod in m.modules():
        kinds[type(mod).__name__] = kinds.get(type(mod).__name__, 0) + 1
    check("QuantizedLinear count", kinds.get("QuantizedLinear", 0), 210)
    check("Linear left dense (lm_head)", kinds.get("Linear", 0), 1)


def _predates(feature_version):
    """True when the *installed* release is older than `feature_version`.

    The skip convention `quantised()` established, generalised: this script
    runs against **published** wheels, so a check written today outruns PyPI
    until the next upload. A version comparison is used here rather than a
    capability probe because the capability being tested -- `loss.backward()`
    -- refuses *by name* in older releases, and a probe could not tell that
    deliberate refusal apart from a regression. The installed version is
    printed either way, so the skip cannot quietly become permanent.
    """
    import importlib.metadata as md

    try:
        from packaging.version import Version
    except ImportError:  # not guaranteed present on a bare runner
        def Version(s):  # noqa: N802  -- enough to order this project's X.Y.ZaN
            m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)a(\d+)", s)
            if not m:
                raise SystemExit(f"cannot order version {s!r} without packaging")
            return tuple(int(g) for g in m.groups())

    try:
        have = md.version("torchnative")
    except md.PackageNotFoundError:
        # Not pip-installed: someone is running this against a source tree
        # (`PYTHONPATH=torchnative/src/main`), where there is no release to be
        # older than. Run the check rather than skip it -- a skip here would
        # mean the section is never exercised until after it has shipped.
        return False, "source tree (no dist metadata)"
    return Version(have) < Version(feature_version), have


def training(torch):
    """`loss.backward()` and an optimizer step -- 0.0.13a0's headline.

    New in 0.0.13a0: the eager autograd engine reached upstream's own path
    (`torch/_tensor.py` -> `_engine_run_backward` -> `_ImperativeEngine`), so
    nothing below is shim-specific -- it is the code any PyTorch user writes.

    The expected values are exact rather than tolerant, deliberately. With the
    weights filled and the input all ones, d(sum(xW^T))/dW is exactly 1 for
    every element, so `0.5 - 3 * 0.1 * 1` is a float32 identity and not a
    measurement -- a platform that disagrees here has an arithmetic fault, not
    a rounding difference. Losses fall 6.0 -> 4.8 -> 3.6 for the same reason.
    """
    old, have = _predates("0.0.13a0")
    if old:
        print(
            f"SKIP  loss.backward() training step -- torchnative {have} "
            f"predates the eager autograd engine (it refuses by name there). "
            f"Not a platform result."
        )
        return

    lin = torch.nn.Linear(3, 4, bias=False)
    with torch.no_grad():
        lin.weight.fill_(0.5)
    opt = torch.optim.SGD(lin.parameters(), lr=0.1)
    x = torch.ones(1, 3)

    losses = []
    for _ in range(3):
        opt.zero_grad()
        loss = lin(x).sum()
        loss.backward()
        losses.append(round(loss.item(), 4))
        grad_ok = lin.weight.grad is not None and lin.weight.grad.tolist()[0] == [1.0, 1.0, 1.0]
        opt.step()

    check("backward produced a .grad", grad_ok, True)
    check("loss falls under SGD", losses, [6.0, 4.8, 3.6])
    check(
        "three steps moved the weight",
        [round(v, 4) for v in lin.weight.tolist()[0]],
        [0.2, 0.2, 0.2],
    )

    # zero_grad has to actually zero, not merely complete: a loop that
    # silently does nothing passes every test that only checks it finished.
    opt.zero_grad()
    check(
        "zero_grad clears the gradient",
        lin.weight.grad is None or lin.weight.grad.abs().sum().item() == 0.0,
        True,
    )


def newer_ops(torch):
    """Operators that landed after 0.0.12a0, each skipped by name if absent.

    Per-op rather than per-release: these arrive one at a time from separate
    rounds, so asking the wheel which ones it has is more precise than asking
    how old it is. `nonzero` is the interesting one -- its output *shape*
    depends on the values, which nothing before it did.
    """
    have = set(torch._C._aten_implemented())

    cases = [
        ("aten.nonzero.default", "nonzero", lambda: torch.nonzero(torch.tensor([0.0, 3.0, 0.0, 5.0])).tolist(), [[1], [3]]),
        ("aten.fmod.Tensor", "fmod keeps the dividend's sign", lambda: torch.fmod(torch.tensor([7.0, -7.0]), 3.0).tolist(), [1.0, -1.0]),
        ("aten.maximum.default", "maximum", lambda: torch.maximum(torch.tensor([1.0, 5.0]), torch.tensor([3.0, 2.0])).tolist(), [3.0, 5.0]),
    ]
    for op, label, fn, want in cases:
        if op not in have:
            print(f"SKIP  {label:<34} -- this wheel has no {op}. Not a platform result.")
            continue
        check(label, fn(), want)


def signal_and_complex(torch):
    """What 0.0.13a0 newly contains beyond the autograd engine: `torch.fft.fftn`,
    complex tensors, `as_strided` as a read-only view, and `lstm`.

    Two different skip mechanisms, because the two groups can be asked
    different questions and the more precise question wins:

    * `as_strided` and `lstm` are ATen operators, so the wheel can be asked
      whether it has them by name -- the per-op probe `newer_ops()` established.
      That is more precise than a release comparison, since these arrive one
      round at a time rather than one release at a time.
    * `torch.fft.fftn` and complex arithmetic have **no ATen table entry to
      probe** (`fft` and `complex` are both absent from `_aten_implemented()`
      even on a wheel where they work), so those fall back to `_predates()`.
      A capability probe would be wrong here for the reason `_predates`
      documents: an older wheel refuses these *by name*, and a probe cannot
      tell a deliberate refusal apart from a regression.

    Every expected value below was produced on macOS arm64 and checked against
    **upstream torch 2.13.0 on the same host**, where all four agree exactly --
    so these are identities, not tolerances, and a platform that disagrees has
    an arithmetic fault rather than a rounding difference.
    """
    have = set(torch._C._aten_implemented())

    old, version = _predates("0.0.13a0")
    if old:
        print(
            f"SKIP  torch.fft.fftn and complex tensors -- torchnative "
            f"{version} predates them (both refuse by name there). "
            f"Not a platform result."
        )
    else:
        # A unit impulse at the origin transforms to all-ones with zero
        # imaginary part -- exact in float32, and it exercises the 2-D
        # multi-axis path rather than a single 1-D transform.
        impulse = torch.zeros(2, 2)
        impulse[0, 0] = 1.0
        spectrum = torch.fft.fftn(impulse)
        check("fftn dtype is complex64", str(spectrum.dtype), "torch.complex64")
        check("fftn of an impulse is ones", spectrum.real.tolist(), [[1.0, 1.0], [1.0, 1.0]])
        check("fftn of an impulse has no phase", spectrum.imag.tolist(), [[0.0, 0.0], [0.0, 0.0]])

        # [1,2,3,4] has an exactly-representable transform whose imaginary part
        # is antisymmetric -- a sign or conjugation error shows up here and
        # nowhere in the impulse above.
        line = torch.fft.fftn(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        check("fftn real part", line.real.tolist(), [10.0, -2.0, -2.0, -2.0])
        check("fftn imaginary part", line.imag.tolist(), [0.0, 2.0, 0.0, -2.0])

        # i*i = -1 is the one multiplication that is wrong under any
        # implementation treating a complex tensor as two independent floats,
        # which is the mistake worth catching on a platform nobody has run.
        z = torch.complex(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 2.0]))
        zz = z * z
        check("complex dtype", str(zz.dtype), "torch.complex64")
        check("(1)^2 and (2i)^2 real", zz.real.tolist(), [1.0, -4.0])
        check("(1)^2 and (2i)^2 imag", zz.imag.tolist(), [0.0, 0.0])

    if "aten.as_strided.default" in have:
        # Strides deliberately not in descending order, so an implementation
        # that ignores the stride argument and reshapes instead answers
        # [[0,1],[2,3]] and is caught.
        base = torch.arange(6.0)
        check("as_strided reads its strides",
              torch.as_strided(base, (2, 2), (1, 2)).tolist(),
              [[0.0, 2.0], [1.0, 3.0]])
    else:
        print("SKIP  as_strided                       -- this wheel has no "
              "aten.as_strided.default. Not a platform result.")

    if "aten.lstm.input" in have:
        lstm = torch.nn.LSTM(3, 4, batch_first=True)
        with torch.no_grad():
            for p in lstm.parameters():
                p.fill_(0.1)
        out, (h_n, c_n) = lstm(torch.ones(1, 2, 3))
        flat = [round(v, 6) for v in out.reshape(-1).tolist()]
        check("lstm output shape", tuple(out.shape), (1, 2, 4))
        check("lstm output values", flat, [0.17427] * 4 + [0.301514] * 4)
        # The second timestep differing from the first is what makes this a
        # recurrence rather than one affine map applied twice.
        check("lstm carried state between steps", flat[0] != flat[4], True)
        check("lstm returned h_n and c_n",
              (tuple(h_n.shape), tuple(c_n.shape)), ((1, 1, 4), (1, 1, 4)))
    else:
        print("SKIP  lstm                             -- this wheel has no "
              "aten.lstm.input. Not a platform result.")


def main():
    want_model = "--model" in sys.argv
    torch = provenance()
    if want_model:
        model(torch)
        quantised(torch)
    else:
        kernels(torch)
        promotion(torch)
        newer_ops(torch)
        training(torch)
        signal_and_complex(torch)

    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILED -- {', '.join(FAILURES)}")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
