"""Guards for the claims docs/verification/REMEASURE2.md corrected in README.md.

Most of that round's corrections already had a test holding them --
`test_collect2.py` for the collectives, `test_train.py` for the convolution
backward rule and the `native_layer_norm` derivative, `test_mpsattn.py` for the
softmax gate, `test_npu2.py` for the Neural Engine and the NNAPI replay. This
file exists for the ones that did **not**, which is the only reason to add a
test at all: a corrected number with nothing behind it goes stale exactly the
way the number it replaced did.

The README said the `mps` host-readback set was **54** ops. It is 85, and it
had been 87 before MPSATTN.md took two out of it. Nothing anywhere compared the
published number to the runtime list, so the drift was invisible -- the same
shape of defect as the golden `ge` floors that could not see `passed < total`
(CLAUDE.md §5.5).
"""

import os
import re

import _C


_README = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "README.md")


def _readme():
    with open(_README, encoding="utf-8") as handle:
        return handle.read()


def test_the_readme_host_readback_count_is_the_number_the_runtime_reports():
    """A published count that nothing compares against the runtime is a guess.

    Deliberately reads the number out of the prose rather than restating it
    here: a constant repeated in the test is a copy that drifts with the
    README, not a check on it.
    """
    text = _readme()
    stated = re.search(
        r"naming the op — \*\*(\d+)\*\* of them \(this cell said 54\)", text)
    assert stated, (
        "README.md no longer states the mps host-readback count in the form "
        "this test reads. If the wording moved, move this pattern with it -- "
        "do not delete the check.")
    actual = len(_C._shim_mps_host_readback_ops())
    assert int(stated.group(1)) == actual, (
        "README.md says %s host-readback ops; the runtime reports %d"
        % (stated.group(1), actual))


def test_softmax_is_not_in_the_host_readback_set():
    """The negative claim this round overturned, provoked rather than read.

    README.md said `aten._softmax.default` was refused on `mps`, so no
    transformer could forward there. It is not in the set. If it ever returns,
    that sentence becomes true again and this test is where that is noticed.
    """
    ops = _C._shim_mps_host_readback_ops()
    assert "aten._softmax.default" not in ops, sorted(ops)


def test_the_readme_no_longer_promises_the_four_collectives_refuse():
    """The sharpest defect this repository has recorded, kept from returning.

    `reduce_scatter`, `scatter`, `all_to_all` and `all_to_all_single` were
    documented as refusing by name while they silently returned each rank's own
    input (docs/distributed/COLLECT2.md). The behaviour is held by
    test_collect2.py; this holds the *sentence*, because the sentence is what
    told a reader there was nothing to check.
    """
    text = _readme()
    assert "Only `allreduce(op=SUM)` is implemented" not in text, (
        "README.md has regained the claim that only allreduce(op=SUM) works. "
        "Eleven collectives run and agree with upstream gloo at world 3 and 4.")


def test_the_roadmap_warning_survived_and_carries_a_date():
    """The table's own warning is the thing that makes it re-measurable."""
    text = _readme()
    assert "This table records what has been **measured**, not what is planned." in text
    assert "Last re-measured 2026-09-07" in text


if __name__ == "__main__":
    import sys
    mod = sys.modules[__name__]
    failed = 0
    for name in sorted(dir(mod)):
        if not name.startswith("test_"):
            continue
        try:
            getattr(mod, name)()
        except Exception as exc:
            failed = 1
            print("FAIL %s: %s" % (name, exc))
        else:
            print("ok   %s" % name)
    sys.exit(failed)
