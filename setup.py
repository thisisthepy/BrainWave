"""The two things about this distribution that `pyproject.toml` cannot say.

Everything declarative lives in `pyproject.toml`. What is left here is the pair
of facts that decide the *wheel tag*, and neither has a `[tool.setuptools]`
spelling:

1.  This is not a pure-Python distribution. It carries `torch/_C.abi3.so`, a
    compiled extension -- but setuptools cannot see that, because the extension
    is *pre-built* by `vendor/install_shim.sh` and arrives as package data
    rather than as an `Extension()` setuptools compiled itself. Left alone,
    `Distribution.is_pure()` answers True and the wheel goes out tagged
    `py3-none-any`: installable on Android, on iOS, on any machine at all, and
    functional on none of them, because the `.so` inside is Mach-O arm64. That
    is exactly the distribution now on PyPI. Overriding `has_ext_modules()` is
    the documented way to correct it.

2.  The extension is a Limited-API build (`abi3-py313`; docs/design/ABI3.md). Saying so
    turns the tag from `cp313-cp313-<plat>` into `cp313-abi3-<plat>`, which is
    the difference between a wheel that serves 3.13 and one that serves 3.13 and
    every CPython after it. `py_limited_api` is a `bdist_wheel` *command
    option*, so it has to be passed through `options=`.

Both are load-bearing for the tag and nothing else; if this file were deleted
the wheel would still build, and would still be wrong in both directions.

Build with `python tools/wheel/build.py`, which checks that the vendored tree
and the shim are actually in place first. `pip wheel .` works too and skips
those checks.
"""

from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """A distribution that is platform-specific because of files it *ships*.

    `has_ext_modules()` normally answers "did setuptools compile anything for
    me". The honest answer for this project is "no, `cargo` did" -- and the
    question the wheel machinery is really asking is "is this archive tied to
    one platform", to which the answer is yes.
    """

    def has_ext_modules(self) -> bool:
        return True

    # `is_pure()` is derived from `has_ext_modules()` in setuptools, so this is
    # not overridden separately -- keeping one source of truth means the two
    # cannot drift apart into a wheel that is platform-tagged but marked
    # `Root-Is-Purelib: true`, which unpacks into the wrong directory.


setup(
    distclass=BinaryDistribution,
    options={"bdist_wheel": {"py_limited_api": "cp313"}},
)
