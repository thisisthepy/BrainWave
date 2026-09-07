/* Layer-3 part three: can the *PyO3 extension module* be dlopen'ed on
 * Emscripten, and does its `PyInit_` run?
 *
 * 7.4 loaded the candle-only cdylib and proved the loader works. This loads the
 * one with `abi3-py313` + `extension-module`, which additionally imports 54
 * CPython symbols (45 functions from "env", 9 data symbols from "GOT.mem").
 * Those are what a real interpreter would supply. There is no CPython built for
 * this target on this machine, so this program supplies them itself, from
 * `pystubs_gen.h`, generated out of the side module's own import table.
 *
 * WHAT THIS CAN AND CANNOT DECIDE -- stated up front so the result is not
 * over-read:
 *
 *   CAN:  that Emscripten's dynamic loader resolves all 54 CPython symbols
 *         against a host module by name, both the function and the GOT.mem
 *         data halves, and instantiates our extension;
 *         that `PyInit_wasm_probe` is exported, found by `dlsym`, and *runs*;
 *         that it returns the module definition PyO3 built, with the right
 *         `m_name`, correctly relocated into the shared linear memory.
 *
 *   CANNOT: that `import torch` works. These are stubs. Nothing here refcounts,
 *         allocates a PyObject, or executes interpreter code. A real answer
 *         needs a Pyodide distribution -- see docs/platform/WASM.md 7.6.
 *
 * Build:
 *   emcc pyinit_host.c -fwasm-exceptions -sMAIN_MODULE=1 -sNODERAWFS=1 -o py.js
 */
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>

static int stub_calls = 0;
static const char *last_stub = "(none)";

static void stub_hit(const char *name) {
    stub_calls++;
    last_stub = name;
}

#include "pystubs_gen.h"

/* Same bitfield as dlopen_host.c: all five candle checks. */
#define EXPECT 31

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <side.wasm>\n", argv[0]);
        return 2;
    }
    printf("== PyInit probe ==\n");
    printf("loading: %s\n", argv[1]);

    void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        printf("dlopen  FAIL: %s\n", dlerror());
        return 1;
    }
    /* Deliberately NOT "all 54 imports resolved". A negative control (docs/platform/WASM.md
     * 7.5a) showed that deleting a CPython symbol from this host still gives a
     * successful dlopen: Emscripten substitutes a stub that aborts when called.
     * So a successful load proves the module instantiated, not that every symbol
     * was found. Resolution is proven per symbol, below, by the fact that a call
     * lands in *this file's* stub_hit rather than aborting. */
    printf("dlopen  PASS -- module instantiated (see 7.5a: this does NOT prove\n"
           "                every one of the 54 CPython imports was resolved)\n");

    int failures = 0;

    /* The candle half still has to work from inside the PyO3 build: if PyO3's
     * presence changed the layout or the gc kept something different, this is
     * where it shows. */
    int (*run)(void) = (int (*)(void))dlsym(h, "wasm_probe_run");
    if (!run) {
        printf("dlsym   FAIL wasm_probe_run: %s\n", dlerror());
        failures++;
    } else {
        int bits = run();
        int ok = (bits == EXPECT);
        failures += !ok;
        printf("candle  wasm_probe_run() = %d  expect %d  %s\n",
               bits, EXPECT, ok ? "PASS" : "FAIL");
    }

    void *(*pyinit)(void) = (void *(*)(void))dlsym(h, "PyInit_wasm_probe");
    if (!pyinit) {
        printf("dlsym   FAIL PyInit_wasm_probe: %s\n", dlerror());
        dlclose(h);
        return 1;
    }
    printf("dlsym   PASS PyInit_wasm_probe=%p\n", (void *)pyinit);

    void *def = pyinit();
    printf("PyInit  returned %p, %d stub call(s), last=%s\n",
           def, stub_calls, last_stub);
    if (!def) {
        printf("PyInit  FAIL -- returned NULL\n");
        failures++;
    }

    /* PyO3's multi-phase init returns `PyModuleDef_Init(&MODULE_DEF)`, and the
     * stub returns its argument, so `def` should be the PyModuleDef itself.
     * Its `m_name` sits after PyModuleDef_Base, whose size depends on the
     * PyObject header -- rather than hard-code a layout that would silently
     * drift, scan the first 64 bytes for a pointer to the expected name. That a
     * pointer *inside* the side module's data is dereferenceable from here at
     * all is itself the thing being tested: it means data relocation into the
     * shared linear memory worked. */
    if (def) {
        int found = -1;
        for (int off = 0; off + (int)sizeof(char *) <= 64; off += 4) {
            char *p = *(char **)((char *)def + off);
            if (p && strcmp(p, "wasm_probe") == 0) {
                found = off;
                break;
            }
        }
        if (found >= 0) {
            printf("PyInit  PASS -- PyModuleDef.m_name == \"wasm_probe\" at +%d\n", found);
        } else {
            printf("PyInit  FAIL -- no \"wasm_probe\" name pointer in the returned def\n");
            failures++;
        }
    }

    dlclose(h);
    printf("== failures = %d ==\n", failures);
    return failures ? 1 : 0;
}
