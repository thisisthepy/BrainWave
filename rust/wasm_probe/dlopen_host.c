/* Layer-3 part two: does Emscripten's dlopen actually load our cdylib and run
 * code out of it?
 *
 * This is the question docs/platform/WASM.md §3a puts at the centre of the whole
 * feasibility argument. WASI has no dlopen, so on WASI `torch._C` cannot be a
 * wheel at all -- it would have to be compiled into the interpreter. Emscripten
 * is claimed to have dlopen. That claim is read from CPython/Pyodide policy
 * documents in §3, not measured. This measures it, on our actual artefact.
 *
 * Deliberately CPython-free. The side module loaded here is the *candle-only*
 * cdylib, not the PyO3 one, so that "does the dynamic loader work" is separated
 * from "can the 54 CPython symbols be resolved" (§7.5). Mixing them would give a
 * failure that could be blamed on either.
 *
 * Built for the host too, as a control: if this program cannot dlopen a plain
 * macOS .dylib then a wasm failure would say nothing about wasm.
 *
 *   host:  cc dlopen_host.c -o host_dl
 *   wasm:  emcc dlopen_host.c -sMAIN_MODULE=1 -o dl.js --preload-file <side>
 */
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

/* Both platforms agree on this value; it is compared, not trusted.
 * bit0 tensor, bit1 reduced, bit2 matmul, bit3 storage, bit4 quantised. */
#define EXPECT 31

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <path-to-shared-object>\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    printf("== dlopen probe ==\n");
    printf("loading: %s\n", path);

    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        printf("dlopen  FAIL: %s\n", dlerror());
        return 1;
    }
    printf("dlopen  PASS (handle=%p)\n", h);

    /* dlsym is the second half and it is not implied by the first: a loader can
     * instantiate a module and still have no symbol table to look names up in. */
    int (*run)(void) = (int (*)(void))dlsym(h, "wasm_probe_run");
    if (!run) {
        printf("dlsym   FAIL: %s\n", dlerror());
        dlclose(h);
        return 1;
    }
    printf("dlsym   PASS (wasm_probe_run=%p)\n", (void *)run);

    /* Calling it is the third half. Resolving a name proves relocation; only a
     * call proves the code in the side module executes against the main
     * module's malloc/free -- which on wasm is a real question, because the two
     * modules share one linear memory and one heap. */
    int bits = run();
    int ok = (bits == EXPECT);
    printf("call    wasm_probe_run() = %d  expect %d  %s\n",
           bits, EXPECT, ok ? "PASS" : "FAIL");

    dlclose(h);
    printf("== %s ==\n", ok ? "failures = 0" : "failures = 1");
    return ok ? 0 : 1;
}
