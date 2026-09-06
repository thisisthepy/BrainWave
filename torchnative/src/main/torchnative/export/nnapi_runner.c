/* nnapi_runner -- execute an upstream `_nnapi/serializer.py` blob on a real
 * NNAPI runtime.
 *
 * docs/NPU2.md is what this file exists for. `torchnative/export/nnapi.py`
 * drives upstream's serialiser and then *decodes* the blob it produced
 * (`parse_model`) -- which proves the layout and nothing about arithmetic.
 * This program is the other half: it reads the same layout and replays it into
 * `ANeuralNetworksModel`, so the operands, the immediates, the weights and the
 * opcodes all reach a driver that has to agree with them or fail.
 *
 * It is deliberately a *replayer*, not a converter. Every number it hands to
 * NNAPI comes out of the blob; there is no second lowering here that could
 * agree with the first by sharing a mistake.
 *
 * Layout (see `parse_model` in nnapi.py, and `serialize_model` in
 * `torch/backends/_nnapi/serializer.py` which writes it):
 *
 *   header    6 x int32   version, n_operands, n_values, n_operations,
 *                         n_inputs, n_outputs
 *   operands  n x (int32 op_type, int32 n_dims, float scale, int32 zero_point)
 *   values    n x (int32 operand, int32 source_type, int32 length)
 *   ops       n x (int32 opcode, int32 n_inputs, int32 n_outputs)
 *   shapes    each operand's dims, int32, already in NNAPI order (fix_shape)
 *   value data   each value's bytes, padded up to a multiple of 4
 *   op args   flat int32 array, inputs then outputs per operation
 *   model inputs / model outputs   int32 operand indices
 *
 * Weights are *not* in the blob: a NUMBERED_BUFFER value carries
 * (buf_num, offset, size) and the bytes live in `used_weights[buf_num]`,
 * already permuted to NHWC where the operand is CHANNELS_LAST. They arrive
 * here in a side file, each buffer as int32 length followed by its bytes.
 *
 * Usage:
 *   nnapi_runner model.blob weights.bin inputs.bin outputs.bin [device-name]
 *
 * With a device name it compiles with `createForDevices` for exactly that
 * driver, so the answer to "which driver ran this" is a decision rather than
 * an observation. It always prints the name it used.
 */
#include <android/NeuralNetworks.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr)                                                            \
    do {                                                                       \
        int _r = (expr);                                                       \
        if (_r != ANEURALNETWORKS_NO_ERROR) {                                  \
            fprintf(stderr, "nnapi_runner: %s failed with %d\n", #expr, _r);    \
            return 3;                                                          \
        }                                                                      \
    } while (0)

static unsigned char *slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "nnapi_runner: cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = (unsigned char *)malloc((size_t)n ? (size_t)n : 1);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "nnapi_runner: short read on %s\n", path);
        fclose(f); free(buf); return NULL;
    }
    fclose(f);
    *len = (size_t)n;
    return buf;
}

/* Bytes per element, by NNAPI operand code. The codes in the blob are
 * upstream's `NNAPI_OperandCode`, which is a transcription of the NNAPI enum,
 * so they are used directly rather than remapped. */
static size_t elem_size(int32_t code) {
    switch (code) {
        case ANEURALNETWORKS_FLOAT32:
        case ANEURALNETWORKS_INT32:
        case ANEURALNETWORKS_UINT32:
        case ANEURALNETWORKS_TENSOR_FLOAT32:
        case ANEURALNETWORKS_TENSOR_INT32:
            return 4;
        case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
        case ANEURALNETWORKS_BOOL:
        case ANEURALNETWORKS_TENSOR_BOOL8:
            return 1;
        case ANEURALNETWORKS_TENSOR_FLOAT16:
        case ANEURALNETWORKS_FLOAT16:
        case ANEURALNETWORKS_TENSOR_QUANT16_SYMM:
        case ANEURALNETWORKS_TENSOR_QUANT16_ASYMM:
            return 2;
        default:
            return 0;
    }
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
                "usage: %s model.blob weights.bin inputs.bin outputs.bin "
                "[device-name]\n", argv[0]);
        return 2;
    }
    const char *want_device = (argc > 5 && argv[5][0]) ? argv[5] : NULL;

    size_t blob_len = 0, weights_len = 0, feed_len = 0;
    unsigned char *blob = slurp(argv[1], &blob_len);
    unsigned char *weights = slurp(argv[2], &weights_len);
    unsigned char *feed = slurp(argv[3], &feed_len);
    if (!blob || !weights || !feed) return 2;

    size_t pos = 0;
#define TAKE(n)                                                                \
    ({                                                                         \
        if (pos + (size_t)(n) > blob_len) {                                    \
            fprintf(stderr, "nnapi_runner: blob truncated at %zu\n", pos);      \
            return 2;                                                          \
        }                                                                      \
        unsigned char *_p = blob + pos;                                        \
        pos += (size_t)(n);                                                    \
        _p;                                                                    \
    })

    int32_t *hdr = (int32_t *)TAKE(24);
    int32_t version = hdr[0], n_operands = hdr[1], n_values = hdr[2];
    int32_t n_operations = hdr[3], n_inputs = hdr[4], n_outputs = hdr[5];
    if (version != 1) {
        fprintf(stderr, "nnapi_runner: unknown blob version %d\n", version);
        return 2;
    }

    typedef struct { int32_t type, n_dims, zero; float scale; } Oper;
    Oper *opers = (Oper *)calloc((size_t)n_operands, sizeof(Oper));
    for (int32_t i = 0; i < n_operands; i++) {
        unsigned char *p = TAKE(16);
        memcpy(&opers[i].type, p, 4);
        memcpy(&opers[i].n_dims, p + 4, 4);
        memcpy(&opers[i].scale, p + 8, 4);
        memcpy(&opers[i].zero, p + 12, 4);
    }
    typedef struct { int32_t operand, source, length; size_t offset; } Val;
    Val *vals = (Val *)calloc((size_t)(n_values ? n_values : 1), sizeof(Val));
    for (int32_t i = 0; i < n_values; i++) {
        int32_t *p = (int32_t *)TAKE(12);
        vals[i].operand = p[0]; vals[i].source = p[1]; vals[i].length = p[2];
    }
    typedef struct { int32_t opcode, n_in, n_out; int32_t *in, *out; } Oper2;
    Oper2 *ops = (Oper2 *)calloc((size_t)(n_operations ? n_operations : 1), sizeof(Oper2));
    for (int32_t i = 0; i < n_operations; i++) {
        int32_t *p = (int32_t *)TAKE(12);
        ops[i].opcode = p[0]; ops[i].n_in = p[1]; ops[i].n_out = p[2];
    }
    uint32_t **dims = (uint32_t **)calloc((size_t)n_operands, sizeof(uint32_t *));
    for (int32_t i = 0; i < n_operands; i++)
        dims[i] = (uint32_t *)TAKE(4 * (size_t)opers[i].n_dims);
    for (int32_t i = 0; i < n_values; i++) {
        int32_t len = vals[i].length;
        int32_t padded = len ? (((len - 1) | 0x3) + 1) : 0;
        vals[i].offset = pos;
        TAKE(padded);
    }
    for (int32_t i = 0; i < n_operations; i++) {
        ops[i].in = (int32_t *)TAKE(4 * (size_t)ops[i].n_in);
        ops[i].out = (int32_t *)TAKE(4 * (size_t)ops[i].n_out);
    }
    int32_t *model_inputs = (int32_t *)TAKE(4 * (size_t)n_inputs);
    int32_t *model_outputs = (int32_t *)TAKE(4 * (size_t)n_outputs);
    if (pos != blob_len) {
        fprintf(stderr, "nnapi_runner: %zu trailing byte(s)\n", blob_len - pos);
        return 2;
    }

    printf("blob: operands=%d values=%d operations=%d inputs=%d outputs=%d\n",
           n_operands, n_values, n_operations, n_inputs, n_outputs);

    /* Pick the driver. Naming one is the point: "NNAPI ran it" and "this
     * driver ran it" are different claims (docs/NPU2.md). */
    uint32_t n_dev = 0;
    CHECK(ANeuralNetworks_getDeviceCount(&n_dev));
    ANeuralNetworksDevice *chosen = NULL;
    const char *chosen_name = NULL;
    for (uint32_t i = 0; i < n_dev; i++) {
        ANeuralNetworksDevice *d = NULL;
        const char *name = NULL;
        CHECK(ANeuralNetworks_getDevice(i, &d));
        CHECK(ANeuralNetworksDevice_getName(d, &name));
        if (want_device && strcmp(name, want_device) == 0) { chosen = d; chosen_name = name; }
    }
    if (want_device && !chosen) {
        fprintf(stderr, "nnapi_runner: no device named %s\n", want_device);
        return 2;
    }

    ANeuralNetworksModel *model = NULL;
    CHECK(ANeuralNetworksModel_create(&model));
    for (int32_t i = 0; i < n_operands; i++) {
        ANeuralNetworksOperandType t;
        t.type = opers[i].type;
        t.dimensionCount = (uint32_t)opers[i].n_dims;
        t.dimensions = opers[i].n_dims ? dims[i] : NULL;
        t.scale = opers[i].scale;
        t.zeroPoint = opers[i].zero;
        CHECK(ANeuralNetworksModel_addOperand(model, &t));
    }
    /* Values. IMMEDIATE carries its bytes inline; NUMBERED_BUFFER carries
     * (buf_num, offset, size) and the bytes are in the weights file. */
    size_t wpos = 0;
    size_t *wbuf_off = NULL; int32_t *wbuf_len = NULL; int n_wbuf = 0, cap_wbuf = 0;
    while (wpos + 4 <= weights_len) {
        int32_t len; memcpy(&len, weights + wpos, 4); wpos += 4;
        if (n_wbuf == cap_wbuf) {
            cap_wbuf = cap_wbuf ? cap_wbuf * 2 : 8;
            wbuf_off = (size_t *)realloc(wbuf_off, (size_t)cap_wbuf * sizeof(size_t));
            wbuf_len = (int32_t *)realloc(wbuf_len, (size_t)cap_wbuf * sizeof(int32_t));
        }
        wbuf_off[n_wbuf] = wpos; wbuf_len[n_wbuf] = len; n_wbuf++;
        wpos += (size_t)len;
    }
    for (int32_t i = 0; i < n_values; i++) {
        if (vals[i].source == 0) {  /* IMMEDIATE */
            CHECK(ANeuralNetworksModel_setOperandValue(
                model, vals[i].operand, blob + vals[i].offset,
                (size_t)vals[i].length));
        } else if (vals[i].source == 2) {  /* NUMBERED_BUFFER */
            int32_t *triple = (int32_t *)(blob + vals[i].offset);
            int32_t buf = triple[0], off = triple[1], size = triple[2];
            if (buf < 0 || buf >= n_wbuf || off + size > wbuf_len[buf]) {
                fprintf(stderr,
                        "nnapi_runner: value %d names buffer %d[%d..%d] and the "
                        "weights file has %d buffer(s)\n", i, buf, off, off + size,
                        n_wbuf);
                return 2;
            }
            CHECK(ANeuralNetworksModel_setOperandValue(
                model, vals[i].operand, weights + wbuf_off[buf] + off, (size_t)size));
        } else {
            fprintf(stderr, "nnapi_runner: value source %d not supported\n",
                    vals[i].source);
            return 2;
        }
    }
    for (int32_t i = 0; i < n_operations; i++) {
        CHECK(ANeuralNetworksModel_addOperation(
            model, (ANeuralNetworksOperationType)ops[i].opcode,
            (uint32_t)ops[i].n_in, (const uint32_t *)ops[i].in,
            (uint32_t)ops[i].n_out, (const uint32_t *)ops[i].out));
    }
    CHECK(ANeuralNetworksModel_identifyInputsAndOutputs(
        model, (uint32_t)n_inputs, (const uint32_t *)model_inputs,
        (uint32_t)n_outputs, (const uint32_t *)model_outputs));
    CHECK(ANeuralNetworksModel_finish(model));

    ANeuralNetworksCompilation *comp = NULL;
    if (chosen) {
        /* Report whether the named driver claims every operation, before
         * compiling for it. A driver that partially supports the model would
         * otherwise fall back invisibly. */
        bool *supported = (bool *)calloc((size_t)(n_operations ? n_operations : 1), 1);
        CHECK(ANeuralNetworksModel_getSupportedOperationsForDevices(
            model, (const ANeuralNetworksDevice *const *)&chosen, 1, supported));
        int n_sup = 0;
        for (int32_t i = 0; i < n_operations; i++) n_sup += supported[i] ? 1 : 0;
        printf("device: %s supports %d/%d operation(s)\n", chosen_name, n_sup,
               n_operations);
        CHECK(ANeuralNetworksCompilation_createForDevices(
            model, (const ANeuralNetworksDevice *const *)&chosen, 1, &comp));
    } else {
        printf("device: (runtime's choice)\n");
        CHECK(ANeuralNetworksCompilation_create(model, &comp));
    }
    CHECK(ANeuralNetworksCompilation_finish(comp));

    ANeuralNetworksExecution *exec = NULL;
    CHECK(ANeuralNetworksExecution_create(comp, &exec));
    size_t fpos = 0;
    for (int32_t i = 0; i < n_inputs; i++) {
        int32_t oid = model_inputs[i];
        size_t n = elem_size(opers[oid].type);
        for (int32_t d = 0; d < opers[oid].n_dims; d++) n *= dims[oid][d];
        if (fpos + n > feed_len) {
            fprintf(stderr, "nnapi_runner: input file is short by %zu byte(s)\n",
                    fpos + n - feed_len);
            return 2;
        }
        CHECK(ANeuralNetworksExecution_setInput(exec, i, NULL, feed + fpos, n));
        fpos += n;
    }
    size_t out_total = 0;
    size_t *out_size = (size_t *)calloc((size_t)(n_outputs ? n_outputs : 1), sizeof(size_t));
    for (int32_t i = 0; i < n_outputs; i++) {
        int32_t oid = model_outputs[i];
        size_t n = elem_size(opers[oid].type);
        for (int32_t d = 0; d < opers[oid].n_dims; d++) n *= dims[oid][d];
        out_size[i] = n;
        out_total += n;
    }
    unsigned char *out = (unsigned char *)calloc(out_total ? out_total : 1, 1);
    size_t opos = 0;
    for (int32_t i = 0; i < n_outputs; i++) {
        CHECK(ANeuralNetworksExecution_setOutput(exec, i, NULL, out + opos, out_size[i]));
        opos += out_size[i];
    }
    CHECK(ANeuralNetworksExecution_compute(exec));

    FILE *f = fopen(argv[4], "wb");
    if (!f) { fprintf(stderr, "nnapi_runner: cannot write %s\n", argv[4]); return 2; }
    fwrite(out, 1, out_total, f);
    fclose(f);
    printf("executed: wrote %zu byte(s) across %d output(s)\n", out_total, n_outputs);
    return 0;
}
