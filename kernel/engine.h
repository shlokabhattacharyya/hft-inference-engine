// shared model definitions for the C inference engine. declares network
// dimensions, the weights layout, and scalar/SIMD inference entry points.


#ifndef ENGINE_H
#define ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>

// network dimensions (keep in sync with train_model.py)
#define NIN 10
#define H1 32
#define H2 16
#define NOUT 3

// 32-byte alignment for AVX2 loads. this lets the SIMD path use
// _mm256_load_ps (aligned) instead of _mm256_loadu_ps (unaligned)
// on weight rows whose length is a multiple of 8 floats.
#define ALIGN __attribute__((aligned(32)))

typedef struct {
    ALIGN float W1[H1][NIN];
    ALIGN float b1[H1];
    ALIGN float W2[H2][H1];
    ALIGN float b2[H2];
    ALIGN float W3[NOUT][H2];
    ALIGN float b3[NOUT];
} Weights;

// portable aligned allocation (POSIX). caller must free with weights_free().
static inline Weights *weights_alloc(void) {
    void *p = NULL;
    if (posix_memalign(&p, 32, sizeof(Weights)) != 0) return NULL;
    return (Weights *)p;
}

static inline void weights_free(Weights *w) { free(w); }

// scalar forward pass (engine.c)
int infer(const float *feat, const Weights *w, float *probs);

// AVX2 forward pass (engine_simd.c)
int infer_simd(const float *feat, const Weights *w, float *probs);

#ifdef __cplusplus
}
#endif

#endif
