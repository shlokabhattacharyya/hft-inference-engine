// shared model definitions for the C inference engine. declares network
// dimensions, the weights layout, and scalar/SIMD inference entry points.


#ifndef ENGINE_H
#define ENGINE_H

// network dimensions (keep in sync with train_model.py)
#define NIN 10
#define H1 32
#define H2 16
#define NOUT 3

typedef struct {
    float W1[H1][NIN];
    float b1[H1];
    float W2[H2][H1];
    float b2[H2];
    float W3[NOUT][H2];
    float b3[NOUT];
} Weights;

// scalar forward pass (engine.c)
int infer(const float *feat, const Weights *w, float *probs);

// AVX2 forward pass (engine_simd.c)
int infer_simd(const float *feat, const Weights *w, float *probs);

#endif