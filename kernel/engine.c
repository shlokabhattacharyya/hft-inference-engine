// scalar forward pass for the MLP inference engine, implementing matmul,
// ReLU, and softmax without SIMD, serving as the baseline implementation.


#include <math.h>
#include <string.h>
#include "engine.h"

static void matmul(const float *W, const float *bias,
                   const float *in, float *out, int m, int n) {
    for (int i = 0; i < m; i++) {
        float acc = bias[i];
        const float *row = W + i * n;
        for (int j = 0; j < n; j++)
            acc += row[j] * in[j];
        out[i] = acc;
    }
}

static void relu(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

static void softmax(float *x, int n) {
    float max = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max) max = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++)
        x[i] *= inv_sum;
}

int infer(const float *feat, const Weights *w, float *probs) {
    ALIGN float h1[H1], h2[H2], out[NOUT];

    matmul(&w->W1[0][0], w->b1, feat, h1, H1, NIN);  relu(h1, H1);
    matmul(&w->W2[0][0], w->b2, h1, h2, H2, H1);   relu(h2, H2);
    matmul(&w->W3[0][0], w->b3, h2, out, NOUT, H2);

    memcpy(probs, out, NOUT * sizeof(float));
    softmax(probs, NOUT);

    int best = 0;
    for (int i = 1; i < NOUT; i++)
        if (probs[i] > probs[best]) best = i;
    return best - 1;  /* -1 = DOWN, 0 = FLAT, 1 = UP */
}

