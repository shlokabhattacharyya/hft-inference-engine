// AVX2 (advanced vector extensions 2)-accelerated forward pass for the MLP 
// inference  engine, containing SIMD implementations of matmul and ReLU, 
// with a scalar fallback when AVX2 isn't enabled.

/*
AVX2 dot product: processes 8 floats per iteration using 256-bit FMA 
(fused multiply-add)

the inner loop of matmul is a dot product. with AXV2, we load 8 floats at
once into a 256-bit register and used FMA to accumulate. this reduces the
iteration count by 8x and eliminates a separate multiply instruction.

note: memory layout matters! row-major weights mean each row is contiguous,
so loadu_ps accesses sequential addresses and no cache misses mid row.
*/


#include <math.h>
#include <string.h>
#include "engine.h"

#ifdef __AVX2__
#include <immintrin.h>


static inline float dot8(const float *a, const float *b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);

    // horizontal sum of 8-wide accumulator
    __m128 lo  = _mm256_castps256_ps128(acc);
    __m128 hi  = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float result = _mm_cvtss_f32(sum);

    for (; i < n; i++) result += a[i] * b[i];  // scalar remainder
    return result;
}

static void matmul_avx2(const float *W, const float *bias,
                         const float *in, float *out, int m, int n) {
    for (int i = 0; i < m; i++)
        out[i] = bias[i] + dot8(W + i * n, in, n);
}

// vectorized ReLU: _mm256_max_ps with zero replaces a branch per element
static void relu_avx2(float *x, int n) {
    __m256 zero = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(x + i, _mm256_max_ps(_mm256_loadu_ps(x + i), zero));
    for (; i < n; i++) x[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

#else
// fallback in plain C if compiler does not have AXV2 instructions enabled
static void matmul_avx2(const float *W, const float *bias,
                         const float *in, float *out, int m, int n) {
    for (int i = 0; i < m; i++) {
        float acc = bias[i];
        for (int j = 0; j < n; j++) acc += W[i*n+j] * in[j];
        out[i] = acc;
    }
}
static void relu_avx2(float *x, int n) {
    for (int i = 0; i < n; i++) x[i] = x[i] > 0.0f ? x[i] : 0.0f;
}
#endif

static void softmax(float *x, int n) {
    float max = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

int infer_simd(const float *feat, const Weights *w, float *probs) {
    float h1[H1], h2[H2], out[NOUT];

    matmul_avx2(&w->W1[0][0], w->b1, feat, h1, H1, NIN);  relu_avx2(h1, H1);
    matmul_avx2(&w->W2[0][0], w->b2, h1,   h2, H2, H1);   relu_avx2(h2, H2);
    matmul_avx2(&w->W3[0][0], w->b3, h2,   out, NOUT, H2);

    memcpy(probs, out, NOUT * sizeof(float));
    softmax(probs, NOUT);

    int best = 0;
    for (int i = 1; i < NOUT; i++) if (probs[i] > probs[best]) best = i;
    return best - 1;
}