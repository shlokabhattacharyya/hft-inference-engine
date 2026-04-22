// binary I/O for model weights and feature normalization statistics.

#include "kernel_io.h"
#include <stdio.h>

int load_weights(Weights *w, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 0; }
    int ok = fread(w->W1, 4, H1*NIN,  f) == (size_t)(H1*NIN) &&
             fread(w->b1, 4, H1,      f) == (size_t)H1 &&
             fread(w->W2, 4, H2*H1,   f) == (size_t)(H2*H1) &&
             fread(w->b2, 4, H2,      f) == (size_t)H2 &&
             fread(w->W3, 4, NOUT*H2, f) == (size_t)(NOUT*H2) &&
             fread(w->b3, 4, NOUT,    f) == (size_t)NOUT;
    fclose(f);
    return ok;
}

int load_normstats(NormStats *s, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 0; }
    int ok = fread(s->mean, 4, NIN, f) == (size_t)NIN &&
             fread(s->std,  4, NIN, f) == (size_t)NIN;
    fclose(f);
    return ok;
}

void normalize_features(float *f, const NormStats *s) {
    for (int i = 0; i < NIN; i++)
        f[i] = (f[i] - s->mean[i]) / (s->std[i] < 1e-8f ? 1e-8f : s->std[i]);
}
