// binary I/O for model weights and feature normalization statistics.
// now both the legac CLI and the C++ runtime can load the same on-disk format.

#ifndef KERNEL_IO_H
#define KERNEL_IO_H

#ifdef __cplusplus
extern "C" {
#endif

#include "engine.h"

typedef struct { float mean[NIN]; float std[NIN]; } NormStats;

int load_weights(Weights *w, const char *path);
int load_normstats(NormStats *s, const char *path);
void normalize_features(float *f, const NormStats *s);

#ifdef __cplusplus
}
#endif

#endif

