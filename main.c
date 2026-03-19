// command-line driver for the inference engine. loads model weights + normalization
// stats, streams tick data, runs inference, and reports latency percentiles.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "engine.h"
#include "latency.h"

// feature normalization
typedef struct { float mean[NIN]; float std[NIN]; } NormStats;

static void normalize(float *f, const NormStats *s) {
    for (int i = 0; i < NIN; i++)
        f[i] = (f[i] - s->mean[i]) / (s->std[i] < 1e-8f ? 1e-8f : s->std[i]);
}

// keep rolling window of ticks to compute features
typedef struct {
    double ts, bid, ask, bid_vol, ask_vol;
    double trade_px, trade_vol;
    int side;       // 1 = buy, -1 = sell
} Tick;

#define WIN 64
typedef struct { Tick t[WIN]; int head, count; } TickBuf;

static void buf_push(TickBuf *b, const Tick *tick) {
    b->t[b->head] = *tick;
    b->head = (b->head + 1) % WIN;
    if (b->count < WIN) b->count++;
}

static const Tick *buf_get(const TickBuf *b, int i) {
    return &b->t[((b->head - 1 - i) % WIN + WIN) % WIN];
}

static int compute_features(const TickBuf *b, float *f) {
    if (b->count < 10) return 0;

    const Tick *t0 = buf_get(b, 0);
    const Tick *t1 = buf_get(b, 1);
    const Tick *t5 = buf_get(b, 5);
    const Tick *t9 = buf_get(b, 9);

    double mid0 = (t0->bid + t0->ask) * 0.5;
    double mid1 = (t1->bid + t1->ask) * 0.5;
    double mid5 = (t5->bid + t5->ask) * 0.5;
    double mid9 = (t9->bid + t9->ask) * 0.5;

    f[0] = (float)((mid0 - mid1) / (mid1 + 1e-10));           // 1-tick return
    f[1] = (float)((mid0 - mid5) / (mid5 + 1e-10));           // 5-tick return
    f[2] = (float)(t0->ask - t0->bid);                        // spread
    f[3] = (float)(f[2] / (mid0 + 1e-10));                    // normalized spread

    double tv = t0->bid_vol + t0->ask_vol;
    f[4] = tv > 0 ? (float)((t0->bid_vol - t0->ask_vol) / tv) : 0.0f; // book imbalance

    double imb = 0;
    for (int i = 0; i < 10; i++) { const Tick *t = buf_get(b, i); imb += t->side * t->trade_vol; }
    f[5] = (float)imb;                                        // signed trade flow

    f[6] = t0->ask_vol > 0 ? (float)(t0->bid_vol / t0->ask_vol) : 1.0f; // vol ratio
    f[7] = (float)((mid0 - mid9) / (mid9 + 1e-10));                     // 10-tick momentum
    f[8] = (float)(b->count < 10 ? b->count : 10);                      // tick intensity

    double vwap_n = 0, vwap_d = 0;
    for (int i = 0; i < 10; i++) {
        const Tick *t = buf_get(b, i);
        double m = (t->bid + t->ask) * 0.5;
        vwap_n += m * t->trade_vol; vwap_d += t->trade_vol;
    }
    double vwap = vwap_d > 0 ? vwap_n / vwap_d : mid0;
    f[9] = (float)((mid0 - vwap) / (vwap + 1e-10));          // VWAP deviation

    return 1;
}

static int load_weights(Weights *w, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 0; }
    int ok = fread(w->W1, 4, H1*NIN, f)  == (size_t)(H1*NIN)  &&
             fread(w->b1, 4, H1,     f)  == (size_t)H1         &&
             fread(w->W2, 4, H2*H1,  f)  == (size_t)(H2*H1)   &&
             fread(w->b2, 4, H2,     f)  == (size_t)H2         &&
             fread(w->W3, 4, NOUT*H2,f)  == (size_t)(NOUT*H2) &&
             fread(w->b3, 4, NOUT,   f)  == (size_t)NOUT;
    fclose(f);
    return ok;
}

static int load_normstats(NormStats *s, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 0; }
    int ok = fread(s->mean, 4, NIN, f) == (size_t)NIN &&
             fread(s->std,  4, NIN, f) == (size_t)NIN;
    fclose(f);
    return ok;
}

// main: load model, stream ticks, run inference, measure latency
int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <ticks.csv> <weights.bin> <normstats.bin> [--simd]\n", argv[0]);
        return 1;
    }

    int use_simd = (argc >= 5 && strcmp(argv[4], "--simd") == 0);

    Weights   *w = malloc(sizeof(Weights));
    NormStats  s;
    if (!load_weights(w, argv[2]) || !load_normstats(&s, argv[3])) return 1;
    fprintf(stderr, "loaded weights (%s)\n", use_simd ? "SIMD" : "scalar");

    FILE *csv = fopen(argv[1], "r");
    if (!csv) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }

    TickBuf buf = (TickBuf){0};
    Latency lat; lat_init(&lat);
    char line[512];
    long processed = 0;

    if (!fgets(line, sizeof(line), csv)) return 1;  // skip header

    while (fgets(line, sizeof(line), csv)) {
        Tick tick;
        if (sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%d",
                   &tick.ts, &tick.bid, &tick.ask,
                   &tick.bid_vol, &tick.ask_vol,
                   &tick.trade_px, &tick.trade_vol, &tick.side) != 8) continue;

        buf_push(&buf, &tick);

        float feat[NIN];
        if (!compute_features(&buf, feat)) continue;
        normalize(feat, &s);

        // measure only the forward pass
        float probs[NOUT];
        u64 t0 = ticks_now();
        int signal = use_simd ? infer_simd(feat, w, probs)
                               : infer(feat, w, probs);
        u64 t1 = ticks_now();

        lat_record(&lat, t1 - t0);
        processed++;

        // emit: timestamp, signal, probabilities
        printf("%.3f %+d  [%.3f %.3f %.3f]\n",
               tick.ts, signal, probs[0], probs[1], probs[2]);
    }

    fclose(csv);
    fprintf(stderr, "processed %ld ticks\n", processed);
    lat_report(&lat);
    free(w);
    return 0;
}