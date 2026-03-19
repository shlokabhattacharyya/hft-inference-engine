// lightweight latency measurement utilities using a fast monotonic counter and percentile reporting.
// uses RDTSCP on x86, mach_absolute_time on macOS, and clock_gettime elsewhere.


#ifndef LATENCY_H
#define LATENCY_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned long long u64;

#if defined(__x86_64__) || defined(_M_X64)
static inline u64 ticks_now(void) {
    unsigned int lo, hi, aux;
    __asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux));
    return ((u64)hi << 32) | lo;
}

// detect CPU frequency from /proc/cpuinfo (Linux). fallback 3.0 GHz.
static inline double ticks_to_ns_scale(void) {
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return 1e9 / (3.0e9);
    char line[256];
    double mhz = 0.0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "cpu MHz", 7) == 0) {
            char *p = strchr(line, ':');
            if (p) { mhz = atof(p + 1); break; }
        }
    }
    fclose(f);
    if (mhz <= 0.0) mhz = 3000.0;
    return 1e9 / (mhz * 1e6); // ns per tick
}

#elif defined(__APPLE__)
#include <mach/mach_time.h>

static inline u64 ticks_now(void) {
    return (u64)mach_absolute_time();
}

// mach_absolute_time ticks * numer/denom = ns
static inline double ticks_to_ns_scale(void) {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    return (double)info.numer / (double)info.denom;
}

#else
#include <time.h>

static inline u64 ticks_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (u64)ts.tv_sec * 1000000000ULL + (u64)ts.tv_nsec;
}

static inline double ticks_to_ns_scale(void) {
    return 1.0; // already ns
}
#endif

#define MAX_SAMPLES 200000

typedef struct {
    u64 samples[MAX_SAMPLES];
    size_t count;
    double ns_per_tick;
} Latency;

static inline void lat_init(Latency *l) {
    l->count = 0;
    l->ns_per_tick = ticks_to_ns_scale();
}

static inline void lat_record(Latency *l, u64 ticks) {
    if (l->count < MAX_SAMPLES) {
        l->samples[l->count++] = (u64)((double)ticks * l->ns_per_tick);
    }
}

static int _cmp_u64(const void *a, const void *b) {
    u64 x = *(const u64 *)a, y = *(const u64 *)b;
    return (x > y) - (x < y);
}

static inline void lat_report(Latency *l) {
    if (!l->count) return;
    qsort(l->samples, l->count, sizeof(u64), _cmp_u64);
    size_t n = l->count;
#define PCT(p) l->samples[(size_t)((p) / 100.0 * (n - 1))]
    printf("\n=== latency (%zu samples) ===\n", n);
    printf("  p50:   %4llu ns\n", PCT(50.0));
    printf("  p95:   %4llu ns\n", PCT(95.0));
    printf("  p99:   %4llu ns\n", PCT(99.0));
    printf("  p99.9: %4llu ns\n", PCT(99.9));
    printf("  min:   %4llu ns\n", l->samples[0]);
    printf("  max:   %4llu ns\n", l->samples[n - 1]);
#undef PCT
}

#endif