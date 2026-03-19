# build rules for the scalar and AVX2 SIMD inference engine binaries.
# produces build/hft_engine (baseline) and build/hft_engine_simd (optimized) from the same sources.

CC = gcc
CFLAGS = -O2 -Wall -Wextra -D_GNU_SOURCE
SIMD = -O3 -march=native -mavx2 -mfma -DUSE_SIMD
LDLIBS = -lm

SRC_COMMON = main.c engine.c engine_simd.c
BUILD = build

all: scalar simd

scalar: $(SRC_COMMON)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -o $(BUILD)/hft_engine $^ $(LDLIBS)
	@echo "built: $(BUILD)/hft_engine"

simd: $(SRC_COMMON)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(SIMD) -o $(BUILD)/hft_engine_simd $^ $(LDLIBS)
	@echo "built: $(BUILD)/hft_engine_simd"

clean:
	rm -rf $(BUILD)

.PHONY: all scalar simd clean