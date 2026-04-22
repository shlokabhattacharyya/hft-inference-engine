// entry point for the C++ runtime. loads model weights and normalization stats
// via the kernel_io model, then runs inference on live market data.
#include <array>
#include <iostream>
#include <iomanip>
#include <memory>
#include <string>

#include "engine.h"
#include "kernel_io.h"

int main(int argc, char **argv) {
    const std::string weights_path   = argc > 1 ? argv[1] : "data/weights.bin";
    const std::string normstats_path = argc > 2 ? argv[2] : "data/normstats.bin";
    const bool selftest = (argc > 3 && std::string(argv[3]) == "--selftest");

    std::cout << "hft_inference_engine v0.1.0\n"
              << "  weights:   " << weights_path << "\n"
              << "  normstats: " << normstats_path << "\n";

    // RAII-managed aligned Weights
    std::unique_ptr<Weights, decltype(&weights_free)> w(
        weights_alloc(), weights_free);
    if (!w) {
        std::cerr << "weights_alloc failed\n";
        return 1;
    }

    if (!load_weights(w.get(), weights_path.c_str())) return 1;

    NormStats s{};
    if (!load_normstats(&s, normstats_path.c_str())) return 1;

    std::cout << "loaded weights + normstats OK\n";

    if (selftest) {
        // Dummy feature vector — arbitrary values. This path exists only
        // to verify the C++ → C inference bridge end-to-end. Real features
        // will come from the live order book (days 2-7).
        std::array<float, NIN> feat = {
            0.001f,    // 1-tick return
            0.003f,    // 5-tick return
            0.02f,     // spread
            0.0001f,   // normalized spread
            0.15f,     // book imbalance
            0.8f,      // signed trade flow
            1.2f,      // volume ratio
            0.005f,    // 10-tick momentum
            3.5f,      // tick intensity
            -0.001f    // VWAP deviation
        };

        normalize_features(feat.data(), &s);

        std::array<float, NOUT> probs{};
        int signal = infer_simd(feat.data(), w.get(), probs.data());

        const char *label = (signal == -1) ? "DOWN"
                          : (signal ==  0) ? "FLAT"
                                           : "UP";

        std::cout << "[selftest] signal=" << label << " probs=["
                  << std::fixed << std::setprecision(3)
                  << probs[0] << ", " << probs[1] << ", " << probs[2]
                  << "]\n";
    } else {
        std::cout << "no data source configured yet - "
                  << "run with --selftest to exercise the inference bridge\n";
    }

    return 0;
}
