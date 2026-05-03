// driver for TradeClient, prints trades as they arrive.


#include "trade_client.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <thread>

static std::atomic<bool> g_running{true};

static void signalHandler(int) {
    g_running = false;
}

int main() {
    std::signal(SIGINT, signalHandler);

    TradeClient client("BTCUSDT");
    client.start();
    std::cout << "Started. Ctrl-C to stop.\n";

    while (g_running) {
        std::this_thread::sleep_for(std::chrono::seconds(2));

        auto trades = client.snapshot();
        if (!trades.empty()) {
            const auto& latest = trades.back();
            std::cout << std::fixed << std::setprecision(2)
                      << "trades buffered=" << trades.size()
                      << " latest: t=" << latest.time_ms
                      << " px=" << latest.price
                      << " qty=" << std::setprecision(6) << latest.quantity
                      << " side=" << static_cast<int>(latest.side)
                      << "\n";
        }
    }

    client.stop();
    std::cout << "Stopped cleanly.\n";
    return 0;
}
