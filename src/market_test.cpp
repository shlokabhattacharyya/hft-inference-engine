// driver for MarketState, prints book + trade summary.


#include "market_state.h"

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

    MarketState market("BTCUSDT", [](const MarketState::Tick& tick) {
        std::cout << std::fixed << std::setprecision(2)
                  << "tick: t=" << tick.time_ms
                  << " mid=" << tick.book.midPrice()
                  << " spread=" << tick.book.spread()
                  << " trades_window=" << tick.recent_trades.size();

        if (!tick.recent_trades.empty()) {
            int buys = 0, sells = 0;
            double total_vol = 0;
            for (const auto& t : tick.recent_trades) {
                if (t.side > 0) buys++;
                else sells++;
                total_vol += t.quantity;
            }
            std::cout << " (" << buys << " buys, " << sells << " sells, "
                      << std::setprecision(4) << total_vol << " vol)";
        }
        std::cout << "\n";
    });

    market.start();
    std::cout << "Started. Ctrl-C to stop.\n";

    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    market.stop();
    std::cout << "Stopped cleanly.\n";
    return 0;
}
