// driver for the BookClient, prints book state on each update.
#include "book_client.h"

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

    BookClient client("BTCUSDT", [](const OrderBook& book) {
        std::cout << std::fixed << std::setprecision(2)
                  << "Book: bid=" << book.bestBid()
                  << " | ask=" << book.bestAsk()
                  << " | spread=" << book.spread()
                  << " | depth=" << (book.bidDepth() + book.askDepth())
                  << "\n";
    });

    client.start();
    std::cout << "Started. Ctrl-C to stop.\n";

    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    client.stop();
    std::cout << "Stopped cleanly.\n";
    return 0;
}