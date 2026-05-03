// Binance aggregate trade stream client. maintains a time-windowed buffer 
// of recent trades for feature extraction.


#pragma once

#include <ixwebsocket/IXWebSocket.h>
#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct Trade {
    uint64_t time_ms;
    double price;
    double quantity;
    int8_t side;  // +1 = buy-initiated, -1 = sell-initiated
};

class TradeClient {
public:
    explicit TradeClient(const std::string& symbol);
    ~TradeClient();

    TradeClient(const TradeClient&) = delete;
    TradeClient& operator=(const TradeClient&) = delete;

    void start();
    void stop();

    // returns a copy of all trades currently in the buffer (caller can 
    // filter by time as needed)
    std::vector<Trade> snapshot() const;

    // drops trades older than (now_ms - max_age_ms) - called periodically
    // from MarketState to prevent unbounded growth
    void pruneOlderThan(uint64_t cutoff_time_ms);

    // diagnostics
    size_t bufferSize() const;

private:
    void onMessage(const std::string& payload);
    void onOpen();
    void onClose();
    Trade parseTrade(const std::string& payload);

    const std::string symbol_;
    ix::WebSocket ws_;

    mutable std::mutex buffer_mutex_;
    std::deque<Trade> buffer_;

    std::atomic<bool> shutting_down_{false};
};
