// multi-stream market data aggregator. owns BookClient and TradeClient, 
// presents a unified Tick callback to user code. the Tick fires on each book 
// update (current model). future: switch to firing on trades instead


#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "book_client.h"
#include "trade_client.h"
#include "order_book.h"

class MarketState {
public:
    struct Tick {
        const OrderBook& book;
        const std::vector<Trade>& recent_trades;
        uint64_t time_ms;
    };

    using TickCallback = std::function<void(const Tick&)>;

    MarketState(const std::string& symbol, TickCallback on_tick);
    ~MarketState() = default;

    MarketState(const MarketState&) = delete;
    MarketState& operator=(const MarketState&) = delete;

    void start();
    void stop();

private:
    void onBookUpdate(const OrderBook& book);
    void fireTick(const OrderBook& book);

    BookClient book_client_;
    TradeClient trade_client_;
    TickCallback on_tick_;

    static constexpr uint64_t TRADE_WINDOW_MS = 60'000;
};
