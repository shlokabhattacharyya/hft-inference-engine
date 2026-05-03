// multi-stream market data aggregator. owns BookClient and TradeClient,
// presents a unified Tick callback to user code. the Tick fires on each book 
// update (current model). future: switch to firing on trades instead


#include "market_state.h"

#include <chrono>

MarketState::MarketState(const std::string& symbol, TickCallback on_tick)
    : book_client_(symbol, [this](const OrderBook& book) {
          onBookUpdate(book);
      }),
      trade_client_(symbol),
      on_tick_(std::move(on_tick)) {
}

void MarketState::start() {
    trade_client_.start();
    book_client_.start();
}

void MarketState::stop() {
    book_client_.stop();
    trade_client_.stop();
}

void MarketState::onBookUpdate(const OrderBook& book) {
    fireTick(book);
}

void MarketState::fireTick(const OrderBook& book) {
    // trim the trade buffer to the time window we care about
    const uint64_t now_ms = book.lastUpdateTimeMs();
    if (now_ms > TRADE_WINDOW_MS) {
        trade_client_.pruneOlderThan(now_ms - TRADE_WINDOW_MS);
    }

    // snapshot the trade buffer for the callback
    std::vector<Trade> trades = trade_client_.snapshot();

    Tick tick{book, trades, now_ms};
    on_tick_(tick);
}
