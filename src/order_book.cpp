// order book implementation. 


#include "order_book.h"
#include <cmath>
#include <limits>
#include <stdexcept>

void OrderBook::applyUpdate(double price, double quantity, bool is_bid) {
    if (is_bid) {
        if (quantity == 0.0) {
            bids_.erase(price);
        } else {
            bids_[price] = quantity;
        }
    } else {
        if (quantity == 0.0) {
            asks_.erase(price);
        } else {
            asks_[price] = quantity;
        }
    }
}

void OrderBook::setLastUpdateTime(uint64_t event_time_ms) {
    last_update_ms_ = event_time_ms;
}

double OrderBook::bestBid() const {
    if (bids_.empty()) return std::numeric_limits<double>::quiet_NaN();
    return bids_.begin()->first;
}

double OrderBook::bestAsk() const {
    if (asks_.empty()) return std::numeric_limits<double>::quiet_NaN();
    return asks_.begin()->first;
}

double OrderBook::midPrice() const {
    if (bids_.empty() || asks_.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return (bestBid() + bestAsk()) / 2.0;
}

double OrderBook::spread() const {
    if (bids_.empty() || asks_.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return bestAsk() - bestBid();
}

double OrderBook::bidVolume(int n_levels) const {
    double total = 0.0;
    int count = 0;
    for (const auto& [price, qty] : bids_) {
        if (count >= n_levels) break;
        total += qty;
        ++count;
    }
    return total;
}

double OrderBook::askVolume(int n_levels) const {
    double total = 0.0;
    int count = 0;
    for (const auto& [price, qty] : asks_) {
        if (count >= n_levels) break;
        total += qty;
        ++count;
    }
    return total;
}

bool OrderBook::isReady() const {
    return !bids_.empty() && !asks_.empty();
}
