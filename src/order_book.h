// order book reconstruction from Binance depth updates (maintains sorted bid/
// ask price levels, provides top-of-book and top-N volume queries for feature
// extraction).


#pragma once
#include <map>
#include <functional>
#include <cstdint>

class OrderBook {
public:
    // apply a single price-level update. quantity == 0 removes the level.
    void applyUpdate(double price, double quantity, bool is_bid);

    // called once per message to mark the book as having a new state
    void setLastUpdateTime(uint64_t event_time_ms);

    // queries (return NaN if book is empty on that side)
    double bestBid() const;
    double bestAsk() const;
    double midPrice() const;
    double spread() const;

    // sum of quantities across top N price levels
    double bidVolume(int n_levels) const;
    double askVolume(int n_levels) const;

    // state inspection
    bool isReady() const;   // true once both sides have at least 1 level
    uint64_t lastUpdateTimeMs() const { return last_update_ms_; }
    size_t bidDepth() const { return bids_.size(); }
    size_t askDepth() const { return asks_.size(); }

private:
    // bids sorted high-to-low (std::greater), asks low-to-high (default).
    // bids_.begin() is best bid; asks_.begin() is best ask.
    std::map<double, double, std::greater<double>> bids_;
    std::map<double, double> asks_;
    uint64_t last_update_ms_ = 0;
};
