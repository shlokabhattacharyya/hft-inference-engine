// Binance order book client with snapshot reconciliation. manages the 
// WebSocket connection, buffers diffs, fetches REST snapshots, and reconciles
// them via update-ID straddle check. detects sequence gaps in the live diff
// stream and triggers automatic resync.


#pragma once

#include <ixwebsocket/IXWebSocket.h>
#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "order_book.h"

class BookClient {
public:
    enum class State { Disconnected, Buffering, Synced, Desynced };

    // called from the IXWebSocket thread after the book has been
    // updated by an applied diff (only while in Synced state)
    using UpdateCallback = std::function<void(const OrderBook&)>;

    BookClient(const std::string& symbol, UpdateCallback on_update);
    ~BookClient();

    // non-copyable, non-movable: holds threads and a socket
    BookClient(const BookClient&) = delete;
    BookClient& operator=(const BookClient&) = delete;

    void start();
    void stop();

    State state() const { return state_.load(); }

private:
    // internal types — same shape as the free functions in ws_test.cpp,
    // but scoped to the class so we don't contaminate the global namespace
    struct PriceLevel { double price; double quantity; };
    struct DepthUpdate {
        uint64_t event_time_ms;
        uint64_t first_update_id;
        uint64_t last_update_id;
        std::vector<PriceLevel> bids;
        std::vector<PriceLevel> asks;
    };
    struct Snapshot {
        uint64_t last_update_id;
        std::vector<PriceLevel> bids;
        std::vector<PriceLevel> asks;
    };

    // WebSocket event handlers
    void onMessage(const std::string& payload);
    void onOpen();
    void onClose();

    // resync flow
    void resyncOnThread();
    Snapshot fetchSnapshot();
    DepthUpdate parseDepthUpdate(const std::string& payload);
    bool reconcileAndApply(const Snapshot& snap);
    void triggerResync();

    // members
    const std::string symbol_;
    UpdateCallback on_update_;

    ix::WebSocket ws_;
    OrderBook book_;

    std::atomic<State> state_{State::Disconnected};

    std::mutex buffer_mutex_;
    std::vector<DepthUpdate> buffer_;       // protected by buffer_mutex_

    uint64_t last_applied_update_id_ = 0;   // only touched from WS thread (Synced)
    std::atomic<bool> resync_in_progress_{false};
};
