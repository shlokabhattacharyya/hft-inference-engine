// Binance order book client with snapshot reconciliation. manages the
// WebSocket connection, buffers diffs, fetches REST snapshots, and reconciles
// them via update-ID straddle check. Detects sequence gaps in the live diff
// stream and triggers automatic resync.


#include "book_client.h"

#include <ixwebsocket/IXHttpClient.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

BookClient::BookClient(const std::string& symbol, UpdateCallback on_update)
    : symbol_(symbol), on_update_(std::move(on_update)) {

    std::string lower;
    lower.reserve(symbol.size());
    for (char c : symbol) lower.push_back(std::tolower(static_cast<unsigned char>(c)));

    ws_.setUrl("wss://stream.binance.us:9443/ws/" + lower + "@depth");
    ws_.setPingInterval(30);

    ws_.setOnMessageCallback([this](const ix::WebSocketMessagePtr& msg) {
        switch (msg->type) {
            case ix::WebSocketMessageType::Message:
                onMessage(msg->str);
                break;
            case ix::WebSocketMessageType::Open:
                onOpen();
                break;
            case ix::WebSocketMessageType::Close:
                onClose();
                break;
            case ix::WebSocketMessageType::Error:
                std::cerr << "[WS ERROR] " << msg->errorInfo.reason << "\n";
                break;
            default:
                break;
        }
    });
}

BookClient::~BookClient() {
    stop();
}

void BookClient::start() { ws_.start(); }

void BookClient::stop()  { ws_.stop(); }

void BookClient::onMessage(const std::string& payload) {
    DepthUpdate update;
    try {
        update = parseDepthUpdate(payload);
    } catch (const std::exception& e) {
        std::cerr << "[parse error] " << e.what() << "\n";
        return;
    }

    State s = state_.load();

    if (s == State::Buffering) {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        buffer_.push_back(std::move(update));
        return;
    }

    if (s == State::Synced) {
        // gap detection: every diff must continue from the last one
        if (update.first_update_id != last_applied_update_id_ + 1) {
            std::cerr << "[GAP] expected U=" << (last_applied_update_id_ + 1)
                      << " got U=" << update.first_update_id
                      << "; resyncing\n";
            triggerResync();
            return;
        }

        // apply the update
        for (const auto& lv : update.bids)
            book_.applyUpdate(lv.price, lv.quantity, /*is_bid=*/true);
        for (const auto& lv : update.asks)
            book_.applyUpdate(lv.price, lv.quantity, /*is_bid=*/false);
        book_.setLastUpdateTime(update.event_time_ms);
        last_applied_update_id_ = update.last_update_id;

        if (book_.isReady()) {
            on_update_(book_);
        }
        return;
    }

    // Disconnected or Desynced: drop the message (Resync will rebuild
    // the book from a fresh snapshot)
}

void BookClient::onOpen() {
    std::cout << "[WS] Connected, buffering...\n";
    state_ = State::Buffering;

    // kick off the snapshot fetch on a background thread so we don't
    // block the WebSocket thread
    std::thread([this]() { resyncOnThread(); }).detach();
}

void BookClient::onClose() {
    std::cout << "[WS] Disconnected\n";
    state_ = State::Disconnected;

    // reset state so the next reconnect starts cleanly
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        buffer_.clear();
    }
    book_.clear();
    last_applied_update_id_ = 0;

    // IXWebSocket auto-reconnects by default - when it does, onOpen will
    // fire again and trigger a fresh resync
}

void BookClient::resyncOnThread() {
    if (resync_in_progress_.exchange(true)) {
        return;
    }

    int attempt = 0;
    while (state_.load() != State::Disconnected) {
        attempt++;
        std::cout << "[RESYNC] attempt " << attempt << "\n";

        try {
            // wait until we've buffered at least a few diffs before fetching (on 
            // thin markets, snapshots can arrive faster than diffs)
            for (int i = 0; i < 50; i++) {
                {
                    std::lock_guard<std::mutex> lock(buffer_mutex_);
                    if (buffer_.size() >= 3) break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            Snapshot snap = fetchSnapshot();
            std::cout << "[RESYNC] snapshot lastUpdateId=" << snap.last_update_id
                      << " bids=" << snap.bids.size()
                      << " asks=" << snap.asks.size() << "\n";

            // wait briefly for a diff covering L+1 to arrive, in case the
            // snapshot is at or beyond our latest buffered diff
            const uint64_t needed = snap.last_update_id + 1;
            for (int i = 0; i < 50; i++) {
                {
                    std::lock_guard<std::mutex> lock(buffer_mutex_);
                    if (!buffer_.empty() && buffer_.back().last_update_id >= needed) {
                        break;
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            if (reconcileAndApply(snap)) {
                std::cout << "[RESYNC] success, now synced\n";
                state_ = State::Synced;
                break;
            }

            std::cout << "[RESYNC] reconciliation failed; retrying\n";
        } catch (const std::exception& e) {
            std::cerr << "[RESYNC ERROR] " << e.what() << "\n";
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    resync_in_progress_ = false;
}

BookClient::Snapshot BookClient::fetchSnapshot() {
    ix::HttpClient client;

    auto args = client.createRequest();
    args->connectTimeout = 10;
    args->transferTimeout = 10;

    std::string url = "https://api.binance.us/api/v3/depth?symbol="
                      + symbol_ + "&limit=1000";
    auto response = client.get(url, args);

    if (response->statusCode != 200) {
        throw std::runtime_error("Snapshot HTTP " +
            std::to_string(response->statusCode) +
            " errorMsg='" + response->errorMsg + "'");
    }

    nlohmann::json j = nlohmann::json::parse(response->body);

    Snapshot snap;
    snap.last_update_id = j["lastUpdateId"].get<uint64_t>();

    for (const auto& bid : j["bids"]) {
        snap.bids.push_back({
            std::stod(bid[0].get<std::string>()),
            std::stod(bid[1].get<std::string>())
        });
    }
    for (const auto& ask : j["asks"]) {
        snap.asks.push_back({
            std::stod(ask[0].get<std::string>()),
            std::stod(ask[1].get<std::string>())
        });
    }

    return snap;
}

BookClient::DepthUpdate BookClient::parseDepthUpdate(const std::string& payload) {
    nlohmann::json j = nlohmann::json::parse(payload);

    DepthUpdate update;
    update.event_time_ms   = j["E"].get<uint64_t>();
    update.first_update_id = j["U"].get<uint64_t>();
    update.last_update_id  = j["u"].get<uint64_t>();

    for (const auto& bid : j["b"]) {
        update.bids.push_back({
            std::stod(bid[0].get<std::string>()),
            std::stod(bid[1].get<std::string>())
        });
    }
    for (const auto& ask : j["a"]) {
        update.asks.push_back({
            std::stod(ask[0].get<std::string>()),
            std::stod(ask[1].get<std::string>())
        });
    }

    return update;
}

bool BookClient::reconcileAndApply(const Snapshot& snap) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);

    const uint64_t L = snap.last_update_id;

    // STEP 1: drop diffs whose entire range is at or before L (already
    // reflected in the snapshot)
    auto first_useful = std::find_if(buffer_.begin(), buffer_.end(),
        [L](const DepthUpdate& u) { return u.last_update_id > L; });
    buffer_.erase(buffer_.begin(), first_useful);

    if (buffer_.empty()) {
        // snapshot is more recent than every buffered diff - wait for more
        // diffs to arrive (the resync loop will retry)
        return false;
    }

    // STEP 2: the first remaining diff must contain L+1, so applying it picks
    // up exactly where the snapshot left off
    const DepthUpdate& first = buffer_.front();
    if (!(first.first_update_id <= L + 1 && L + 1 <= first.last_update_id)) {
        // gap between snapshot's view and the diff stream - discard the
        // buffer and try again
        std::cerr << "[RESYNC] straddle check failed: L+1=" << (L + 1)
                  << " first diff U=" << first.first_update_id
                  << " u=" << first.last_update_id << "\n";
        buffer_.clear();
        return false;
    }

    // STEP 3: apply snapshot to a fresh book
    book_.clear();
    for (const auto& lv : snap.bids)
        book_.applyUpdate(lv.price, lv.quantity, true);
    for (const auto& lv : snap.asks)
        book_.applyUpdate(lv.price, lv.quantity, false);

    // STEP 4: replay buffered diffs in order
    for (const auto& up : buffer_) {
        for (const auto& lv : up.bids)
            book_.applyUpdate(lv.price, lv.quantity, true);
        for (const auto& lv : up.asks)
            book_.applyUpdate(lv.price, lv.quantity, false);
        book_.setLastUpdateTime(up.event_time_ms);
        last_applied_update_id_ = up.last_update_id;
    }
    buffer_.clear();

    return true;
}

void BookClient::triggerResync() {
    state_ = State::Desynced;

    std::thread([this]() {
        // move back to Buffering so any new diffs get queued during the
        // resync (resyncOnThread will then transition us to Synced on
        // success)
        state_ = State::Buffering;
        resyncOnThread();
    }).detach();
}
