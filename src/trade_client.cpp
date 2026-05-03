// Binance aggregate trade stream client. maintains a time-windowed buffer
// of recent trades for feature extraction.


#include "trade_client.h"

#include <nlohmann/json.hpp>

#include <cctype>
#include <iostream>
#include <stdexcept>

TradeClient::TradeClient(const std::string& symbol)
    : symbol_(symbol) {

    std::string lower;
    lower.reserve(symbol.size());
    for (char c : symbol)
        lower.push_back(std::tolower(static_cast<unsigned char>(c)));

    ws_.setUrl("wss://stream.binance.us:9443/ws/" + lower + "@aggTrade");
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
                std::cerr << "[TRADE WS ERROR] " << msg->errorInfo.reason << "\n";
                break;
            default:
                break;
        }
    });
}

TradeClient::~TradeClient() {
    shutting_down_ = true;
    ws_.stop();
}

void TradeClient::start() { ws_.start(); }
void TradeClient::stop()  { ws_.stop(); }

void TradeClient::onOpen() {
    std::cout << "[TRADE WS] Connected\n";
}

void TradeClient::onClose() {
    std::cout << "[TRADE WS] Disconnected\n";
}

void TradeClient::onMessage(const std::string& payload) {
    Trade trade;
    try {
        trade = parseTrade(payload);
    } catch (const std::exception& e) {
        std::cerr << "[TRADE parse error] " << e.what() << "\n";
        return;
    }

    std::lock_guard<std::mutex> lock(buffer_mutex_);
    buffer_.push_back(trade);
}

Trade TradeClient::parseTrade(const std::string& payload) {
    nlohmann::json j = nlohmann::json::parse(payload);

    Trade trade;
    trade.time_ms  = j["T"].get<uint64_t>();
    trade.price    = std::stod(j["p"].get<std::string>());
    trade.quantity = std::stod(j["q"].get<std::string>());
    bool is_buyer_maker = j["m"].get<bool>();
    trade.side = is_buyer_maker ? -1 : +1;

    return trade;
}

std::vector<Trade> TradeClient::snapshot() const {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    return std::vector<Trade>(buffer_.begin(), buffer_.end());
}

void TradeClient::pruneOlderThan(uint64_t cutoff_time_ms) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    while (!buffer_.empty() && buffer_.front().time_ms < cutoff_time_ms) {
        buffer_.pop_front();
    }
}

size_t TradeClient::bufferSize() const {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    return buffer_.size();
}
