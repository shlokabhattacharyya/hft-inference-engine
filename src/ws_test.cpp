#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXHttpClient.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <atomic>
#include <csignal>
#include <chrono>
#include <thread>
#include <iomanip>
#include <vector>
#include "order_book.h"

struct PriceLevel {
    double price;
    double quantity;
};

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

static DepthUpdate parseDepthUpdate(const std::string& message) {
    nlohmann::json j = nlohmann::json::parse(message);

    DepthUpdate update;
    update.event_time_ms = j["E"].get<uint64_t>();
    update.first_update_id = j["U"].get<uint64_t>();
    update.last_update_id = j["u"].get<uint64_t>();

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

static Snapshot fetchSnapshot(const std::string& symbol) {
    ix::HttpClient client;

    auto args = client.createRequest();
    args->connectTimeout = 10;
    args->transferTimeout = 10;

    std::string url = "https://api.binance.us/api/v3/depth?symbol="
                      + symbol + "&limit=1000";

    auto response = client.get(url, args);

    if (response->statusCode != 200) {
        throw std::runtime_error("Snapshot HTTP " +
            std::to_string(response->statusCode) +
	    " errorMsg='" + response->errorMsg + "'" +
	    " body='" + response->body + "'");
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

std::atomic<bool> g_running{true};

void signalHandler(int) {
    g_running = false;
}

int main() {
    std::signal(SIGINT, signalHandler);

    ix::WebSocket ws;
    ws.setUrl("wss://stream.binance.us:9443/ws/btcusdt@depth");

    ws.setPingInterval(30);

    OrderBook book;

    ws.setOnMessageCallback([&book](const ix::WebSocketMessagePtr& msg) {
        switch (msg->type) {
            case ix::WebSocketMessageType::Message:
    	        try {
        	    auto update = parseDepthUpdate(msg->str);

        	    for (const auto& level : update.bids) {
                        book.applyUpdate(level.price, level.quantity, /*is_bid=*/true);
        	    }
        	    for (const auto& level : update.asks) {
            	 	book.applyUpdate(level.price, level.quantity, /*is_bid=*/false);
        	    }
        	    book.setLastUpdateTime(update.event_time_ms);

        	    if (book.isReady()) {
            		std::cout << std::fixed << std::setprecision(2)
                      	    << "Book: bid=" << book.bestBid()
                            << " (" << std::setprecision(3) << book.bidVolume(1) << " BTC)"
                      	    << " | ask=" << std::setprecision(2) << book.bestAsk()
                      	    << " (" << std::setprecision(3) << book.askVolume(1) << " BTC)"
                      	    << " | spread=" << std::setprecision(2) << book.spread()
                      	    << "\n";
        	    }
    		} catch (const std::exception& e) {
        	    std::cerr << "[parse error] " << e.what() << "\n";
    		}
    		break;
            case ix::WebSocketMessageType::Open:
                std::cout << "[OPEN] Connected to " << msg->openInfo.uri << "\n";
                break;
            case ix::WebSocketMessageType::Close:
                std::cout << "[CLOSE] code=" << msg->closeInfo.code
                          << " reason=" << msg->closeInfo.reason << "\n";
                break;
            case ix::WebSocketMessageType::Error:
                std::cerr << "[ERROR] " << msg->errorInfo.reason << "\n";
                break;
            case ix::WebSocketMessageType::Ping:
            case ix::WebSocketMessageType::Pong:
                break;
            default:
                break;
        }
    });

    ws.start();
    std::cout << "Started. Ctrl-C to stop.\n";

    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    ws.stop();
    std::cout << "Stopped cleanly.\n";
    return 0;
}
