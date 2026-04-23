#include <ixwebsocket/IXWebSocket.h>
#include <iostream>
#include <atomic>
#include <csignal>
#include <chrono>
#include <thread>

std::atomic<bool> g_running{true};

void signalHandler(int) {
    g_running = false;
}

int main() {
    std::signal(SIGINT, signalHandler);

    ix::WebSocket ws;
    ws.setUrl("wss://stream.binance.us:9443/ws/btcusdt@depth");

     ws.setPingInterval(30);

    ws.setOnMessageCallback([](const ix::WebSocketMessagePtr& msg) {
        switch (msg->type) {
            case ix::WebSocketMessageType::Message:
                std::cout << "[MSG] " << msg->str.substr(0, 200)
                          << (msg->str.size() > 200 ? "..." : "") << "\n";
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
