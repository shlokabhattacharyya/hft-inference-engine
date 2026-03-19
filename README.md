# HFT Inference Engine

A low-latency (sub-400ns) inference engine in C for market microstructure prediction. Trained in Python on real crypto data via Binance, deployed in C with AVX2 SIMD, benchmarked with hardware cycle counters. A 3-layer MLP (10→32→16→3) trained offline on market-style tick data, then deployed as a standalone C inference engine. The model predicts short-term price direction (UP / FLAT / DOWN) from 10 microstructure features.

## Features

Ten features computed over a rolling 10-tick window:

| Feature | Intuition |
|---|---|
| 1-tick mid-price return | Immediate momentum |
| 5-tick mid-price return | Short-term trend |
| Bid-ask spread | Liquidity cost / uncertainty |
| Normalized spread | Spread relative to price level |
| Book imbalance | Directional pressure (`(bid_vol - ask_vol) / total_vol`) |
| Signed trade flow | Buyer/seller initiated volume over window |
| Volume ratio | Queue skew (`bid_vol / ask_vol`) |
| 10-tick momentum | Longer trend signal |
| Tick intensity | Activity rate |
| VWAP deviation | Price relative to recent execution average |


## Installation:

1. Clone this repository:
```
git clone https://github.com/shlokabhattacharyya/hft-inference-engine.git
cd hft-inference-engine
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Fetch data and train:
```
python train_model.py
```
4. Build both engines:
```
make
```
5. Run scalar:
```
./build/hft_engine data/ticks.csv data/weights.bin data/normstats.bin
```
6. Run SIMD:
```
./build/hft_engine_simd data/ticks.csv data/weights.bin data/normstats.bin --simd
```

Output per tick: `timestamp signal [p_down p_flat p_up]`