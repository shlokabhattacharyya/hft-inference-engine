# hft inference engine

a low-latency (sub-400ns) inference engine in C for market microstructure prediction. trained in Python on real crypto data via Binance, deployed in C with AVX2 SIMD, benchmarked with hardware cycle counters. a 3-layer MLP (10→32→16→3) trained offline on market-style tick data, then deployed as a standalone C inference engine. the model predicts short-term price direction (UP / FLAT / DOWN) from 10 microstructure features.

## features

ten features computed over a rolling 10-tick window:

| feature | Intuition |
|---|---|
| 1-tick mid-price return | immediate momentum |
| 5-tick mid-price return | short-term trend |
| bid-ask spread | liquidity cost / uncertainty |
| normalized spread | spread relative to price level |
| book imbalance | directional pressure (`(bid_vol - ask_vol) / total_vol`) |
| signed trade flow | buyer/seller initiated volume over window |
| volume ratio | queue skew (`bid_vol / ask_vol`) |
| 10-tick momentum | longer trend signal |
| tick intensity | activity rate |
| VWAP deviation | price relative to recent execution average |


## installation:

1. clone this repository:
```
git clone https://github.com/shlokabhattacharyya/hft-inference-engine.git
cd hft-inference-engine
```
2. install dependencies:
```
pip install -r requirements.txt
```
3. fetch data and train:
```
python train_model.py
```
4. build both engines:
```
make
```
5. run scalar:
```
./build/hft_engine data/ticks.csv data/weights.bin data/normstats.bin
```
6. run SIMD:
```
./build/hft_engine_simd data/ticks.csv data/weights.bin data/normstats.bin --simd
```

output per tick: `timestamp signal [p_down p_flat p_up]`