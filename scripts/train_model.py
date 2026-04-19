# fetch real BTC/USDT trade data from Binance public archives (data.binance.vision),
# downsample it into a manageable tick stream, compute microstructure features, train an MLP,
# and export weights for the C engine.

"""
data source: Binance public data vault (data.binance.vision)

note: the spot/daily archive contains `trades` for BTCUSDT, but does not provide
historical `bookTicker` files under data/spot/daily/bookTicker/... (404/empty).
this script therefore trains from real trades, and approximates bid/ask and
bid_vol/ask_vol from recent trade prices/flow so the C engine schema stays the same.
"""


### IMPORT LIBRARIES
import os, io, zipfile, datetime, urllib.request
import numpy as np


### CONFIGURATIONS
SYMBOL = "BTCUSDT"
DAYS_BACK = 3  # how many calendar days to pull
SAMPLE_EVERY = 50  # keep every Nth trade to keep the dataset manageable (e.g. 50 -> ~300k ticks)

THRESHOLD = 5e-5  # 0.005% price change → UP or DOWN label
LOOKAHEAD = 10  # ticks forward to measure change (after downsampling)
WIN = 10  # rolling window for features (must match engine.h NIN=10)

FLOW_VOL_WIN = 200  # rolling window (ticks) for buy/sell volume proxy
SPREAD_RET_WIN = 200  # rolling window (ticks) for spread proxy from abs returns

BASE = "https://data.binance.vision/data/spot/daily"
os.makedirs("data", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)


### DOWNLOAD HELPERS

def binance_url(kind, symbol, date):  # kind = 'trades'
    d = date.strftime("%Y-%m-%d")
    return f"{BASE}/{kind}/{symbol}/{symbol}-{kind}-{d}.zip"

# download a zip from Binance data vault, return CSV bytes inside
def fetch_zip_csv(url):
    fname = os.path.join("data/raw", url.split("/")[-1])
    if not os.path.exists(fname):
        print(f"  downloading {url.split('/')[-1]} ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, fname)
            print("ok")
        except Exception as e:
            print(f"FAILED ({e})")
            return None
    else:
        print(f"  cached: {fname.split('/')[-1]}")
    with zipfile.ZipFile(fname) as z:
        name = z.namelist()[0]
        return z.read(name)


### FETCH AND PARSE
#
# trades columns (no header):
#   0  trade_id
#   1  price
#   2  qty
#   3  quote_qty
#   4  time              (ms)
#   5  is_buyer_maker    (true = seller initiated, false = buyer initiated)

def parse_trades(raw):
    # check if file has a header line (newer Binance files do)
    first_line = raw[:200].decode("utf-8", errors="replace").split("\n")[0]
    skip = 1 if first_line and first_line[0].isalpha() else 0

    # numeric cols: price (1), qty (2), time_ms (4)
    num = np.genfromtxt(
        io.BytesIO(raw),
        delimiter = ",",
        skip_header = skip,
        usecols = (1, 2, 4),
        dtype = np.float64,
        invalid_raise = False,
    )
    if num.ndim == 1:
        num = num.reshape(1, -1)

    price = num[:, 0]
    qty = num[:, 1]
    time_ms = num[:, 2]

    # side col: parse only the last field using rsplit
    raw_text = raw.decode("utf-8", errors="replace")
    lines = raw_text.splitlines()[skip:]
    is_bm = np.empty(len(lines), dtype = np.int8)

    for i, line in enumerate(lines):
        parts = line.rsplit(",", 1)
        if len(parts) != 2:
            is_bm[i] = 1
            continue
        is_bm[i] = 1 if parts[1].strip().lower() == "true" else 0

    side = np.where(is_bm == 1, -1.0, 1.0)

    # align lengths defensively
    n = min(len(time_ms), len(side))
    return time_ms[:n], price[:n], qty[:n], side[:n]


today = datetime.date.today()
dates = [today - datetime.timedelta(days = i + 1) for i in range(DAYS_BACK)]
dates.reverse()  # chronological order

all_time = []
all_px = []
all_qty = []
all_side = []

for date in dates:
    print(f"\n── {date} ──")

    raw = fetch_zip_csv(binance_url("trades", SYMBOL, date))
    if raw is None:
        continue

    time_ms, price, qty, side = parse_trades(raw)

    all_time.append(time_ms)
    all_px.append(price)
    all_qty.append(qty)
    all_side.append(side)

    print(f"  trade rows: {len(time_ms):,}")

if not all_time:
    raise SystemExit("No trade data downloaded. Check data.binance.vision availability.")

time_ms = np.concatenate(all_time)
trade_px = np.concatenate(all_px)
trade_vol = np.concatenate(all_qty)
side = np.concatenate(all_side)

# sort by time (should already be sorted, but just in case)
ordr = np.argsort(time_ms)
time_ms = time_ms[ordr]
trade_px = trade_px[ordr]
trade_vol = trade_vol[ordr]
side = side[ordr]

print(f"\ntotal trades: {len(time_ms):,}")

# downsample early (critical for speed)
time_ms = time_ms[::SAMPLE_EVERY]
trade_px = trade_px[::SAMPLE_EVERY]
trade_vol = trade_vol[::SAMPLE_EVERY]
side = side[::SAMPLE_EVERY]

N = len(time_ms)
print(f"ticks: {N:,}  (downsampled by SAMPLE_EVERY = {SAMPLE_EVERY})")


### APPROXIMATE TOP-OF-BOOK FIELDS (PROXIES)
# we approximate:
#   - bid/ask via a spread proxy from recent abs returns
#   - bid_vol/ask_vol via rolling buy vs sell volume

mid = trade_px.copy()

# spread proxy from rolling mean abs return
ret = np.empty(N, dtype = np.float64)
ret[0] = 0.0
ret[1:] = np.abs(np.diff(mid) / (mid[:-1] + 1e-12))

absret = ret
ps = np.cumsum(absret)

roll_mean_absret = np.empty(N, dtype = np.float64)
for i in range(N):
    j0 = 0 if i < SPREAD_RET_WIN else i - SPREAD_RET_WIN
    s = ps[i] - (ps[j0 - 1] if j0 > 0 else 0.0)
    roll_mean_absret[i] = s / (i - j0 + 1)

spread = mid * np.clip(2.0 * roll_mean_absret, 1e-4, 1e-3)  # 1bp..10bp
bid_px = mid - spread / 2.0
ask_px = mid + spread / 2.0

# rolling buy/sell volume via prefix sums
buy_vol = np.where(side > 0, trade_vol, 0.0)
sell_vol = np.where(side < 0, trade_vol, 0.0)

buy_ps = np.cumsum(buy_vol)
sell_ps = np.cumsum(sell_vol)

bid_qty = np.empty(N, dtype = np.float64)
ask_qty = np.empty(N, dtype = np.float64)
for i in range(N):
    j0 = 0 if i < FLOW_VOL_WIN else i - FLOW_VOL_WIN
    bid_qty[i] = buy_ps[i] - (buy_ps[j0 - 1] if j0 > 0 else 0.0)
    ask_qty[i] = sell_ps[i] - (sell_ps[j0 - 1] if j0 > 0 else 0.0)

bid_qty = np.maximum(bid_qty, 1e-6)
ask_qty = np.maximum(ask_qty, 1e-6)


### WRITE CSV
# write ts in seconds so main.c output is readable/consistent.
print("\nwriting data/ticks.csv ...", end = " ", flush = True)
with open("data/ticks.csv", "w") as f:
    f.write("ts,bid,ask,bid_vol,ask_vol,trade_px,trade_vol,side\n")
    for i in range(N):
        ts_s = time_ms[i] / 1000.0
        f.write(f"{ts_s:.3f},"
                f"{bid_px[i]:.6f},{ask_px[i]:.6f},"
                f"{bid_qty[i]:.6f},{ask_qty[i]:.6f},"
                f"{trade_px[i]:.6f},{trade_vol[i]:.6f},"
                f"{int(side[i])}\n")
print("done")


### COMPUTE FEATURES
mid = (bid_px + ask_px) / 2.0

def compute_features(i):
    if i < WIN:
        return None

    m0 = mid[i]; m1 = mid[i - 1]; m5 = mid[i - 5]; m9 = mid[i - 9]

    tv = bid_qty[i] + ask_qty[i]
    book_imb = (bid_qty[i] - ask_qty[i]) / tv if tv > 0 else 0.0
    flow = float(np.sum(side[i - WIN + 1:i + 1] * trade_vol[i - WIN + 1:i + 1]))
    vol_r = bid_qty[i] / ask_qty[i] if ask_qty[i] > 0 else 1.0

    w_px = mid[i - WIN + 1:i + 1]
    w_vol = trade_vol[i - WIN + 1:i + 1]
    vwap_d = w_vol.sum()
    vwap = (w_px * w_vol).sum() / vwap_d if vwap_d > 0 else m0

    return [
        (m0 - m1) / (m1 + 1e-10),  # 1-tick return
        (m0 - m5) / (m5 + 1e-10),  # 5-tick return
        float(ask_px[i] - bid_px[i]),  # spread (proxy)
        (ask_px[i] - bid_px[i]) / (m0 + 1e-10),  # normalized spread
        book_imb,  # book imbalance (proxy)
        flow,  # signed trade flow (real)
        vol_r,  # volume ratio (proxy)
        (m0 - m9) / (m9 + 1e-10),  # 10-tick momentum
        float(WIN),  # tick intensity
        (m0 - vwap) / (vwap + 1e-10),  # VWAP deviation
    ]

print("computing features ...", end = " ", flush = True)
X_list, y_list = [], []

for i in range(WIN, N - LOOKAHEAD):
    feat = compute_features(i)
    if feat is None:
        continue
    m_now = mid[i]
    m_fut = mid[i + LOOKAHEAD]
    change = (m_fut - m_now) / (m_now + 1e-10)

    if change > THRESHOLD: label = 2  # UP
    elif change < -THRESHOLD: label = 0  # DOWN
    else: label = 1  # FLAT

    X_list.append(feat)
    y_list.append(label)

X = np.array(X_list, dtype = np.float32)
y = np.array(y_list, dtype = np.int64)

counts = {0: int(np.sum(y == 0)), 1: int(np.sum(y == 1)), 2: int(np.sum(y == 2))}
majority_baseline = max(counts.values()) / len(y)
print("done")
print(f"samples: {len(X):,}  |  DOWN = {counts[0]:,}  FLAT = {counts[1]:,}  UP = {counts[2]:,}")
print(f"label balance: DOWN = {counts[0] / len(y):.1%}  FLAT = {counts[1] / len(y):.1%}  UP = {counts[2] / len(y):.1%}")

if counts[1] / len(y) > 0.85:
    print(f"NOTE: FLAT-heavy labels ({counts[1] / len(y):.0%}). "
          f"try lowering THRESHOLD (currently {THRESHOLD:.1e}) or increasing LOOKAHEAD.")


### NORMALIZE
mean = X.mean(0)
std = X.std(0)
std[std < 1e-8] = 1.0
X_norm = (X - mean) / std


### TRAIN
try:
    import torch
    import torch.nn as nn

    X_t = torch.tensor(X_norm)
    y_t = torch.tensor(y)
    ds = torch.utils.data.TensorDataset(X_t, y_t)
    dl = torch.utils.data.DataLoader(ds, batch_size = 512, shuffle = True)

    model = nn.Sequential(
        nn.Linear(10, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 3)
    )
    opt = torch.optim.Adam(model.parameters(), lr = 1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print("\ntraining (PyTorch) ...")
    for epoch in range(30):
        total = 0
        for xb, yb in dl:
            opt.zero_grad()
            l = loss_fn(model(xb), yb)
            l.backward(); opt.step()
            total += l.item()
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1:3d}  loss = {total / len(dl):.4f}")

    with torch.no_grad():
        pred = model(X_t).argmax(1)
        acc = (pred == y_t).float().mean().item()
    print(f"train accuracy: {acc:.3f}  (majority baseline: {majority_baseline:.3f})")

    layers = [p.detach().numpy() for p in model.parameters()]
    W1, b1, W2, b2, W3, b3 = layers

except ImportError:
    print("\nPyTorch not found. falling back to sklearn MLP.")
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes = (32, 16), activation = "relu", max_iter = 50, random_state = 0, verbose = False)
    clf.fit(X_norm, y)
    acc = clf.score(X_norm, y)
    print(f"train accuracy: {acc:.3f}  (majority baseline: {majority_baseline:.3f})")
    W1 = clf.coefs_[0].T.astype(np.float32)
    b1 = clf.intercepts_[0].astype(np.float32)
    W2 = clf.coefs_[1].T.astype(np.float32)
    b2 = clf.intercepts_[1].astype(np.float32)
    W3 = clf.coefs_[2].T.astype(np.float32)
    b3 = clf.intercepts_[2].astype(np.float32)


### EXPORT BINARY TO C
def write_bin(path, *arrays):
    with open(path, "wb") as f:
        for a in arrays:
            f.write(np.asarray(a, dtype = np.float32).tobytes())

write_bin("data/weights.bin", W1, b1, W2, b2, W3, b3)
write_bin("data/normstats.bin", mean, std)
print(f"\nwrote data/weights.bin    ({os.path.getsize('data/weights.bin'):,} bytes)")
print(f"wrote data/normstats.bin  ({os.path.getsize('data/normstats.bin'):,} bytes)")
print()
print("next steps:")
print("  make")
print("  ./build/hft_engine data/ticks.csv data/weights.bin data/normstats.bin")
print("  ./build/hft_engine_simd data/ticks.csv data/weights.bin data/normstats.bin --simd")