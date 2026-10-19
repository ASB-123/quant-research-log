"""
bitunix_backtest_v19.py
幣種        : ETHUSDT / BTCUSDT / SOLUSDT（三幣並行，合併統計）
進場 K線    : 5m
方向過濾    : close vs EMA20（多/空二選一，移除三線排列滯後）
進場觸發    : MACD(7,10,6) histogram 金叉/死叉
倉位權重    : |交叉時histogram| / ATR 越小權重越高（越接近0軸=越早期訊號）
              weight = max(0.3, 1/(1+hist_norm))
              qty = NOTIONAL_BASE * weight / entry
TP 參考     : 15m Volume Profile 量峰
              TP1 ≥ entry × 0.16%，平 60%，SL → entry
              TP2 下一個量峰，平 25%，SL → TP1
              TP3 下一個量峰，平剩餘 15%
SL          : ATR(14) × 1.5，隨 TP 觸發移動
多倉        : 同幣種多筆並行，訊號各自獨立
手續費      : VIP0 taker 0.06%（開倉一次 + 每次部分平倉一次）
資金費率    : 固定 0.0574%/8h
Alpha/Beta  : 基準按各幣交易次數加權
"""

import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime

BASE_URL  = "https://fapi.bitunix.com"

SYMBOLS       = ["ETHUSDT", "BTCUSDT", "SOLUSDT"]
NOTIONAL_BASE = 25.0       # 滿倉(weight=1)時的名義倉位 USDT
WEIGHT_FLOOR  = 0.3
PAGES_5M      = 1060       # 約 2 年 5m 數據
PAGES_15M     = 360        # 約 2 年 15m 數據
FEE_RATE      = 0.0006
FUND_RATE     = 0.000574

ATR_PERIOD     = 14
ATR_MULTIPLIER = 1.5

MACD_FAST   = 7
MACD_SLOW   = 10
MACD_SIGNAL = 6

VP_LOOKBACK = 200
VP_BINS     = 35
TP1_MIN_PCT = 0.0016

TP1_RATIO = 0.60
TP2_RATIO = 0.25
TP3_RATIO = 0.15

# ----------------------------------------------
# 數據抓取
# ----------------------------------------------

def fetch_klines(symbol, interval, limit=200, end_time=None):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time:
        params["endTime"] = end_time
    for attempt in range(5):
        try:
            res  = requests.get(f"{BASE_URL}/api/v1/futures/market/kline",
                                params=params, timeout=30)
            data = res.json().get("data", [])
            return data
        except Exception as e:
            wait = (attempt + 1) * 5
            print(f"\n  重試 {attempt+1}，等待 {wait}s：{e}")
            time.sleep(wait)
    return []

def fetch_all_klines(symbol, interval, pages):
    cache = f"cache_{symbol}_{interval}.csv"
    if os.path.exists(cache):
        df_c = pd.read_csv(cache)
        df_c["time"] = df_c["time"].astype(int)
        all_data   = df_c.to_dict("records")
        end_time   = df_c["time"].min()
        start_page = len(all_data) // 200
        print(f"  [{symbol} {interval}] 快取 {len(all_data)} 根，從第 {start_page} 頁繼續")
    else:
        all_data, end_time, start_page = [], None, 0

    for p in range(start_page, pages):
        batch = fetch_klines(symbol, interval, limit=200, end_time=end_time)
        if not batch:
            break
        all_data.extend(batch)
        end_time = batch[-1]["time"]
        if (p + 1) % 50 == 0:
            pd.DataFrame(all_data).drop_duplicates("time").to_csv(cache, index=False)
        print(f"  [{symbol} {interval}] {p+1}/{pages}", end="\r")
        time.sleep(0.15)

    df = pd.DataFrame(all_data).drop_duplicates("time")
    for col in ["time", "close", "high", "low", "open"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["baseVol"] = pd.to_numeric(df.get("baseVol", 0), errors="coerce").fillna(0)
    df = df.sort_values("time").reset_index(drop=True)
    df.to_csv(cache, index=False)
    print(f"\n  [{symbol} {interval}] 完成，共 {len(df)} 根")
    return df

# ----------------------------------------------
# 指標（5m）
# ----------------------------------------------

def add_indicators(df):
    c = df["close"]
    h = df["high"]
    l = df["low"]

    df["ema20"] = c.ewm(span=20, adjust=False).mean()

    ema_fast   = c.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow   = c.ewm(span=MACD_SLOW, adjust=False).mean()
    macd       = ema_fast - ema_slow
    signal     = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["hist"] = macd - signal

    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr"]  = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    df["risk"] = df["atr"] * ATR_MULTIPLIER
    return df

# ----------------------------------------------
# Volume Profile（15m）：找多個量峰
# ----------------------------------------------

def find_vp_peaks(df15, ref_time_ms, entry_price, side, n=3):
    window = df15[df15["time"] < ref_time_ms].tail(VP_LOOKBACK)
    if len(window) < 10:
        return [None] * n

    min_p = window["low"].min()
    max_p = window["high"].max()
    if max_p <= min_p:
        return [None] * n

    step = (max_p - min_p) / VP_BINS
    bins = np.zeros(VP_BINS)

    for _, row in window.iterrows():
        lo  = row["low"]
        hi  = row["high"]
        vol = float(row["baseVol"] or 0)
        if vol <= 0:
            continue
        b_lo = max(0, min(VP_BINS - 1, int((lo - min_p) / step)))
        b_hi = max(0, min(VP_BINS - 1, int((hi - min_p) / step)))
        span = b_hi - b_lo + 1
        for b in range(b_lo, b_hi + 1):
            bins[b] += vol / span

    peaks = []
    min_tp1 = entry_price * (1 + TP1_MIN_PCT) if side == "BUY" else entry_price * (1 - TP1_MIN_PCT)

    if side == "BUY":
        entry_bin = max(0, min(VP_BINS - 1, int((entry_price - min_p) / step)))
        for i in range(entry_bin + 1, VP_BINS - 1):
            if bins[i] > bins[i - 1] and bins[i] > bins[i + 1]:
                tp = min_p + step * (i + 0.5)
                if len(peaks) == 0 and tp < min_tp1:
                    continue
                if tp > entry_price:
                    peaks.append(tp)
            if len(peaks) >= n:
                break
    else:
        entry_bin = max(0, min(VP_BINS - 1, int((entry_price - min_p) / step)))
        for i in range(entry_bin - 1, 0, -1):
            if bins[i] > bins[i - 1] and bins[i] > bins[i + 1]:
                tp = min_p + step * (i + 0.5)
                if len(peaks) == 0 and tp > min_tp1:
                    continue
                if tp < entry_price:
                    peaks.append(tp)
            if len(peaks) >= n:
                break

    while len(peaks) < n:
        peaks.append(None)
    return peaks

# ----------------------------------------------
# 資金費率
# ----------------------------------------------

def calc_funding_fee(entry_ms, exit_ms, notional):
    periods = int((exit_ms - entry_ms) / 1000 / 3600 / 8)
    return periods * FUND_RATE * notional

# ----------------------------------------------
# 單幣回測（MACD權重倉位，多空雙向，多倉並行）
# ----------------------------------------------

def backtest_symbol(symbol, df5, df15):
    positions = []
    trades    = []

    df5 = df5.copy()
    df5["returns"]          = df5["close"].pct_change()
    df5["strategy_returns"] = 0.0

    for i in range(122, len(df5)):
        row   = df5.iloc[i]
        prev  = df5.iloc[i - 1]
        prev2 = df5.iloc[i - 2]
        high  = float(row["high"])
        low   = float(row["low"])
        t     = int(row["time"])

        # ── 更新所有持倉 ──
        closed_idx = []
        for pi, pos in enumerate(positions):
            side = pos["side"]

            sl_hit = (side == "BUY"  and low  <= pos["sl"]) or \
                     (side == "SELL" and high >= pos["sl"])

            if sl_hit:
                qty_r    = pos["remaining_qty"]
                notional = pos["entry_price"] * qty_r
                pnl_raw  = (pos["sl"] - pos["entry_price"]) * qty_r
                if side == "SELL":
                    pnl_raw = -pnl_raw
                fee     = notional * FEE_RATE
                funding = calc_funding_fee(pos["entry_time"], t, notional)
                net     = pnl_raw - fee - funding + pos["realized_pnl"]

                df5.at[i, "strategy_returns"] += \
                    (pos["sl"] - pos["entry_price"]) / pos["entry_price"] * (1 if side == "BUY" else -1)

                trades.append(_make_trade(symbol, pos, t, pos["sl"], "SL", net,
                                          fee + pos["fee_paid"], funding + pos["fund_paid"]))
                closed_idx.append(pi)
                continue

            tp1_hit = (side == "BUY"  and pos["tp_hit"] == 0 and pos["tp1"] and high >= pos["tp1"]) or \
                      (side == "SELL" and pos["tp_hit"] == 0 and pos["tp1"] and low  <= pos["tp1"])
            if tp1_hit:
                qty_c   = pos["initial_qty"] * TP1_RATIO
                notional= pos["entry_price"] * qty_c
                pnl_raw = abs(pos["tp1"] - pos["entry_price"]) * qty_c
                fee     = notional * FEE_RATE
                funding = calc_funding_fee(pos["entry_time"], t, notional)
                net     = pnl_raw - fee - funding
                pos["realized_pnl"]  += net
                pos["fee_paid"]      += fee
                pos["fund_paid"]     += funding
                pos["remaining_qty"] -= qty_c
                pos["tp_hit"]         = 1
                pos["sl"]             = pos["entry_price"]

            tp2_hit = (side == "BUY"  and pos["tp_hit"] == 1 and pos["tp2"] and high >= pos["tp2"]) or \
                      (side == "SELL" and pos["tp_hit"] == 1 and pos["tp2"] and low  <= pos["tp2"])
            if tp2_hit:
                qty_c   = pos["initial_qty"] * TP2_RATIO
                notional= pos["entry_price"] * qty_c
                pnl_raw = abs(pos["tp2"] - pos["entry_price"]) * qty_c
                fee     = notional * FEE_RATE
                funding = calc_funding_fee(pos["entry_time"], t, notional)
                net     = pnl_raw - fee - funding
                pos["realized_pnl"]  += net
                pos["fee_paid"]      += fee
                pos["fund_paid"]     += funding
                pos["remaining_qty"] -= qty_c
                pos["tp_hit"]         = 2
                pos["sl"]             = pos["tp1"]

            tp3_hit = (side == "BUY"  and pos["tp_hit"] == 2 and pos["tp3"] and high >= pos["tp3"]) or \
                      (side == "SELL" and pos["tp_hit"] == 2 and pos["tp3"] and low  <= pos["tp3"])
            if tp3_hit:
                qty_c   = pos["remaining_qty"]
                notional= pos["entry_price"] * qty_c
                pnl_raw = abs(pos["tp3"] - pos["entry_price"]) * qty_c
                fee     = notional * FEE_RATE
                funding = calc_funding_fee(pos["entry_time"], t, notional)
                net     = pnl_raw - fee - funding + pos["realized_pnl"]

                df5.at[i, "strategy_returns"] += \
                    abs(pos["tp3"] - pos["entry_price"]) / pos["entry_price"]

                trades.append(_make_trade(symbol, pos, t, pos["tp3"], "TP3", net,
                                          fee + pos["fee_paid"], funding + pos["fund_paid"]))
                closed_idx.append(pi)

        for pi in sorted(closed_idx, reverse=True):
            positions.pop(pi)

        # ── 進場條件：EMA20 方向過濾 + MACD(7,10,6) 交叉觸發 ──
        entry = float(row["close"])
        risk  = float(row["risk"])
        atr   = float(row["atr"])

        above_ema20 = entry > prev["ema20"]
        below_ema20 = entry < prev["ema20"]
        macd_cross  = prev["hist"] > 0 and prev2["hist"] <= 0   # 金叉
        macd_death  = prev["hist"] < 0 and prev2["hist"] >= 0   # 死叉

        for side, cond in [("BUY", above_ema20 and macd_cross), ("SELL", below_ema20 and macd_death)]:
            if not cond:
                continue
            if atr <= 0:
                continue

            # 倉位權重：交叉時 histogram 距 0 軸越近權重越高
            hist_cross_val = float(prev["hist"])
            hist_norm = abs(hist_cross_val) / atr
            weight    = max(WEIGHT_FLOOR, 1 / (1 + hist_norm))

            peaks       = find_vp_peaks(df15, t, entry, side, n=3)
            tp1, tp2, tp3 = peaks

            if side == "BUY":
                if tp1 is None: tp1 = entry + risk * 1.5
                if tp2 is None: tp2 = entry + risk * 2.5
                if tp3 is None: tp3 = entry + risk * 4.0
                sl = entry - risk
            else:
                if tp1 is None: tp1 = entry - risk * 1.5
                if tp2 is None: tp2 = entry - risk * 2.5
                if tp3 is None: tp3 = entry - risk * 4.0
                sl = entry + risk

            notional = NOTIONAL_BASE * weight
            qty      = notional / entry
            open_fee = notional * FEE_RATE

            positions.append({
                "symbol"        : symbol,
                "side"          : side,
                "entry_price"   : entry,
                "sl"            : sl,
                "tp1"           : tp1,
                "tp2"           : tp2,
                "tp3"           : tp3,
                "tp_hit"        : 0,
                "weight"        : weight,
                "initial_qty"   : qty,
                "remaining_qty" : qty,
                "realized_pnl"  : -open_fee,
                "fee_paid"      : open_fee,
                "fund_paid"     : 0.0,
                "entry_time"    : t,
            })

    return trades, df5


def _make_trade(symbol, pos, exit_time, exit_price, result, net_pnl, fee_total, fund_total):
    return {
        "symbol"    : symbol,
        "side"      : pos["side"],
        "weight"    : round(pos["weight"], 3),
        "entry_time": datetime.fromtimestamp(pos["entry_time"] / 1000).strftime("%m-%d %H:%M"),
        "exit_time" : datetime.fromtimestamp(exit_time / 1000).strftime("%m-%d %H:%M"),
        "entry"     : round(pos["entry_price"], 4),
        "exit"      : round(exit_price, 4),
        "tp1"       : round(pos["tp1"], 4) if pos["tp1"] else "-",
        "tp2"       : round(pos["tp2"], 4) if pos["tp2"] else "-",
        "tp3"       : round(pos["tp3"], 4) if pos["tp3"] else "-",
        "tp_hit"    : pos["tp_hit"],
        "result"    : result,
        "net_pnl"   : round(net_pnl, 4),
        "fee_total" : round(fee_total, 4),
        "fund_total": round(fund_total, 4),
    }

# ----------------------------------------------
# 輸出
# ----------------------------------------------

def print_results(all_trades, dfs):
    if not all_trades:
        print("無交易信號")
        return

    df_all     = pd.DataFrame(all_trades)
    sym_counts = df_all.groupby("symbol").size().to_dict()
    total      = len(df_all)

    wins   = df_all[df_all["net_pnl"] > 0]
    losses = df_all[df_all["net_pnl"] <= 0]

    tp1_hit = (df_all["tp_hit"] >= 1).sum()
    tp2_hit = (df_all["tp_hit"] >= 2).sum()
    tp3_hit = (df_all["result"] == "TP3").sum()

    total_net  = df_all["net_pnl"].sum()
    total_fee  = df_all["fee_total"].sum()
    total_fund = df_all["fund_total"].sum()
    avg_win    = wins["net_pnl"].mean()   if len(wins)   else 0
    avg_loss   = losses["net_pnl"].mean() if len(losses) else 0

    long_trades  = df_all[df_all["side"] == "BUY"]
    short_trades = df_all[df_all["side"] == "SELL"]

    strat_all, bench_all = [], []
    for sym, df5 in dfs.items():
        n     = sym_counts.get(sym, 0)
        strat = df5["strategy_returns"].replace(0, np.nan).dropna()
        bench = df5["returns"].loc[strat.index]
        strat_all.extend(strat.tolist())
        bench_all.extend((bench * n / total).tolist())

    strat_s   = pd.Series(strat_all)
    bench_s   = pd.Series(bench_all)
    bench_var = bench_s.var()
    if bench_var > 0 and len(strat_s) > 1:
        beta  = strat_s.cov(bench_s) / bench_var
        alpha = (strat_s.mean() - beta * bench_s.mean()) * 365
    else:
        beta, alpha = float("nan"), float("nan")

    print(f"\n{'='*68}")
    print(f"{' / '.join(SYMBOLS)}  5m進場(EMA20+MACD{MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL})  15m VP TP")
    print(f"基準倉位：{NOTIONAL_BASE} USDT × weight(下限{WEIGHT_FLOOR})  SL：ATR({ATR_PERIOD})×{ATR_MULTIPLIER}")
    print(f"手續費：VIP0 taker {FEE_RATE*100}%  資金費率：{FUND_RATE*100}%/8h")
    print(f"{'='*68}")
    for sym in SYMBOLS:
        n       = sym_counts.get(sym, 0)
        sym_net = df_all[df_all["symbol"] == sym]["net_pnl"].sum()
        sym_l   = (df_all[df_all["symbol"] == sym]["side"] == "BUY").sum()
        sym_s   = (df_all[df_all["symbol"] == sym]["side"] == "SELL").sum()
        print(f"  {sym:<10} 交易 {n:>4} 次（多 {sym_l} / 空 {sym_s}）  淨盈虧 {sym_net:>+.4f} USDT")
    print(f"{'─'*68}")
    print(f"總交易次數  : {total}（多 {len(long_trades)} / 空 {len(short_trades)}）")
    print(f"平均倉位權重: {df_all['weight'].mean():.3f}")
    print(f"勝率        : {len(wins)/total*100:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"TP層次分佈  : TP1={tp1_hit}  TP2={tp2_hit}  TP3={tp3_hit}  SL={len(losses)}")
    print(f"總淨盈虧    : {total_net:+.4f} USDT")
    print(f"總手續費    : -{total_fee:.4f} USDT")
    print(f"總資金費率  : -{total_fund:.4f} USDT")
    print(f"平均盈利    : {avg_win:+.4f} USDT")
    print(f"平均虧損    : {avg_loss:+.4f} USDT")
    if avg_loss != 0:
        print(f"盈虧比      : {abs(avg_win/avg_loss):.2f}")
    print(f"Beta        : {beta:.3f}")
    print(f"Alpha (年化): {alpha:.6f}")
    print(f"{'='*68}")

    out = "backtest_result_v19.csv"
    try:
        df_all.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n結果已存至 {out}")
    except PermissionError:
        print(f"\n[!] {out} 被佔用，請關閉 Excel")

# ----------------------------------------------
# 主程序
# ----------------------------------------------

if __name__ == "__main__":
    all_trades = []
    dfs        = {}

    for sym in SYMBOLS:
        print(f"\n=== {sym} ===")
        df5  = fetch_all_klines(sym, "5m",  PAGES_5M)
        df5  = add_indicators(df5)
        df15 = fetch_all_klines(sym, "15m", PAGES_15M)

        print(f"  回測中...")
        trades, df5 = backtest_symbol(sym, df5, df15)
        all_trades.extend(trades)
        dfs[sym] = df5
        print(f"  {sym} 完成，{len(trades)} 筆交易")

    print_results(all_trades, dfs)