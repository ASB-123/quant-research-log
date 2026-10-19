"""
bitunix_backtest_v15sc.py
K線        : 5m
strategy   : EMA20/60/120 + MACD golden cross
sl         : ATR(14) × 1.5 動態止損，隨 TP 觸發上移
tp         : Volume Profile 多層量峰分批出場
             TP1 → 平 30%，SL 移到 entry（保本）
             TP2 → 平 30%，SL 移到 TP1
             TP3 → 平剩餘 40%
手續費      : Bitunix VIP0 taker 0.06% × 2（每次部分平倉各算一次）
資金費率    : 固定 0.0574%/8h
alpha/beta : 相對 ETH buy-and-hold
"""

import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime

BASE_URL   = "https://fapi.bitunix.com"
CACHE_FILE = "klines_5m_cache.csv"

SYMBOL         = "ETHUSDT"
INTERVAL       = "5m"
QTY            = 0.01          # 總倉位（ETH）
PAGES          = 2628          # 5m 一年約 105120 根，每頁 200，需 526 頁；留 2628 相容 1m 快取
PAGES_5M       = 530           # 5m 實際頁數
FEE_RATE       = 0.0006        # VIP0 taker 0.06%
FUND_RATE      = 0.000574      # 0.0574%/8h

ATR_PERIOD     = 14
ATR_MULTIPLIER = 1.5

VP_LOOKBACK    = 200           # 建 Profile 用的回看根數（5m K 線）
VP_BINS        = 35

# 分批出場比例
TP1_RATIO = 0.30
TP2_RATIO = 0.30
TP3_RATIO = 0.40   # 剩餘全平

# ----------------------------------------------
# 數據抓取
# ----------------------------------------------

def fetch_klines(symbol, interval, limit=200, end_time=None):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time:
        params["endTime"] = end_time
    for attempt in range(5):
        try:
            res = requests.get(f"{BASE_URL}/api/v1/futures/market/kline",
                               params=params, timeout=30)
            data = res.json().get("data", [])
            return data
        except Exception as e:
            wait = (attempt + 1) * 5
            print(f"\n  第{attempt+1}次重試，等待{wait}秒：{e}")
            time.sleep(wait)
    return []

def fetch_all_klines(symbol, interval, pages):
    if os.path.exists(CACHE_FILE):
        print(f"發現快取 {CACHE_FILE}，從斷點繼續...")
        df_cache = pd.read_csv(CACHE_FILE)
        df_cache["time"] = df_cache["time"].astype(int)
        all_data   = df_cache.to_dict("records")
        end_time   = df_cache["time"].min()
        start_page = len(all_data) // 200
        print(f"已有 {len(all_data)} 根，從第 {start_page} 頁繼續")
    else:
        all_data   = []
        end_time   = None
        start_page = 0

    for p in range(start_page, pages):
        batch = fetch_klines(symbol, interval, limit=200, end_time=end_time)
        if not batch:
            print(f"\n  第{p+1}頁無數據，停止")
            break
        all_data.extend(batch)
        end_time = batch[-1]["time"]

        if (p + 1) % 50 == 0:
            df_tmp = pd.DataFrame(all_data).drop_duplicates(subset="time")
            df_tmp.to_csv(CACHE_FILE, index=False)
            print(f"\n  已存快取：{len(df_tmp)} 根")

        eta = (pages - p - 1) * 0.15 / 60
        print(f" {p+1}/{pages} pages，共 {len(all_data)} 根，剩餘約 {eta:.1f} 分鐘", end="\r")
        time.sleep(0.15)

    df = pd.DataFrame(all_data).drop_duplicates(subset="time")
    df["time"]  = df["time"].astype(int)
    df["close"] = df["close"].astype(float)
    df["high"]  = df["high"].astype(float)
    df["low"]   = df["low"].astype(float)
    df = df.sort_values("time").reset_index(drop=True)
    df.to_csv(CACHE_FILE, index=False)
    print(f"\n數據擷取完成，已存至 {CACHE_FILE}")
    return df

# ----------------------------------------------
# 指標
# ----------------------------------------------

def add_indicators(df):
    c = df["close"]
    h = df["high"]
    l = df["low"]

    df["ema20"]  = c.ewm(span=20,  adjust=False).mean()
    df["ema60"]  = c.ewm(span=60,  adjust=False).mean()
    df["ema120"] = c.ewm(span=120, adjust=False).mean()

    ema12      = c.ewm(span=12, adjust=False).mean()
    ema26      = c.ewm(span=26, adjust=False).mean()
    macd       = ema12 - ema26
    signal     = macd.ewm(span=9, adjust=False).mean()
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
# Volume Profile：找多個量峰
# ----------------------------------------------

def find_vp_peaks(df, current_idx, entry_price, n=3):
    """
    找 entry_price 上方的前 n 個局部量峰，由近到遠回傳 list。
    找不到時對應位置填 None。
    """
    start  = max(0, current_idx - VP_LOOKBACK)
    window = df.iloc[start:current_idx]

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
        vol = float(row.get("baseVol", 0) or 0)
        if vol <= 0:
            continue
        b_lo = max(0, min(VP_BINS - 1, int((lo - min_p) / step)))
        b_hi = max(0, min(VP_BINS - 1, int((hi - min_p) / step)))
        span = b_hi - b_lo + 1
        for b in range(b_lo, b_hi + 1):
            bins[b] += vol / span

    entry_bin = max(0, min(VP_BINS - 1, int((entry_price - min_p) / step)))
    peaks = []
    for i in range(entry_bin + 1, VP_BINS - 1):
        if bins[i] > bins[i - 1] and bins[i] > bins[i + 1]:
            tp_price = min_p + step * (i + 0.5)
            if tp_price > entry_price:
                peaks.append(tp_price)
        if len(peaks) >= n:
            break

    # 不足 n 個補 None
    while len(peaks) < n:
        peaks.append(None)
    return peaks

# ----------------------------------------------
# 資金費率
# ----------------------------------------------

def calc_funding_fee(entry_time_ms, exit_time_ms, notional):
    hold_hours = (exit_time_ms - entry_time_ms) / 1000 / 3600
    periods    = int(hold_hours / 8)
    return periods * FUND_RATE * notional

# ----------------------------------------------
# 回測（分批出場）
# ----------------------------------------------

def backtest(df):
    trades   = []
    position = None

    df["returns"]          = df["close"].pct_change()
    df["strategy_returns"] = 0.0

    for i in range(122, len(df)):
        row   = df.iloc[i]
        prev  = df.iloc[i - 1]
        prev2 = df.iloc[i - 2]

        if position:
            high = float(row["high"])
            low  = float(row["low"])
            t    = int(row["time"])

            # ── SL 檢查（用當前移動後的 SL）──
            if low <= position["sl"]:
                # 剩餘倉位全部止損
                remaining_qty  = position["remaining_qty"]
                exit_price     = position["sl"]
                notional       = position["entry_price"] * remaining_qty
                pnl_pct        = (exit_price - position["entry_price"]) / position["entry_price"]
                pnl_raw        = pnl_pct * notional
                fee            = notional * FEE_RATE * 2
                funding        = calc_funding_fee(position["entry_time"], t, notional)
                net_pnl        = pnl_raw - fee - funding

                # 加上已實現盈虧
                total_net_pnl  = net_pnl + position["realized_pnl"]
                df.at[i, "strategy_returns"] = pnl_pct

                trades.append({
                    "entry_time"  : datetime.fromtimestamp(position["entry_time"] / 1000).strftime("%m-%d %H:%M"),
                    "exit_time"   : datetime.fromtimestamp(t / 1000).strftime("%m-%d %H:%M"),
                    "side"        : "BUY",
                    "entry"       : round(position["entry_price"], 2),
                    "exit"        : round(exit_price, 2),
                    "tp1"         : round(position["tp1"], 2) if position["tp1"] else "-",
                    "tp2"         : round(position["tp2"], 2) if position["tp2"] else "-",
                    "tp3"         : round(position["tp3"], 2) if position["tp3"] else "-",
                    "tp_hit"      : position["tp_hit"],
                    "result"      : "SL",
                    "net_pnl"     : round(total_net_pnl, 4),
                    "fee_total"   : round(fee + position["fee_paid"], 4),
                    "fund_total"  : round(funding + position["fund_paid"], 4),
                })
                position = None
                continue

            # ── TP 分批出場 ──
            tp_triggered = False

            # TP1
            if position["tp_hit"] == 0 and position["tp1"] and high >= position["tp1"]:
                qty_close      = QTY * TP1_RATIO
                notional       = position["entry_price"] * qty_close
                pnl_raw        = (position["tp1"] - position["entry_price"]) * qty_close
                fee            = notional * FEE_RATE * 2
                funding        = calc_funding_fee(position["entry_time"], t, notional)
                net             = pnl_raw - fee - funding

                position["realized_pnl"]   += net
                position["fee_paid"]        += fee
                position["fund_paid"]       += funding
                position["remaining_qty"]  -= qty_close
                position["tp_hit"]          = 1
                position["sl"]             = position["entry_price"]  # 移到保本
                tp_triggered = True

            # TP2
            if position["tp_hit"] == 1 and position["tp2"] and high >= position["tp2"]:
                qty_close      = QTY * TP2_RATIO
                notional       = position["entry_price"] * qty_close
                pnl_raw        = (position["tp2"] - position["entry_price"]) * qty_close
                fee            = notional * FEE_RATE * 2
                funding        = calc_funding_fee(position["entry_time"], t, notional)
                net             = pnl_raw - fee - funding

                position["realized_pnl"]   += net
                position["fee_paid"]        += fee
                position["fund_paid"]       += funding
                position["remaining_qty"]  -= qty_close
                position["tp_hit"]          = 2
                position["sl"]             = position["tp1"]  # 移到 TP1
                tp_triggered = True

            # TP3（全平）
            if position["tp_hit"] == 2 and position["tp3"] and high >= position["tp3"]:
                qty_close      = position["remaining_qty"]
                notional       = position["entry_price"] * qty_close
                pnl_raw        = (position["tp3"] - position["entry_price"]) * qty_close
                fee            = notional * FEE_RATE * 2
                funding        = calc_funding_fee(position["entry_time"], t, notional)
                net             = pnl_raw - fee - funding

                total_net_pnl  = net + position["realized_pnl"]
                df.at[i, "strategy_returns"] = (position["tp3"] - position["entry_price"]) / position["entry_price"]

                trades.append({
                    "entry_time"  : datetime.fromtimestamp(position["entry_time"] / 1000).strftime("%m-%d %H:%M"),
                    "exit_time"   : datetime.fromtimestamp(t / 1000).strftime("%m-%d %H:%M"),
                    "side"        : "BUY",
                    "entry"       : round(position["entry_price"], 2),
                    "exit"        : round(position["tp3"], 2),
                    "tp1"         : round(position["tp1"], 2) if position["tp1"] else "-",
                    "tp2"         : round(position["tp2"], 2) if position["tp2"] else "-",
                    "tp3"         : round(position["tp3"], 2) if position["tp3"] else "-",
                    "tp_hit"      : 3,
                    "result"      : "TP3",
                    "net_pnl"     : round(total_net_pnl, 4),
                    "fee_total"   : round(fee + position["fee_paid"], 4),
                    "fund_total"  : round(funding + position["fund_paid"], 4),
                })
                position = None
                tp_triggered = True

            continue

        # ── 進場條件：EMA 多頭排列 + MACD 金叉 ──
        ma_ok      = prev["ema20"] > prev["ema60"] > prev["ema120"]
        macd_cross = prev["hist"] > 0 and prev2["hist"] < 0

        if ma_ok and macd_cross:
            entry = float(row["close"])
            risk  = float(row["risk"])

            peaks = find_vp_peaks(df, i, entry, n=3)
            tp1, tp2, tp3 = peaks

            # 找不到的 TP 用 ATR 倍數補
            if tp1 is None: tp1 = entry + risk * 1.5
            if tp2 is None: tp2 = entry + risk * 2.5
            if tp3 is None: tp3 = entry + risk * 4.0

            position = {
                "side"          : "BUY",
                "entry_price"   : entry,
                "sl"            : entry - risk,
                "tp1"           : tp1,
                "tp2"           : tp2,
                "tp3"           : tp3,
                "tp_hit"        : 0,           # 0=未打到, 1=TP1, 2=TP2
                "remaining_qty" : QTY,
                "realized_pnl"  : 0.0,
                "fee_paid"      : 0.0,
                "fund_paid"     : 0.0,
                "entry_time"    : int(row["time"]),
            }

    return trades, df

# ----------------------------------------------
# 輸出
# ----------------------------------------------

def print_results(trades, df):
    start = datetime.fromtimestamp(df["time"].iloc[0]  / 1000).strftime("%Y-%m-%d")
    end   = datetime.fromtimestamp(df["time"].iloc[-1] / 1000).strftime("%Y-%m-%d")

    if not trades:
        print(f"回測期間 {start} ~ {end}，無交易信號")
        return

    total     = len(trades)
    wins      = [t for t in trades if t["result"] != "SL"]
    losses    = [t for t in trades if t["result"] == "SL"]
    win_rate  = len(wins) / total * 100
    total_net = sum(t["net_pnl"]   for t in trades)
    total_fee = sum(t["fee_total"] for t in trades)
    total_fund= sum(t["fund_total"]for t in trades)
    avg_win   = sum(t["net_pnl"] for t in wins)   / len(wins)   if wins   else 0
    avg_loss  = sum(t["net_pnl"] for t in losses) / len(losses) if losses else 0

    tp1_count = sum(1 for t in trades if t["tp_hit"] >= 1)
    tp2_count = sum(1 for t in trades if t["tp_hit"] >= 2)
    tp3_count = sum(1 for t in trades if t["tp_hit"] >= 3)

    # Alpha / Beta
    strat = df["strategy_returns"].replace(0, np.nan).dropna()
    bench = df["returns"].loc[strat.index]
    bench_var = bench.var()
    if bench_var > 0 and len(strat) > 1:
        beta  = strat.cov(bench) / bench_var
        alpha = (strat.mean() - beta * bench.mean()) * 365
    else:
        beta, alpha = float("nan"), float("nan")

    print(f"\n{'='*66}")
    print(f"{SYMBOL}  {INTERVAL}  {start} ~ {end}")
    print(f"SL : ATR({ATR_PERIOD}) × {ATR_MULTIPLIER}  TP : VP量峰 分批 30/30/40%")
    print(f"手續費：VIP0 taker {FEE_RATE*100}% × 2  資金費率：{FUND_RATE*100}%/8h")
    print(f"{'='*66}")
    print(f"總交易次數  : {total}")
    print(f"勝率        : {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"TP層次分佈  : TP1={tp1_count}  TP2={tp2_count}  TP3={tp3_count}  SL={len(losses)}")
    print(f"總淨盈虧    : {total_net:+.4f} USDT")
    print(f"總手續費    : -{total_fee:.4f} USDT")
    print(f"總資金費率  : -{total_fund:.4f} USDT")
    print(f"平均盈利    : {avg_win:+.4f} USDT")
    print(f"平均虧損    : {avg_loss:+.4f} USDT")
    if avg_loss != 0:
        print(f"盈虧比      : {abs(avg_win/avg_loss):.2f}")
    print(f"Beta        : {beta:.3f}")
    print(f"Alpha (年化): {alpha:.6f}")
    print(f"{'='*66}")

    df_trades = pd.DataFrame(trades)
    out = "backtest_result_v18.csv"
    try:
        df_trades.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n結果已存至 {out}")
    except PermissionError:
        print(f"\n[!] {out} 被佔用，請關閉 Excel 後重試")

# ----------------------------------------------
# 主程序
# ----------------------------------------------

if __name__ == "__main__":
    if os.path.exists(CACHE_FILE):
        print(f"載入快取 {CACHE_FILE}...")
        df = pd.read_csv(CACHE_FILE)
        df["time"]    = df["time"].astype(int)
        df["close"]   = df["close"].astype(float)
        df["high"]    = df["high"].astype(float)
        df["low"]     = df["low"].astype(float)
        df["baseVol"] = pd.to_numeric(df.get("baseVol", pd.Series(dtype=float)), errors="coerce").fillna(0)
        df = df.sort_values("time").reset_index(drop=True)
        print(f"共 {len(df)} 根 K 線")

        ans = input("重新拉取數據？(Y/N): ").strip().lower()
        if ans == "y":
            os.remove(CACHE_FILE)
            df = fetch_all_klines(SYMBOL, INTERVAL, pages=PAGES_5M)
    else:
        print(f"拉取 {SYMBOL} {INTERVAL} 一年數據（約需 {PAGES_5M*0.15/60:.1f} 分鐘）...")
        df = fetch_all_klines(SYMBOL, INTERVAL, pages=PAGES_5M)

    print(f"\n計算指標...")
    df = add_indicators(df)

    print(f"開始回測...")
    trades, df = backtest(df)

    print_results(trades, df)