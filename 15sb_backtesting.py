"""
bitunix_backtest_v15sb.py
strategy    : EMA20/60/120 + MACD golden cross
sl          : ATR(14) * 1.5 動態止損
tp          : Volume Profile 局部量峰（上方第一個）
手續費       : Bitunix VIP0 taker 0.06% * 2
資金費率     : 固定 0.0574%/8h(當前費率)
alpha/beta  : 相對 ETH buy-and-hold 日報酬
"""

import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime

BASE_URL   = "https://fapi.bitunix.com"
CACHE_FILE = "klines_cache.csv"

SYMBOL         = "ETHUSDT"
INTERVAL       = "1m"
QTY            = 0.01
PAGES          = 2628
FEE_RATE       = 0.0006    # VIP0 taker 0.06%，開+平共 0.12%
FUND_RATE      = 0.000574  # 當前資金費率 0.0574%/8h

# ATR 止損參數（參考 KhanSaab Pine Script）
ATR_PERIOD     = 14
ATR_MULTIPLIER = 1.5

# Volume Profile 參數（參考 AlgoAlpha 鯨魚）
VP_LOOKBACK    = 200   # 建 Profile 用的回看根數
VP_BINS        = 35    # 分價區間數量

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

def fetch_all_klines(symbol, interval, pages=PAGES):
    if os.path.exists(CACHE_FILE):
        print(f"發現快取檔案，從斷點繼續...")
        df_cache = pd.read_csv(CACHE_FILE)
        df_cache["time"] = df_cache["time"].astype(int)
        all_data = df_cache.to_dict("records")
        end_time = df_cache["time"].min()
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
        print(f" {p+1} pages /{pages}，共 {len(all_data)} 根，剩餘約 {eta:.1f} 分鐘", end="\r")
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
# 指標計算
# ----------------------------------------------

def add_indicators(df):
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # EMA 趨勢
    df["ema20"]  = c.ewm(span=20,  adjust=False).mean()
    df["ema60"]  = c.ewm(span=60,  adjust=False).mean()
    df["ema120"] = c.ewm(span=120, adjust=False).mean()

    # MACD histogram
    ema12      = c.ewm(span=12, adjust=False).mean()
    ema26      = c.ewm(span=26, adjust=False).mean()
    macd       = ema12 - ema26
    signal     = macd.ewm(span=9, adjust=False).mean()
    df["hist"] = macd - signal

    # ATR
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr"]  = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    df["risk"] = df["atr"] * ATR_MULTIPLIER  # 止損距離

    return df

# ----------------------------------------------
# Volume Profile：找上方第一個量峰
# 邏輯參考 AlgoAlpha Whale Liquidity absorption peak
# ----------------------------------------------

def find_vp_tp(df, current_idx, entry_price):
    """
    用 current_idx 前 VP_LOOKBACK 根 K 線建 Volume Profile，
    找 entry_price 上方第一個局部量峰作為 TP。
    找不到時回傳 None。
    """
    start = max(0, current_idx - VP_LOOKBACK)
    window = df.iloc[start:current_idx]

    if len(window) < 10:
        return None

    min_p = window["low"].min()
    max_p = window["high"].max()
    if max_p <= min_p:
        return None

    step = (max_p - min_p) / VP_BINS
    bins = np.zeros(VP_BINS)

    # 每根 K 線的成交量平均分配到 [low, high] 覆蓋的區間
    for _, row in window.iterrows():
        lo, hi, vol = row["low"], row["high"], float(row.get("baseVol", 0) or 0)
        if vol <= 0:
            continue
        bin_lo = int((lo - min_p) / step)
        bin_hi = int((hi - min_p) / step)
        bin_lo = max(0, min(VP_BINS - 1, bin_lo))
        bin_hi = max(0, min(VP_BINS - 1, bin_hi))
        span = bin_hi - bin_lo + 1
        for b in range(bin_lo, bin_hi + 1):
            bins[b] += vol / span

    # 找 entry_price 上方的局部量峰（左右鄰居都比它小）
    entry_bin = int((entry_price - min_p) / step)
    entry_bin = max(0, min(VP_BINS - 1, entry_bin))

    for i in range(entry_bin + 1, VP_BINS - 1):
        if bins[i] > bins[i - 1] and bins[i] > bins[i + 1]:
            # 量峰區間中點作為 TP
            tp_price = min_p + step * (i + 0.5)
            if tp_price > entry_price:
                return tp_price

    return None

# ----------------------------------------------
# 資金費率
# ----------------------------------------------

def calc_funding_fee(entry_time_ms, exit_time_ms, notional):
    hold_hours  = (exit_time_ms - entry_time_ms) / 1000 / 3600
    periods     = int(hold_hours / 8)
    funding_fee = periods * FUND_RATE * notional
    return funding_fee

# ----------------------------------------------
# 回測
# ----------------------------------------------

def backtest(df):
    trades   = []
    position = None

    # 用於計算 alpha/beta 的逐日報酬
    df["returns"]          = df["close"].pct_change()
    df["strategy_returns"] = 0.0

    for i in range(122, len(df)):
        row   = df.iloc[i]
        prev  = df.iloc[i - 1]
        prev2 = df.iloc[i - 2]

        if position:
            high = row["high"]
            low  = row["low"]

            hit_tp = high >= position["tp"] if position["tp"] else False
            hit_sl = low  <= position["sl"]

            if hit_tp or hit_sl:
                exit_price   = position["tp"] if hit_tp else position["sl"]
                exit_time_ms = int(row["time"])
                notional     = position["entry_price"] * QTY

                pnl_pct  = (exit_price - position["entry_price"]) / position["entry_price"]
                pnl_raw  = pnl_pct * notional
                fee      = notional * FEE_RATE * 2
                funding  = calc_funding_fee(position["entry_time"], exit_time_ms, notional)
                net_pnl  = pnl_raw - fee - funding

                # 策略報酬寫回 df（用於 alpha/beta）
                df.at[i, "strategy_returns"] = pnl_pct

                trades.append({
                    "entry_time": datetime.fromtimestamp(position["entry_time"] / 1000).strftime("%m-%d %H:%M"),
                    "exit_time" : datetime.fromtimestamp(exit_time_ms / 1000).strftime("%m-%d %H:%M"),
                    "side"      : position["side"],
                    "entry"     : round(position["entry_price"], 2),
                    "exit"      : round(exit_price, 2),
                    "tp_level"  : round(position["tp"], 2) if position["tp"] else "ATR×2",
                    "result"    : "TP" if hit_tp else "SL",
                    "pnl_pct"  : round(pnl_pct * 100, 3),
                    "pnl_raw"  : round(pnl_raw, 4),
                    "fee"      : round(fee, 4),
                    "funding"  : round(funding, 4),
                    "net_pnl"  : round(net_pnl, 4),
                })
                position = None
            continue

        # 進場條件：EMA 多頭排列 + MACD 金叉
        ma_ok      = prev["ema20"] > prev["ema60"] > prev["ema120"]
        macd_cross = prev["hist"] > 0 and prev2["hist"] < 0

        if ma_ok and macd_cross:
            entry = float(row["close"])
            risk  = float(row["risk"])  # ATR × 1.5

            # TP：Volume Profile 量峰，找不到退用 ATR × 2
            tp = find_vp_tp(df, i, entry)
            if tp is None:
                tp = entry + risk * 2

            position = {
                "side"       : "BUY",
                "entry_price": entry,
                "tp"         : tp,
                "sl"         : entry - risk,   # ATR 動態止損
                "entry_time" : int(row["time"]),
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

    total      = len(trades)
    wins       = [t for t in trades if t["result"] == "TP"]
    losses     = [t for t in trades if t["result"] == "SL"]
    win_rate   = len(wins) / total * 100
    total_net  = sum(t["net_pnl"]  for t in trades)
    total_fee  = sum(t["fee"]      for t in trades)
    total_fund = sum(t["funding"]  for t in trades)
    avg_win    = sum(t["net_pnl"]  for t in wins)   / len(wins)   if wins   else 0
    avg_loss   = sum(t["net_pnl"]  for t in losses) / len(losses) if losses else 0

    # Alpha / Beta（相對 ETH buy-and-hold 日報酬）
    # 只取有交易的 bar 避免大量 0 稀釋
    strat = df["strategy_returns"].replace(0, np.nan).dropna()
    bench = df["returns"].loc[strat.index]
    bench_var = bench.var()

    if bench_var > 0 and len(strat) > 1:
        beta  = strat.cov(bench) / bench_var
        # 年化（365 天，crypto 無休市）
        alpha = (strat.mean() - beta * bench.mean()) * 365
    else:
        beta, alpha = float("nan"), float("nan")

    print(f"\n{'='*64}")
    print(f"{SYMBOL}  {INTERVAL}  {start} ~ {end}")
    print(f"SL : ATR({ATR_PERIOD}) × {ATR_MULTIPLIER}   TP : Volume Profile 量峰")
    print(f"手續費：VIP0 taker {FEE_RATE*100}% × 2  資金費率：{FUND_RATE*100}%/8h")
    print(f"{'='*64}")
    print(f"總交易次數  : {total}")
    print(f"勝率        : {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"總淨盈虧    : {total_net:+.4f} USDT")
    print(f"總手續費    : -{total_fee:.4f} USDT")
    print(f"總資金費率  : -{total_fund:.4f} USDT")
    print(f"平均盈利    : {avg_win:+.4f} USDT")
    print(f"平均虧損    : {avg_loss:+.4f} USDT")
    if avg_loss != 0:
        print(f"盈虧比      : {abs(avg_win/avg_loss):.2f}")
    print(f"Beta        : {beta:.3f}")
    print(f"Alpha (年化): {alpha:.6f}")
    print(f"{'='*64}")

    # 存 CSV
    df_trades = pd.DataFrame(trades)
    df_trades.to_csv("backtest_result.csv", index=False, encoding="utf-8-sig")
    print(f"\n結果已存至 backtest_result.csv")

    print(f"\n  {'進場':<14} {'出場':<14} {'TP位':<9} {'結果':<4} {'盈虧%':<8} {'原始':<9} {'手續費':<9} {'資金費':<9} 淨盈虧")
    print(f"  {'-'*90}")
    for t in trades[-20:]:
        print(f"  {t['entry_time']:<14} {t['exit_time']:<14} {str(t['tp_level']):<9} {t['result']:<4} "
              f"{t['pnl_pct']:>+6.3f}%  {t['pnl_raw']:>+8.4f} "
              f"-{t['fee']:<8.4f} -{t['funding']:<8.4f} {t['net_pnl']:>+.4f}")

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
        df["baseVol"] = pd.to_numeric(df.get("baseVol", 0), errors="coerce").fillna(0)
        df = df.sort_values("time").reset_index(drop=True)
        print(f"共 {len(df)} 根 K 線")

        ans = input("重新拉取數據？(Y/N): ").strip().lower()
        if ans == "y":
            os.remove(CACHE_FILE)
            df = fetch_all_klines(SYMBOL, INTERVAL, pages=PAGES)
    else:
        print(f"拉取 {SYMBOL} {INTERVAL} 一年數據（約需 {PAGES*0.15/60:.0f} 分鐘）...")
        df = fetch_all_klines(SYMBOL, INTERVAL, pages=PAGES)

    print(f"\n計算指標...")
    df = add_indicators(df)

    print(f"開始回測...")
    trades, df = backtest(df)

    print_results(trades, df)