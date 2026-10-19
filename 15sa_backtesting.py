"""
bitunix_backtest_v15sa.py
strategy    :20EMA > 60EMA > 120EMA + MACD 金叉
sl/tp       :TP 1.5% / SL 0.8%
手續費       :Bitunix VIP0 taker 0.06% * 2
資金費率     :固定 0.0574%/8h(當前費率)
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime

BASE_URL   = "https://fapi.bitunix.com"
CACHE_FILE = "klines_cache.csv"

SYMBOL    = "ETHUSDT"
INTERVAL  = "1m"
QTY       = 0.01
TP_PCT    = 0.015
SL_PCT    = 0.008
PAGES     = 2628
FEE_RATE  = 0.0006   # VIP0 taker 0.06%，開+平共 0.12%
FUND_RATE = 0.000574 # 當前資金費率 0.0574%/8h

# ----------------------------------------------
# statistics
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
# indicator
# ----------------------------------------------

def add_indicators(df):
    c = df["close"]
    df["ema20"]  = c.ewm(span=20,  adjust=False).mean()
    df["ema60"]  = c.ewm(span=60,  adjust=False).mean()
    df["ema120"] = c.ewm(span=120, adjust=False).mean()

    ema12      = c.ewm(span=12, adjust=False).mean()
    ema26      = c.ewm(span=26, adjust=False).mean()
    macd       = ema12 - ema26
    signal     = macd.ewm(span=9, adjust=False).mean()
    df["hist"] = macd - signal
    return df

# ----------------------------------------------
# backtest
# ----------------------------------------------

def calc_funding_fee(entry_time_ms, exit_time_ms, notional):
    hold_hours  = (exit_time_ms - entry_time_ms) / 1000 / 3600
    periods     = int(hold_hours / 8)
    funding_fee = periods * FUND_RATE * notional
    return funding_fee

def backtest(df):
    trades   = []
    position = None

    for i in range(122, len(df)):
        row   = df.iloc[i]
        prev  = df.iloc[i - 1]
        prev2 = df.iloc[i - 2]

        if position:
            high = row["high"]
            low  = row["low"]

            if position["side"] == "BUY":
                hit_tp = high >= position["tp"]
                hit_sl = low  <= position["sl"]
            else:
                hit_tp = low  <= position["tp"]
                hit_sl = high >= position["sl"]

            if hit_tp or hit_sl:
                exit_price   = position["tp"] if hit_tp else position["sl"]
                exit_time_ms = int(row["time"])
                notional     = position["entry_price"] * QTY

                pnl_pct = (exit_price - position["entry_price"]) / position["entry_price"]
                if position["side"] == "SELL":
                    pnl_pct = -pnl_pct
                pnl_raw  = pnl_pct * notional
                fee      = notional * FEE_RATE * 2
                funding  = calc_funding_fee(position["entry_time"], exit_time_ms, notional)
                net_pnl  = pnl_raw - fee - funding

                trades.append({
                    "entry_time": datetime.fromtimestamp(position["entry_time"] / 1000).strftime("%m-%d %H:%M"),
                    "exit_time" : datetime.fromtimestamp(exit_time_ms / 1000).strftime("%m-%d %H:%M"),
                    "side"      : position["side"],
                    "entry"     : round(position["entry_price"], 2),
                    "exit"      : round(exit_price, 2),
                    "result"    : "TP" if hit_tp else "SL",
                    "pnl_pct"  : round(pnl_pct * 100, 3),
                    "pnl_raw"  : round(pnl_raw, 4),
                    "fee"      : round(fee, 4),
                    "funding"  : round(funding, 4),
                    "net_pnl"  : round(net_pnl, 4),
                })
                position = None
            continue

        ma_ok      = prev["ema20"] > prev["ema60"] > prev["ema120"]
        macd_cross = prev["hist"] > 0 and prev2["hist"] < 0

        if ma_ok and macd_cross:
            entry = row["close"]
            position = {
                "side"       : "BUY",
                "entry_price": entry,
                "tp"         : entry * (1 + TP_PCT),
                "sl"         : entry * (1 - SL_PCT),
                "entry_time" : int(row["time"]),
            }

    return trades

# ----------------------------------------------
# output
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

    print(f"\n{'='*62}")
    print(f"{SYMBOL}  {INTERVAL}  {start} ~ {end}")
    print(f"TP:{TP_PCT*100}%  SL:{SL_PCT*100}%  QTY:{QTY}")
    print(f"手續費：VIP0 taker {FEE_RATE*100}% × 2  資金費率：{FUND_RATE*100}%/8h")
    print(f"{'='*62}")
    print(f"總交易次數  : {total}")
    print(f"勝率        : {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"總淨盈虧    : {total_net:+.4f} USDT")
    print(f"總手續費    : -{total_fee:.4f} USDT")
    print(f"總資金費率  : -{total_fund:.4f} USDT")
    print(f"平均盈利    : {avg_win:+.4f} USDT")
    print(f"平均虧損    : {avg_loss:+.4f} USDT")
    if avg_loss != 0:
        print(f"  盈虧比      : {abs(avg_win/avg_loss):.2f}")
    print(f"{'='*62}")

    # 存 CSV
    df_trades = pd.DataFrame(trades)
    df_trades.to_csv("backtest_result.csv", index=False, encoding="utf-8-sig")

    print(f"\n  {'進場':<14} {'出場':<14} {'結果':<4} {'盈虧%':<8} {'原始':<9} {'手續費':<9} {'資金費':<9} 淨盈虧")
    print(f"  {'-'*85}")
    for t in trades[-20:]:
        print(f"  {t['entry_time']:<14} {t['exit_time']:<14} {t['result']:<4} "
              f"{t['pnl_pct']:>+6.3f}%  {t['pnl_raw']:>+8.4f} "
              f"-{t['fee']:<8.4f} -{t['funding']:<8.4f} {t['net_pnl']:>+.4f}")

if __name__ == "__main__":
    if os.path.exists(CACHE_FILE):
        print(f"載入快取 {CACHE_FILE}...")
        df = pd.read_csv(CACHE_FILE)
        df["time"]  = df["time"].astype(int)
        df["close"] = df["close"].astype(float)
        df["high"]  = df["high"].astype(float)
        df["low"]   = df["low"].astype(float)
        df = df.sort_values("time").reset_index(drop=True)
        print(f"共 {len(df)} 根 K 線")

        ans = input("重新拉取數據？(y/N): ").strip().lower()
        if ans == "y":
            os.remove(CACHE_FILE)
            df = fetch_all_klines(SYMBOL, INTERVAL, pages=PAGES)
    else:
        print(f"拉取 {SYMBOL} {INTERVAL} 一年數據（約需 {PAGES*0.15/60:.0f} 分鐘）...")
        df = fetch_all_klines(SYMBOL, INTERVAL, pages=PAGES)

    print(f"\n計算指標...")
    df     = add_indicators(df)
    trades = backtest(df)
    print_results(trades, df)