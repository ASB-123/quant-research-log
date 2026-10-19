"""
bitunix_backtest_v20.py
幣種        : ETHUSDT / BTCUSDT / SOLUSDT（三幣並行，合併統計）
進場 K線    : 5m
進場觸發    : 收盤站上/跌破 15m VP POC（Point of Control，成交量最大價格區間）
              多單：5m close 收在 POC 上方
              空單：5m close 收在 POC 下方
SL          : 設在 POC 區間另一端（突破失敗回到 POC 內視為無效）
TP 參考     : 15m Volume Profile 量峰（POC 以外的局部量峰）
              TP1 ≥ entry × 0.16%，平 60%，SL → entry（套保）
              TP2 下一個量峰，平 25%，SL → TP1
              TP3 下一個量峰，平剩餘 15%
倉位        : 固定名義 NOTIONAL USDT
手續費      : VIP0 taker 0.06%（開倉一次 + 每次部分平倉一次）
資金費率    : 固定 0.0574%/8h
R倍數       : RISK = |Entry-Stop|×Size，R_Multiple = Result÷Risk
套保        : TP1觸發後 Hedged=Y，視為已鎖定不虧本金
統計        : 累積PnL / 勝率 / 平均R(不含套保) / 平均盈利單 / 平均虧損單 /
              平均盈虧比 / 夏普比率 / 最大回撤 / 套保策略數 / 套保後平均R
"""

import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime

BASE_URL  = "https://fapi.bitunix.com"

SYMBOLS   = ["ETHUSDT", "BTCUSDT", "SOLUSDT"]
NOTIONAL  = 25.0
PAGES_5M  = 1060
PAGES_15M = 360
FEE_RATE  = 0.0006
FUND_RATE = 0.000574

VP_LOOKBACK = 200
VP_BINS     = 35
TP1_MIN_PCT = 0.0016

TP1_RATIO = 0.60
TP2_RATIO = 0.25
TP3_RATIO = 0.15

ATR_PERIOD = 14   # 僅用於 risk 估算/找不到POC時兜底

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
# 指標：5m 只需 ATR 做兜底，15m 不需要額外指標
# ----------------------------------------------

def add_atr(df):
    c, h, l = df["close"], df["high"], df["low"]
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    return df

# ----------------------------------------------
# Volume Profile：建 bins，回傳 POC + 上方/下方量峰序列
# ----------------------------------------------

def build_vp_fast(lows, highs, vols, min_p, max_p):
    """向量化建 VP，lows/highs/vols 為 numpy array"""
    step = (max_p - min_p) / VP_BINS
    bins = np.zeros(VP_BINS)

    b_lo = np.clip(((lows - min_p) / step).astype(int), 0, VP_BINS - 1)
    b_hi = np.clip(((highs - min_p) / step).astype(int), 0, VP_BINS - 1)

    for lo, hi, vol in zip(b_lo, b_hi, vols):
        if vol <= 0:
            continue
        span = hi - lo + 1
        bins[lo:hi+1] += vol / span

    return bins, step

def precompute_vp_by_15m_bar(df15):
    """
    對每根 15m K 線收線時間點，預先算好對應的 VP（用該時點之前 VP_LOOKBACK 根 15m）。
    回傳 dict: {15m_bar_close_time: vp_dict}
    """
    lows  = df15["low"].values
    highs = df15["high"].values
    vols  = df15["baseVol"].values
    times = df15["time"].values

    vp_cache = {}
    n = len(df15)

    for idx in range(VP_LOOKBACK, n):
        start = idx - VP_LOOKBACK
        window_low  = lows[start:idx]
        window_high = highs[start:idx]
        window_vol  = vols[start:idx]

        min_p = window_low.min()
        max_p = window_high.max()
        if max_p <= min_p:
            continue

        bins, step = build_vp_fast(window_low, window_high, window_vol, min_p, max_p)

        poc_idx = int(np.argmax(bins))
        poc_lo  = min_p + step * poc_idx
        poc_hi  = min_p + step * (poc_idx + 1)

        vp_cache[int(times[idx])] = {
            "min_p": min_p, "max_p": max_p, "step": step, "bins": bins,
            "poc_lo": poc_lo, "poc_hi": poc_hi, "poc_mid": (poc_lo + poc_hi) / 2,
        }

    return vp_cache

def get_vp_for_time(vp_times_sorted, vp_cache, ref_time_ms):
    """二分搜尋找出 ref_time_ms 之前最近的一個已收線 15m VP"""
    import bisect
    pos = bisect.bisect_right(vp_times_sorted, ref_time_ms) - 1
    if pos < 0:
        return None
    return vp_cache[vp_times_sorted[pos]]

def vp_peaks_from(vp, entry_price, side, n=3):
    """從已建好的 vp 結構找 entry 之外方向的局部量峰"""
    if vp is None:
        return [None] * n
    bins, min_p, step = vp["bins"], vp["min_p"], vp["step"]
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
# 單幣回測：POC 突破進場
# ----------------------------------------------

def backtest_symbol(symbol, df5, df15):
    positions = []
    trades    = []

    df5 = df5.copy()
    df5 = add_atr(df5)
    df5["returns"]          = df5["close"].pct_change()
    df5["strategy_returns"] = 0.0

    print(f"    建立 VP 快取（15m POC，僅每根15m算一次）...")
    vp_cache = precompute_vp_by_15m_bar(df15)
    vp_times_sorted = sorted(vp_cache.keys())
    print(f"    VP 快取完成，共 {len(vp_times_sorted)} 個時間點")

    for i in range(30, len(df5)):
        row  = df5.iloc[i]
        prev = df5.iloc[i - 1]
        high = float(row["high"])
        low  = float(row["low"])
        t    = int(row["time"])
        close = float(row["close"])

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

                trades.append(_make_trade(symbol, pos, t, pos["sl"], "撞到止損" if pos["tp_hit"] == 0 else "撞到止損(已套保)",
                                          net, fee + pos["fee_paid"], funding + pos["fund_paid"]))
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
                pos["hedged"]         = True

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

                trades.append(_make_trade(symbol, pos, t, pos["tp3"], "撞到止盈",
                                          net, fee + pos["fee_paid"], funding + pos["fund_paid"]))
                closed_idx.append(pi)

        for pi in sorted(closed_idx, reverse=True):
            positions.pop(pi)

        # ── 進場條件：收盤站上/跌破 POC ──
        vp = get_vp_for_time(vp_times_sorted, vp_cache, t)
        if vp is None:
            continue

        prev_close = float(prev["close"])
        poc_mid    = vp["poc_mid"]
        poc_lo     = vp["poc_lo"]
        poc_hi     = vp["poc_hi"]

        # 突破訊號：前一根還在 POC 內/反方向，這一根收盤站到另一側
        long_break  = prev_close <= poc_hi and close > poc_hi
        short_break = prev_close >= poc_lo and close < poc_lo

        for side, cond in [("BUY", long_break), ("SELL", short_break)]:
            if not cond:
                continue

            entry = close
            sl    = poc_lo if side == "BUY" else poc_hi   # SL 設在 POC 另一端

            risk_dist = abs(entry - sl)
            if risk_dist <= 0:
                continue

            peaks = vp_peaks_from(vp, entry, side, n=3)
            tp1, tp2, tp3 = peaks
            atr = float(row["atr"]) if not np.isnan(row["atr"]) else risk_dist

            if side == "BUY":
                if tp1 is None: tp1 = entry + risk_dist * 1.5
                if tp2 is None: tp2 = entry + risk_dist * 2.5
                if tp3 is None: tp3 = entry + risk_dist * 4.0
            else:
                if tp1 is None: tp1 = entry - risk_dist * 1.5
                if tp2 is None: tp2 = entry - risk_dist * 2.5
                if tp3 is None: tp3 = entry - risk_dist * 4.0

            qty      = NOTIONAL / entry
            open_fee = NOTIONAL * FEE_RATE
            risk_usdt = risk_dist * qty

            positions.append({
                "symbol"        : symbol,
                "side"          : side,
                "entry_price"   : entry,
                "sl"            : sl,
                "tp1"           : tp1,
                "tp2"           : tp2,
                "tp3"           : tp3,
                "tp_hit"        : 0,
                "hedged"        : False,
                "initial_qty"   : qty,
                "remaining_qty" : qty,
                "realized_pnl"  : -open_fee,
                "fee_paid"      : open_fee,
                "fund_paid"     : 0.0,
                "entry_time"    : t,
                "risk_usdt"     : risk_usdt,
            })

    return trades, df5


def _make_trade(symbol, pos, exit_time, exit_price, exit_reason, net_pnl, fee_total, fund_total):
    risk = pos["risk_usdt"] if pos["risk_usdt"] > 0 else 1e-9
    r_multiple = net_pnl / risk
    return {
        "Time"      : datetime.fromtimestamp(pos["entry_time"] / 1000).strftime("%Y/%m/%d %H:%M"),
        "Symbol"    : symbol,
        "Direction" : "long" if pos["side"] == "BUY" else "short",
        "Entry"     : round(pos["entry_price"], 4),
        "Stop"      : round(pos["sl"] if pos["tp_hit"] == 0 else pos["entry_price"], 4),
        "Target"    : round(pos["tp1"], 4) if pos["tp1"] else "-",
        "Exit"      : round(exit_price, 4),
        "Size"      : round(pos["initial_qty"], 6),
        "進場原因"   : "5m收盤站上/跌破15m POC，視為流動性轉移突破",
        "出場原因"   : exit_reason,
        "RISK"      : round(risk, 4),
        "Result"    : round(net_pnl, 4),
        "R_Multiple": round(r_multiple, 4),
        "Hedged?"   : "Y" if pos["hedged"] else "N",
        "tp_hit"    : pos["tp_hit"],
        "exit_time" : datetime.fromtimestamp(exit_time / 1000).strftime("%Y/%m/%d %H:%M"),
        "fee_total" : round(fee_total, 4),
        "fund_total": round(fund_total, 4),
    }

# ----------------------------------------------
# 統計計算
# ----------------------------------------------

def compute_stats(df_all):
    df_all = df_all.sort_values("Time").reset_index(drop=True)
    df_all["cum_pnl"] = df_all["Result"].cumsum()

    total  = len(df_all)
    wins   = df_all[df_all["Result"] > 0]
    losses = df_all[df_all["Result"] <= 0]
    win_rate = len(wins) / total * 100 if total else 0

    # 不含套保的 R 倍數（Hedged == N 的單，也就是純粹一刀切 SL/TP 出場）
    not_hedged = df_all[df_all["Hedged?"] == "N"]
    avg_r_no_hedge = not_hedged["R_Multiple"].mean() if len(not_hedged) else 0

    hedged = df_all[df_all["Hedged?"] == "Y"]
    hedged_count = len(hedged)
    avg_r_hedged = hedged["R_Multiple"].mean() if hedged_count else 0

    avg_win  = wins["Result"].mean()   if len(wins)   else 0
    avg_loss = losses["Result"].mean() if len(losses) else 0
    avg_rr   = abs(avg_win / avg_loss) if avg_loss != 0 else float("nan")

    # 夏普比率：用逐筆 Result 報酬序列（未年化標準，年化用 sqrt(交易頻率)）
    returns = df_all["Result"]
    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(len(returns))
    else:
        sharpe = float("nan")

    # 最大回撤
    cum = df_all["cum_pnl"]
    running_max = cum.cummax()
    drawdown = cum - running_max
    max_dd = drawdown.min()

    return {
        "total": total,
        "win_rate": win_rate,
        "avg_r_no_hedge": avg_r_no_hedge,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_rr": avg_rr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "hedged_count": hedged_count,
        "avg_r_hedged": avg_r_hedged,
        "final_cum_pnl": cum.iloc[-1] if total else 0,
    }

# ----------------------------------------------
# 輸出
# ----------------------------------------------

def print_results(all_trades, dfs):
    if not all_trades:
        print("無交易信號")
        return

    df_all = pd.DataFrame(all_trades)
    stats  = compute_stats(df_all)

    sym_counts = df_all.groupby("Symbol").size().to_dict()
    total_fee  = df_all["fee_total"].sum()
    total_fund = df_all["fund_total"].sum()

    # Alpha / Beta
    strat_all, bench_all = [], []
    for sym, df5 in dfs.items():
        n     = sym_counts.get(sym, 0)
        strat = df5["strategy_returns"].replace(0, np.nan).dropna()
        bench = df5["returns"].loc[strat.index]
        strat_all.extend(strat.tolist())
        bench_all.extend((bench * n / stats["total"]).tolist())

    strat_s   = pd.Series(strat_all)
    bench_s   = pd.Series(bench_all)
    bench_var = bench_s.var()
    if bench_var > 0 and len(strat_s) > 1:
        beta  = strat_s.cov(bench_s) / bench_var
        alpha = (strat_s.mean() - beta * bench_s.mean()) * 365
    else:
        beta, alpha = float("nan"), float("nan")

    print(f"\n{'='*70}")
    print(f"{' / '.join(SYMBOLS)}  5m進場(POC突破)  15m VP TP+SL")
    print(f"倉位：{NOTIONAL} USDT/筆  手續費：VIP0 {FEE_RATE*100}%  資金費率：{FUND_RATE*100}%/8h")
    print(f"{'='*70}")
    for sym in SYMBOLS:
        n       = sym_counts.get(sym, 0)
        sym_net = df_all[df_all["Symbol"] == sym]["Result"].sum()
        sym_l   = (df_all[df_all["Symbol"] == sym]["Direction"] == "long").sum()
        sym_s   = (df_all[df_all["Symbol"] == sym]["Direction"] == "short").sum()
        print(f"  {sym:<10} 交易 {n:>4} 次（多 {sym_l} / 空 {sym_s}）  淨盈虧 {sym_net:>+.4f} USDT")
    print(f"{'─'*70}")
    print(f"總交易次數      : {stats['total']}")
    print(f"累積 PnL        : {stats['final_cum_pnl']:+.4f} USDT")
    print(f"勝率(%)         : {stats['win_rate']:.1f}%")
    print(f"平均R倍數(不含套保): {stats['avg_r_no_hedge']:+.4f}")
    print(f"平均盈利單      : {stats['avg_win']:+.4f} USDT")
    print(f"平均虧損單      : {stats['avg_loss']:+.4f} USDT")
    print(f"平均盈虧比      : {stats['avg_rr']:.2f}")
    print(f"夏普比率        : {stats['sharpe']:.3f}")
    print(f"最大回撤        : {stats['max_dd']:.4f} USDT")
    print(f"套保策略數      : {stats['hedged_count']}")
    print(f"套保後平均R倍數 : {stats['avg_r_hedged']:+.4f}")
    print(f"{'─'*70}")
    print(f"總手續費        : -{total_fee:.4f} USDT")
    print(f"總資金費率      : -{total_fund:.4f} USDT")
    print(f"Beta            : {beta:.3f}")
    print(f"Alpha (年化)    : {alpha:.6f}")
    print(f"{'='*70}")

    out = "backtest_result_v20sa.csv"
    cols = ["Time","Symbol","Direction","Entry","Stop","Target","Exit","Size",
            "進場原因","出場原因","RISK","Result","R_Multiple","Hedged?"]
    try:
        df_all[cols].to_csv(out, index=False, encoding="utf-8-sig")
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
        df15 = fetch_all_klines(sym, "15m", PAGES_15M)

        print(f"  回測中...")
        trades, df5 = backtest_symbol(sym, df5, df15)
        all_trades.extend(trades)
        dfs[sym] = df5
        print(f"  {sym} 完成，{len(trades)} 筆交易")

    print_results(all_trades, dfs)