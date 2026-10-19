"""
bitunix_backtest_v15sg_merged_v3_macd7146.py
====================
基礎機制（EMA20/60/120三線排列判斷、TP/SL/滑價/資金費率/固定風險倉位）：
與 v15sd_merged_v2 完全相同，未改動。

[v3 —— 兩個策略性變更]

  1. MACD週期從 12/26/9 改成 7/14/6（更快，對短期動能更敏感）

  2. 加碼規則：MACD快線穿越0軸後，在同一段"水上"（或"水下"）期間內，
     第二次出現金叉（或死叉）時，倉位放大2倍（風險金額也等比例放大，
     即該筆risk_usdt = TARGET_RISK_USDT × 2，不是notional放大2倍但risk不變）。
     其餘次數的金叉/死叉（第1次、第3次...）維持原本1倍倉位不變。

[⚠ 資料窺探狀態說明 —— 誠實揭露，不要跳過]
  這是一個全新假說（更快週期+動能延續加碼），不是從這批資料的虧損交易反推
  出來的規則，設計本身不算資料窺探。但你手上2022~2026這整段快取資料，
  已經先後被原始訊號測試、反轉訊號測試、ADX regime診斷用過三輪。
  也就是說，這批資料作為"樣本外驗證"的乾淨程度已經耗盡——不管這次結果
  好不好看，都只能當探索性(exploratory)結果，不是確認性(confirmatory)結果。
  如果這版看起來有東西，下一步應該是：
    (a) 找全新資料（更早的歷史，如果Bitunix有的話）驗證一次，或
    (b) 直接進紙上交易(paper trading)，用還沒發生的真實資料做確認
  不要再用同一批2022-2026資料反覆調整這個新規則的細節（例如"要不要改成
  第三次加碼"、"倍數要不要調成1.5"），那樣做基本等於在同一份雜訊裡挖到
  地老天荒，遲早挖到一個看起來能用、實際上是雜訊的組合。

[原v2配套]，全部保留：equity curve、Sharpe/Sortino/Calmar/MaxDD、
stationary bootstrap CI、Beta/Alpha（equity curve對齊版本）、
IS/OOS切分（此版沿用v2的70/30切分，不是reversed版的clean/contaminated切分，
因為這次的問題不是"這批資料被看過沒"，是"探索性 vs 確認性"，兩種不同的警語）
"""

import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timezone, timedelta

BASE_URL  = "https://fapi.bitunix.com"

SYMBOLS   = ["ETHUSDT", "BTCUSDT", "SOLUSDT"]
PAGES_5M  = 3000
PAGES_15M = 1000
FEE_RATE  = 0.0006
FUND_RATE = 0.000574

ATR_PERIOD     = 14
ATR_MULTIPLIER = 1.5

VP_LOOKBACK = 200
VP_BINS     = 35
TP1_MIN_PCT = 0.0016
MIN_TP1_RR  = 1.8

TP1_RATIO = 0.60
TP2_RATIO = 0.25
TP3_RATIO = 0.15

FUNDING_HOURS_UTC = [0, 8, 16]

TARGET_RISK_USDT  = 0.12
MAX_NOTIONAL_USDT = 500.0
MIN_NOTIONAL_USDT = 10.0

FEE_PCT_OF_RISK_CEILING = 0.30
MIN_RISK_PCT_OF_PRICE   = 2 * FEE_RATE / FEE_PCT_OF_RISK_CEILING

SLIPPAGE_BPS = 2.0

# [v3新增] 水上第2次金叉 / 水下第2次死叉的加碼倍數
DOUBLE_CROSS_TRIGGER   = 2   # 第幾次交叉觸發加碼（"第二次"）
DOUBLE_CROSS_MULTIPLIER = 2.0   # 加碼倍數

# [新增] 權益曲線初始資金 —— 純粹是報酬率換算的分母，不影響訊號/PnL金額本身
INITIAL_CAPITAL = 10000.0

# [新增] 樣本外切分比例：前IS_RATIO用來看「策略本身有沒有東西」，
# 後面(1-IS_RATIO)完全不看到，等IS分析都定案後才驗證一次，不可回頭調參
IS_RATIO = 0.70

# [新增] Bootstrap重抽樣次數與區塊長度（stationary bootstrap，保留自相關）
BOOTSTRAP_N     = 2000
BOOTSTRAP_BLOCK = 5  # 平均區塊長度（天），日報酬率短期常有動能延續，區塊抽樣避免低估變異

# ----------------------------------------------
# 數據抓取（未變更）
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
# 指標（未變更 —— 策略本體）
# ----------------------------------------------

def add_indicators(df):
    c = df["close"]; h = df["high"]; l = df["low"]
    df["ema20"]  = c.ewm(span=20,  adjust=False).mean()
    df["ema60"]  = c.ewm(span=60,  adjust=False).mean()
    df["ema120"] = c.ewm(span=120, adjust=False).mean()
    # [v3改動] MACD週期 12/26/9 → 7/14/6
    ema_fast   = c.ewm(span=7,  adjust=False).mean()
    ema_slow   = c.ewm(span=14, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal     = macd_line.ewm(span=6, adjust=False).mean()
    df["macd_line"] = macd_line   # [v3新增] 原版只存histogram，這裡額外保留快線本身，判斷0軸穿越要用
    df["hist"] = macd_line - signal
    tr = pd.concat([
        h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr"]  = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    df["risk"] = df["atr"] * ATR_MULTIPLIER
    df = compute_cross_sequence(df)   # [v3新增] 計算水上/水下第N次交叉
    return df


def compute_cross_sequence(df):
    """
    [v3新增] 追蹤MACD快線相對0軸的"水上/水下"狀態，並在同一段期間內對
    金叉(histogram負轉正)、死叉(histogram正轉負)分別計數。
    每次macd_line穿越0軸，對應方向的計數器歸零重新算。
    golden_cross_seq[i] = 若bar i是一次金叉，這是本次水上期間的第幾次金叉；
                           不是金叉發生的bar則為0
    death_cross_seq[i]  同理，水下期間內第幾次死叉
    """
    macd_line = df["macd_line"].values
    hist      = df["hist"].values
    n = len(df)
    golden_seq = np.zeros(n, dtype=int)
    death_seq  = np.zeros(n, dtype=int)
    gc_count = 0
    dc_count = 0
    zone = 0  # +1 = 水上(macd_line>0)，-1 = 水下，0 = 尚未確定(起始幾根)

    for i in range(1, n):
        if macd_line[i] > 0 and macd_line[i - 1] <= 0:
            zone = 1
            gc_count = 0   # 進入新的水上期間，金叉計數重置
        elif macd_line[i] < 0 and macd_line[i - 1] >= 0:
            zone = -1
            dc_count = 0   # 進入新的水下期間，死叉計數重置

        golden = hist[i] > 0 and hist[i - 1] <= 0
        death  = hist[i] < 0 and hist[i - 1] >= 0

        if golden and zone == 1:
            gc_count += 1
            golden_seq[i] = gc_count
        if death and zone == -1:
            dc_count += 1
            death_seq[i] = dc_count

    df = df.copy()
    df["golden_cross_seq"] = golden_seq
    df["death_cross_seq"]  = death_seq
    return df

# ----------------------------------------------
# Volume Profile（未變更 —— 策略本體）
# ----------------------------------------------

def find_vp_peaks(df15, ref_time_ms, entry_price, side, n=3, risk_dist=0.0, min_tp1_rr=MIN_TP1_RR):
    window = df15[df15["time"] < ref_time_ms].tail(VP_LOOKBACK)
    if len(window) < 10:
        return [None] * n
    min_p = window["low"].min(); max_p = window["high"].max()
    if max_p <= min_p:
        return [None] * n
    step = (max_p - min_p) / VP_BINS
    bins = np.zeros(VP_BINS)
    for _, row in window.iterrows():
        lo, hi, vol = row["low"], row["high"], float(row["baseVol"] or 0)
        if vol <= 0:
            continue
        b_lo = max(0, min(VP_BINS - 1, int((lo - min_p) / step)))
        b_hi = max(0, min(VP_BINS - 1, int((hi - min_p) / step)))
        span = b_hi - b_lo + 1
        for b in range(b_lo, b_hi + 1):
            bins[b] += vol / span

    peaks = []
    min_dist = max(entry_price * TP1_MIN_PCT, risk_dist * min_tp1_rr)
    min_tp1 = entry_price + min_dist if side == "BUY" else entry_price - min_dist

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
# 資金費率 / 滑價 / 同根K棒判斷 / 兩平價（未變更）
# ----------------------------------------------

def calc_funding_fee_v2(entry_ms, exit_ms, notional):
    if exit_ms <= entry_ms:
        return 0.0
    entry_dt = datetime.fromtimestamp(entry_ms / 1000, tz=timezone.utc)
    exit_dt  = datetime.fromtimestamp(exit_ms  / 1000, tz=timezone.utc)
    count = 0
    day = entry_dt.date(); last_day = exit_dt.date()
    while day <= last_day:
        for h in FUNDING_HOURS_UTC:
            point = datetime(day.year, day.month, day.day, h, 0, 0, tzinfo=timezone.utc)
            if entry_dt < point <= exit_dt:
                count += 1
        day = day.fromordinal(day.toordinal() + 1)
    return count * FUND_RATE * notional

def apply_slippage(price, side, is_entry):
    s = SLIPPAGE_BPS / 10000.0
    if side == "BUY":
        return price * (1 + s) if is_entry else price * (1 - s)
    else:
        return price * (1 - s) if is_entry else price * (1 + s)

def resolve_same_bar_sl_vs_tp(side, o, high, low, sl_price, tp_price):
    sl_hit_raw = (side == "BUY"  and low  <= sl_price) or \
                 (side == "SELL" and high >= sl_price)
    tp_hit_raw = False
    if tp_price is not None:
        tp_hit_raw = (side == "BUY"  and high >= tp_price) or \
                     (side == "SELL" and low  <= tp_price)
    if sl_hit_raw and tp_hit_raw:
        sl_dist = abs(o - sl_price); tp_dist = abs(o - tp_price)
        return (True, False) if sl_dist <= tp_dist else (False, True)
    return sl_hit_raw, tp_hit_raw

def breakeven_price(entry_price, side, fee_rate=FEE_RATE):
    if side == "BUY":
        return entry_price * (1 + fee_rate)
    else:
        return entry_price * (1 - fee_rate)

# ----------------------------------------------
# 單幣回測 —— 進場/TP訊號邏輯未變更；新增：daily mark-to-market快照
# ----------------------------------------------

def backtest_symbol(symbol, df5, df15, min_tp1_rr=MIN_TP1_RR):
    positions = []
    trades    = []
    daily_records = []   # [新增] (date, unrealized_incl_partial, realized_closed_cum)
    running_realized = 0.0

    df5 = df5.copy()
    df5["returns"]          = df5["close"].pct_change()
    df5["strategy_returns"] = 0.0

    current_date = None

    for i in range(122, len(df5)):
        row   = df5.iloc[i]
        prev  = df5.iloc[i - 1]
        prev2 = df5.iloc[i - 2]
        o     = float(row["open"]); high = float(row["high"]); low = float(row["low"])
        t     = int(row["time"])

        # [新增] 跨日快照：用前一根K棒收盤價mark所有未平倉部位
        row_date = datetime.fromtimestamp(t / 1000, tz=timezone.utc).date()
        if current_date is None:
            current_date = row_date
        elif row_date != current_date:
            mark_price = float(df5.iloc[i - 1]["close"])
            unrealized_incl_partial = 0.0
            for pos in positions:
                diff = (mark_price - pos["entry_price"]) * pos["remaining_qty"]
                if pos["side"] == "SELL":
                    diff = -diff
                unrealized_incl_partial += diff + pos["realized_pnl"]
            daily_records.append({
                "date": current_date,
                "unrealized_incl_partial": unrealized_incl_partial,
                "realized_closed_cum": running_realized,
            })
            current_date = row_date

        # ── 更新所有持倉（訊號/TP邏輯未變更）──
        closed_idx = []
        for pi, pos in enumerate(positions):
            side = pos["side"]
            if side == "BUY":
                pos["worst_price"] = min(pos["worst_price"], low)
                pos["best_price"]  = max(pos["best_price"], high)
            else:
                pos["worst_price"] = max(pos["worst_price"], high)
                pos["best_price"]  = min(pos["best_price"], low)

            tp_price_for_order = None
            if pos["tp_hit"] == 0 and pos["tp1"]:
                tp_price_for_order = pos["tp1"]
            elif pos["tp_hit"] == 1 and pos["tp2"]:
                tp_price_for_order = pos["tp2"]
            elif pos["tp_hit"] == 2 and pos["tp3"]:
                tp_price_for_order = pos["tp3"]

            sl_hit, _ = resolve_same_bar_sl_vs_tp(side, o, high, low, pos["sl"], tp_price_for_order)

            if sl_hit:
                exit_fill = apply_slippage(pos["sl"], side, is_entry=False)
                qty_r    = pos["remaining_qty"]
                notional = pos["entry_price"] * qty_r
                pnl_raw  = (exit_fill - pos["entry_price"]) * qty_r
                if side == "SELL":
                    pnl_raw = -pnl_raw
                fee     = notional * FEE_RATE
                funding = calc_funding_fee_v2(pos["entry_time"], t, notional)
                net     = pnl_raw - fee - funding + pos["realized_pnl"]
                running_realized += net   # [新增]

                df5.at[i, "strategy_returns"] += \
                    (exit_fill - pos["entry_price"]) / pos["entry_price"] * (1 if side == "BUY" else -1)

                trades.append(_make_trade(symbol, pos, t, exit_fill, "SL", net,
                                          fee + pos["fee_paid"], funding + pos["fund_paid"],
                                          exit_bar_index=i))
                closed_idx.append(pi)
                continue

            tp1_hit = (side == "BUY"  and pos["tp_hit"] == 0 and pos["tp1"] and high >= pos["tp1"]) or \
                      (side == "SELL" and pos["tp_hit"] == 0 and pos["tp1"] and low  <= pos["tp1"])
            if tp1_hit:
                qty_c   = pos["initial_qty"] * TP1_RATIO
                notional= pos["entry_price"] * qty_c
                pnl_raw = abs(pos["tp1"] - pos["entry_price"]) * qty_c
                fee     = notional * FEE_RATE
                funding = calc_funding_fee_v2(pos["entry_time"], t, notional)
                net     = pnl_raw - fee - funding
                pos["realized_pnl"]  += net
                pos["fee_paid"]      += fee
                pos["fund_paid"]     += funding
                pos["remaining_qty"] -= qty_c
                pos["tp_hit"]         = 1
                pos["sl"]             = breakeven_price(pos["entry_price"], side)
                pos["hedged"]         = True

                new_sl_hit, _ = resolve_same_bar_sl_vs_tp(side, o, high, low, pos["sl"], pos["tp2"])
                if new_sl_hit:
                    qty_r    = pos["remaining_qty"]
                    notional = pos["entry_price"] * qty_r
                    fee      = notional * FEE_RATE
                    funding  = calc_funding_fee_v2(pos["entry_time"], t, notional)
                    net      = 0.0 - fee - funding + pos["realized_pnl"]
                    running_realized += net   # [新增]
                    trades.append(_make_trade(symbol, pos, t, pos["sl"], "SL(已套保)", net,
                                              fee + pos["fee_paid"], funding + pos["fund_paid"],
                                              exit_bar_index=i))
                    closed_idx.append(pi)
                    continue

            tp2_hit = (side == "BUY"  and pos["tp_hit"] == 1 and pos["tp2"] and high >= pos["tp2"]) or \
                      (side == "SELL" and pos["tp_hit"] == 1 and pos["tp2"] and low  <= pos["tp2"])
            if tp2_hit:
                qty_c   = pos["initial_qty"] * TP2_RATIO
                notional= pos["entry_price"] * qty_c
                pnl_raw = abs(pos["tp2"] - pos["entry_price"]) * qty_c
                fee     = notional * FEE_RATE
                funding = calc_funding_fee_v2(pos["entry_time"], t, notional)
                net     = pnl_raw - fee - funding
                pos["realized_pnl"]  += net
                pos["fee_paid"]      += fee
                pos["fund_paid"]     += funding
                pos["remaining_qty"] -= qty_c
                pos["tp_hit"]         = 2
                pos["sl"]             = breakeven_price(pos["tp1"], side)

                new_sl_hit, _ = resolve_same_bar_sl_vs_tp(side, o, high, low, pos["sl"], pos["tp3"])
                if new_sl_hit:
                    exit_fill = apply_slippage(pos["sl"], side, is_entry=False)
                    qty_r    = pos["remaining_qty"]
                    notional = pos["entry_price"] * qty_r
                    pnl_raw  = abs(exit_fill - pos["entry_price"]) * qty_r
                    fee      = notional * FEE_RATE
                    funding  = calc_funding_fee_v2(pos["entry_time"], t, notional)
                    net      = pnl_raw - fee - funding + pos["realized_pnl"]
                    running_realized += net   # [新增]
                    trades.append(_make_trade(symbol, pos, t, exit_fill, "SL(已套保)", net,
                                              fee + pos["fee_paid"], funding + pos["fund_paid"],
                                              exit_bar_index=i))
                    closed_idx.append(pi)
                    continue

            tp3_hit = (side == "BUY"  and pos["tp_hit"] == 2 and pos["tp3"] and high >= pos["tp3"]) or \
                      (side == "SELL" and pos["tp_hit"] == 2 and pos["tp3"] and low  <= pos["tp3"])
            if tp3_hit:
                qty_c   = pos["remaining_qty"]
                notional= pos["entry_price"] * qty_c
                pnl_raw = abs(pos["tp3"] - pos["entry_price"]) * qty_c
                fee     = notional * FEE_RATE
                funding = calc_funding_fee_v2(pos["entry_time"], t, notional)
                net     = pnl_raw - fee - funding + pos["realized_pnl"]
                running_realized += net   # [新增]

                df5.at[i, "strategy_returns"] += \
                    abs(pos["tp3"] - pos["entry_price"]) / pos["entry_price"]

                trades.append(_make_trade(symbol, pos, t, pos["tp3"], "TP3", net,
                                          fee + pos["fee_paid"], funding + pos["fund_paid"],
                                          exit_bar_index=i))
                closed_idx.append(pi)

        for pi in sorted(closed_idx, reverse=True):
            positions.pop(pi)

        # ── 進場條件（EMA排列判斷未變更；MACD改用7/14/6週期，新增加碼判斷）──
        entry     = float(row["close"])
        risk_dist = float(row["risk"])
        bull_ma    = prev["ema20"] > prev["ema60"] > prev["ema120"]
        bear_ma    = prev["ema20"] < prev["ema60"] < prev["ema120"]
        macd_cross = prev["hist"] > 0 and prev2["hist"] <= 0
        macd_death = prev["hist"] < 0 and prev2["hist"] >= 0

        # [v3新增] 加碼倍數：水上第DOUBLE_CROSS_TRIGGER次金叉 / 水下第DOUBLE_CROSS_TRIGGER次死叉
        golden_seq_at_prev = int(prev["golden_cross_seq"])
        death_seq_at_prev  = int(prev["death_cross_seq"])
        buy_multiplier  = DOUBLE_CROSS_MULTIPLIER if golden_seq_at_prev == DOUBLE_CROSS_TRIGGER else 1.0
        sell_multiplier = DOUBLE_CROSS_MULTIPLIER if death_seq_at_prev  == DOUBLE_CROSS_TRIGGER else 1.0

        for side, cond, size_mult in [("BUY", bull_ma and macd_cross, buy_multiplier),
                                       ("SELL", bear_ma and macd_death, sell_multiplier)]:
            if not cond or risk_dist <= 0:
                continue
            if (risk_dist / entry) < MIN_RISK_PCT_OF_PRICE:
                continue

            peaks       = find_vp_peaks(df15, t, entry, side, n=3, risk_dist=risk_dist, min_tp1_rr=min_tp1_rr)
            tp1, tp2, tp3 = peaks
            if side == "BUY":
                if tp1 is None: tp1 = entry + risk_dist * 1.5
                if tp2 is None: tp2 = entry + risk_dist * 2.5
                if tp3 is None: tp3 = entry + risk_dist * 4.0
            else:
                if tp1 is None: tp1 = entry - risk_dist * 1.5
                if tp2 is None: tp2 = entry - risk_dist * 2.5
                if tp3 is None: tp3 = entry - risk_dist * 4.0

            entry_fill = apply_slippage(entry, side, is_entry=True)
            sl = entry_fill - risk_dist if side == "BUY" else entry_fill + risk_dist

            # [v3改動] qty反推自 TARGET_RISK_USDT × size_mult，讓risk_usdt也等比例放大2倍
            target_risk_this_trade = TARGET_RISK_USDT * size_mult
            qty      = target_risk_this_trade / risk_dist
            notional = qty * entry_fill
            if notional > MAX_NOTIONAL_USDT * size_mult or notional < MIN_NOTIONAL_USDT:
                continue

            open_fee  = notional * FEE_RATE
            risk_usdt = risk_dist * qty

            positions.append({
                "symbol": symbol, "side": side, "entry_price": entry_fill, "sl": sl,
                "tp1": tp1, "tp2": tp2, "tp3": tp3, "tp_hit": 0, "hedged": False,
                "initial_qty": qty, "remaining_qty": qty, "realized_pnl": -open_fee,
                "fee_paid": open_fee, "fund_paid": 0.0, "entry_time": t,
                "risk_usdt": risk_usdt, "worst_price": entry_fill, "best_price": entry_fill,
                "entry_bar_index": i, "size_mult": size_mult,
            })

    # [新增] 收尾快照，把最後一天也記錄進去
    if current_date is not None:
        mark_price = float(df5.iloc[-1]["close"])
        unrealized_incl_partial = 0.0
        for pos in positions:
            diff = (mark_price - pos["entry_price"]) * pos["remaining_qty"]
            if pos["side"] == "SELL":
                diff = -diff
            unrealized_incl_partial += diff + pos["realized_pnl"]
        daily_records.append({
            "date": current_date,
            "unrealized_incl_partial": unrealized_incl_partial,
            "realized_closed_cum": running_realized,
        })

    return trades, df5, daily_records


def _make_trade(symbol, pos, exit_time, exit_price, result, net_pnl, fee_total, fund_total,
                 exit_bar_index=None):
    risk = pos["risk_usdt"] if pos["risk_usdt"] > 0 else 1e-9
    bars_held  = (exit_bar_index - pos["entry_bar_index"]) if exit_bar_index is not None else np.nan
    r_multiple = net_pnl / risk

    side = pos["side"]
    if side == "BUY":
        mae = pos["entry_price"] - pos["worst_price"]; mfe = pos["best_price"] - pos["entry_price"]
    else:
        mae = pos["worst_price"] - pos["entry_price"]; mfe = pos["entry_price"] - pos["best_price"]
    risk_dist = risk / pos["initial_qty"] if pos["initial_qty"] else np.nan
    mae_r = mae / risk_dist if risk_dist else np.nan
    mfe_r = mfe / risk_dist if risk_dist else np.nan
    fee_pct_of_risk = fee_total / risk if risk else np.nan

    return {
        "symbol": symbol, "side": pos["side"],
        # [新增] 原始ms時間戳，供equity curve/benchmark對齊使用
        "entry_ts": int(pos["entry_time"]),
        "exit_ts": int(exit_time),
        "entry_time": datetime.fromtimestamp(pos["entry_time"] / 1000).strftime("%m-%d %H:%M"),
        "exit_time" : datetime.fromtimestamp(exit_time / 1000).strftime("%m-%d %H:%M"),
        "entry": round(pos["entry_price"], 4), "exit": round(exit_price, 4),
        "tp1": round(pos["tp1"], 4) if pos["tp1"] else "-",
        "tp2": round(pos["tp2"], 4) if pos["tp2"] else "-",
        "tp3": round(pos["tp3"], 4) if pos["tp3"] else "-",
        "tp_hit": pos["tp_hit"], "result": result, "net_pnl": round(net_pnl, 4),
        "fee_total": round(fee_total, 4), "fund_total": round(fund_total, 4),
        "risk_usdt": round(risk, 4), "R_Multiple": round(r_multiple, 4),
        "MAE_R": round(mae_r, 4) if not np.isnan(mae_r) else np.nan,
        "MFE_R": round(mfe_r, 4) if not np.isnan(mfe_r) else np.nan,
        "fee_pct_of_risk": round(fee_pct_of_risk, 4) if not np.isnan(fee_pct_of_risk) else np.nan,
        "hedged": "Y" if pos["hedged"] else "N", "bars_held": bars_held,
        "size_mult": pos.get("size_mult", 1.0),   # [v3新增] 這筆是否為第2次交叉加碼倉位
    }

# ----------------------------------------------
# [新增] 組合層權益曲線：合併多幣種的daily_records成單一portfolio equity curve
# ----------------------------------------------

def build_portfolio_equity(daily_records_by_symbol, initial_capital=INITIAL_CAPITAL):
    """
    daily_records_by_symbol: {symbol: [ {date, unrealized_incl_partial, realized_closed_cum}, ... ]}
    每個symbol的realized_closed_cum是「該symbol自己累積到當天」的已實現淨盈虧，
    unrealized_incl_partial是「當天收盤時，該symbol所有未平倉部位」的浮動盈虧
    （已含該部位自己的部分止盈已實現金額）。
    Portfolio當天權益 = initial_capital + Σ_symbol(realized_closed_cum + unrealized_incl_partial)
    """
    all_dates = sorted(set(
        rec["date"] for recs in daily_records_by_symbol.values() for rec in recs
    ))
    if not all_dates:
        return pd.Series(dtype=float)

    # 每個symbol先轉成 date -> (realized_cum, unrealized) 的查表，並做前向填補
    per_symbol_series = {}
    for sym, recs in daily_records_by_symbol.items():
        recs_sorted = sorted(recs, key=lambda r: r["date"])
        s_realized = pd.Series(
            {r["date"]: r["realized_closed_cum"] for r in recs_sorted}
        ).reindex(all_dates).ffill().fillna(0.0)
        s_unreal = pd.Series(
            {r["date"]: r["unrealized_incl_partial"] for r in recs_sorted}
        ).reindex(all_dates)
        # 未平倉浮動盈虧不能前向填補到「已經沒部位」的日子，沒記錄的日子視為0
        s_unreal = s_unreal.fillna(0.0)
        per_symbol_series[sym] = s_realized + s_unreal

    combined = sum(per_symbol_series.values())
    equity = initial_capital + combined
    equity.index = pd.to_datetime(equity.index)
    equity = equity.sort_index()
    return equity

# ----------------------------------------------
# [新增] 風險調整指標：Sharpe / Sortino / Calmar / MaxDD / Profit Factor / Expectancy
# ----------------------------------------------

def compute_perf_stats(equity, trades_df=None, periods_per_year=365, initial_capital=INITIAL_CAPITAL):
    """
    [v3 bug修正] 原本用 equity.pct_change() 算報酬率，這假設資金是複利成長
    （今天報酬率基於昨天資金水位）。但TARGET_RISK_USDT是固定金額倉位，不隨
    資金水位縮放——這是arithmetic sizing，不是geometric。用複利公式套在
    arithmetic模型上，當累積虧損接近/超過initial_capital這個純粹當分母用的
    假設值時，pct_change會在低base上產生數學上失真的巨大波動（v3
    MACD7/14/6版曾算出年化波動率234%~429%，不是槓桿真的失控，是這裡的
    計算方式跟倉位模型不匹配）。

    改用 diff()/initial_capital：固定資金基準上的每日損益比例（arithmetic
    return），不管equity怎麼變動都不會有除以趨近0的分母問題，且跟固定
    風險金額倉位的模型假設一致。annualization也跟著從複利公式
    (1+r)^365-1 改成算術年化 r_mean×365，同一套邏輯貫穿到底。
    """
    stats = {}
    ret = equity.diff().dropna() / initial_capital
    if len(ret) < 2 or ret.std() == 0:
        stats.update({"Sharpe": np.nan, "Sortino": np.nan, "Calmar": np.nan,
                      "MaxDD_pct": np.nan, "AnnReturn_pct": np.nan})
        return stats, ret

    ann_return = ret.mean() * periods_per_year      # [v3修正] 算術年化，不用複利公式
    ann_vol    = ret.std() * np.sqrt(periods_per_year)
    sharpe     = ann_return / ann_vol if ann_vol > 0 else np.nan

    downside = ret[ret < 0]
    down_vol = downside.std() * np.sqrt(periods_per_year) if len(downside) > 1 else np.nan
    sortino  = ann_return / down_vol if down_vol and down_vol > 0 else np.nan

    running_max = equity.cummax()
    drawdown    = (equity - running_max) / running_max
    max_dd      = drawdown.min()
    calmar      = ann_return / abs(max_dd) if max_dd < 0 else np.nan

    stats.update({
        "Sharpe": sharpe, "Sortino": sortino, "Calmar": calmar,
        "MaxDD_pct": max_dd * 100, "AnnReturn_pct": ann_return * 100,
        "AnnVol_pct": ann_vol * 100,
    })

    if trades_df is not None and len(trades_df):
        wins   = trades_df[trades_df["net_pnl"] > 0]["net_pnl"]
        losses = trades_df[trades_df["net_pnl"] <= 0]["net_pnl"]
        gross_win  = wins.sum()
        gross_loss = abs(losses.sum())
        profit_factor = gross_win / gross_loss if gross_loss > 0 else np.nan
        expectancy = trades_df["net_pnl"].mean()
        stats.update({"ProfitFactor": profit_factor, "Expectancy_USDT": expectancy})

    # [v3新增] 資金基準是否足夠的安全檢查：如果equity曾經<=0，代表initial_capital
    # 相對於這個策略的累積虧損規模太小，MaxDD百分比會失真（>100%在邏輯上代表
    # 帳戶已經歸零倒欠），這裡明確示警而不是讓失真數字悄悄流出去
    if (equity <= 0).any():
        stats["_capital_inadequate_warning"] = True

    return stats, ret

# ----------------------------------------------
# [新增] Stationary Bootstrap：對日報酬率序列做區塊重抽樣，估計Sharpe的95% CI
# ----------------------------------------------

def stationary_bootstrap_sharpe(ret, n_boot=BOOTSTRAP_N, avg_block=BOOTSTRAP_BLOCK,
                                 periods_per_year=365, seed=42):
    """
    區塊長度服從幾何分布（Politis & Romano, 1994），避免像naive iid bootstrap
    一樣打散短期自相關結構（策略連續幾天贏/連續幾天輸的模式會被保留）。
    """
    rng = np.random.default_rng(seed)
    r = ret.values
    n = len(r)
    if n < 10:
        return np.nan, np.nan, np.array([])

    p = 1.0 / avg_block
    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        pos = rng.integers(0, n)
        for k in range(n):
            if k == 0 or rng.random() < p:
                pos = rng.integers(0, n)
            else:
                pos = (pos + 1) % n
            idx[k] = pos
        sample = r[idx]
        mu, sd = sample.mean(), sample.std()
        if sd == 0:
            sharpes[b] = np.nan
        else:
            # [v3修正] ret現在是arithmetic return(diff/initial_capital)，不是
            # pct_change，年化要用算術年化 mu×periods，不能再用複利公式(1+mu)^n-1
            ann_ret = mu * periods_per_year
            ann_vol = sd * np.sqrt(periods_per_year)
            sharpes[b] = ann_ret / ann_vol
    valid = sharpes[~np.isnan(sharpes)]
    if len(valid) == 0:
        return np.nan, np.nan, valid
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return lo, hi, valid

# ----------------------------------------------
# [新增] 正確對齊的 Beta / Alpha：equity curve日報酬 vs 三幣等權basket日報酬
# ----------------------------------------------

def compute_beta_alpha(strat_ret, symbol_close_caches, periods_per_year=365):
    """
    symbol_close_caches: {symbol: df5[["time","close"]]}，用來建立每日收盤價序列
    basket日報酬 = 三幣等權組合的日報酬率（不是策略部位的實際曝險，是「市場整體」的參照）
    """
    daily_closes = {}
    for sym, df5 in symbol_close_caches.items():
        s = df5.copy()
        s["date"] = pd.to_datetime(s["time"], unit="ms").dt.date
        daily_close = s.groupby("date")["close"].last()
        daily_closes[sym] = daily_close

    df_close = pd.DataFrame(daily_closes).sort_index().ffill()
    basket_ret = df_close.pct_change().mean(axis=1).dropna()
    basket_ret.index = pd.to_datetime(basket_ret.index)

    aligned = pd.concat([strat_ret.rename("strat"), basket_ret.rename("bench")], axis=1, sort=True).dropna()
    if len(aligned) < 10 or aligned["bench"].var() == 0:
        return np.nan, np.nan, len(aligned)

    beta  = aligned["strat"].cov(aligned["bench"]) / aligned["bench"].var()
    alpha_daily = aligned["strat"].mean() - beta * aligned["bench"].mean()
    alpha_annualized = alpha_daily * periods_per_year
    return beta, alpha_annualized, len(aligned)

# ----------------------------------------------
# 輸出
# ----------------------------------------------

def _print_stats_block(label, equity, trades_df, symbol_close_caches):
    stats, ret = compute_perf_stats(equity, trades_df)
    print(f"\n【{label}】區間: {equity.index.min().date()} ~ {equity.index.max().date()}"
          f"（{len(equity)} 天）")
    if stats.get("_capital_inadequate_warning"):
        print(f"  [!] 權益曲線曾經<=0（INITIAL_CAPITAL={INITIAL_CAPITAL}相對這段累積虧損"
              f"規模太小），最大回撤%數字會失真，僅供參考，建議調高INITIAL_CAPITAL重跑")
    print(f"  年化報酬率     : {stats.get('AnnReturn_pct', float('nan')):+.2f}%")
    print(f"  年化波動率     : {stats.get('AnnVol_pct', float('nan')):.2f}%")
    print(f"  Sharpe         : {stats.get('Sharpe', float('nan')):.3f}")
    print(f"  Sortino        : {stats.get('Sortino', float('nan')):.3f}")
    print(f"  Calmar         : {stats.get('Calmar', float('nan')):.3f}")
    print(f"  最大回撤       : {stats.get('MaxDD_pct', float('nan')):.2f}%")
    if "ProfitFactor" in stats:
        print(f"  Profit Factor  : {stats['ProfitFactor']:.3f}")
        print(f"  Expectancy/筆  : {stats['Expectancy_USDT']:+.4f} USDT")

    lo, hi, _ = stationary_bootstrap_sharpe(ret)
    print(f"  Sharpe 95% CI  : [{lo:.3f}, {hi:.3f}]  "
          f"(stationary bootstrap, n={BOOTSTRAP_N}, avg_block={BOOTSTRAP_BLOCK}天)")
    if not np.isnan(lo) and lo < 0 < hi:
        print(f"  → CI涵蓋0，樣本數不足以證明Sharpe顯著大於0，結論保留")

    beta, alpha, n_days = compute_beta_alpha(ret, symbol_close_caches)
    print(f"  Beta (vs三幣等權basket) : {beta:.3f}  （對齊天數={n_days}）")
    print(f"  Alpha (年化)            : {alpha:+.4f}")


def print_results(all_trades, dfs, daily_records_by_symbol, run_label="", min_tp1_rr=MIN_TP1_RR):
    if not all_trades:
        print("無交易信號")
        return

    df_all = pd.DataFrame(all_trades)
    sym_counts = df_all.groupby("symbol").size().to_dict()
    total = len(df_all)
    wins   = df_all[df_all["net_pnl"] > 0]
    losses = df_all[df_all["net_pnl"] <= 0]
    tp1_hit = (df_all["tp_hit"] >= 1).sum()
    tp2_hit = (df_all["tp_hit"] >= 2).sum()
    tp3_hit = (df_all["result"] == "TP3").sum()
    total_net  = df_all["net_pnl"].sum()
    total_fee  = df_all["fee_total"].sum()
    total_fund = df_all["fund_total"].sum()
    avg_win  = wins["net_pnl"].mean() if len(wins) else 0
    avg_loss = losses["net_pnl"].mean() if len(losses) else 0
    avg_r    = df_all["R_Multiple"].mean()

    long_trades  = df_all[df_all["side"] == "BUY"]
    short_trades = df_all[df_all["side"] == "SELL"]

    print(f"\n{'='*68}")
    print(f"{' / '.join(SYMBOLS)}  5m進場(MACD 7/14/6) / 15m VP TP  多空雙向  "
          f"[v15sd_merged_v3_macd7146{run_label}]  水上/水下第{DOUBLE_CROSS_TRIGGER}次交叉×"
          f"{DOUBLE_CROSS_MULTIPLIER}倉位  MIN_TP1_RR={min_tp1_rr}")
    print(f"風控：固定風險 {TARGET_RISK_USDT} USDT/筆  SL：ATR({ATR_PERIOD})×{ATR_MULTIPLIER}")
    print(f"手續費：{FEE_RATE*100}%  資金費率：{FUND_RATE*100}%/8h  滑價：{SLIPPAGE_BPS}bps")
    print(f"{'='*68}")
    for sym in SYMBOLS:
        n = sym_counts.get(sym, 0)
        sym_net = df_all[df_all["symbol"] == sym]["net_pnl"].sum()
        sym_l = (df_all[df_all["symbol"] == sym]["side"] == "BUY").sum()
        sym_s = (df_all[df_all["symbol"] == sym]["side"] == "SELL").sum()
        print(f"  {sym:<10} 交易 {n:>4} 次（多 {sym_l} / 空 {sym_s}）  淨盈虧 {sym_net:>+.4f} USDT")
    print(f"{'─'*68}")
    print(f"總交易次數  : {total}（多 {len(long_trades)} / 空 {len(short_trades)}）")
    print(f"勝率        : {len(wins)/total*100:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"TP層次分佈  : TP1={tp1_hit}  TP2={tp2_hit}  TP3={tp3_hit}  SL={len(losses)}")
    print(f"總淨盈虧    : {total_net:+.4f} USDT")
    print(f"總手續費    : -{total_fee:.4f} USDT")
    print(f"總資金費率  : -{total_fund:.4f} USDT")
    print(f"平均盈利    : {avg_win:+.4f} USDT")
    print(f"平均虧損    : {avg_loss:+.4f} USDT")
    print(f"平均R倍數   : {avg_r:+.4f}")
    if avg_loss != 0:
        print(f"盈虧比      : {abs(avg_win/avg_loss):.2f}")

    # ── [新增] 全樣本權益曲線 + 風險調整指標 ──
    equity_full = build_portfolio_equity(daily_records_by_symbol)
    if len(equity_full) < 5:
        print("\n[!] 天數過少，無法計算有意義的權益曲線指標")
        return

    symbol_close_caches = {sym: dfs[sym][["time", "close"]] for sym in SYMBOLS}

    print(f"\n{'='*68}")
    print("風險調整指標（基於逐日Mark-to-Market權益曲線，非trade-level統計量）")
    print(f"{'='*68}")
    _print_stats_block("全樣本 (Full Sample)", equity_full, df_all, symbol_close_caches)

    # ── [新增] 樣本外驗證：依時間切前IS_RATIO / 後(1-IS_RATIO) ──
    split_idx = int(len(equity_full) * IS_RATIO)
    split_date = equity_full.index[split_idx]
    equity_is  = equity_full.iloc[:split_idx + 1]
    equity_oos = equity_full.iloc[split_idx:]

    trades_is  = df_all[pd.to_datetime(df_all["exit_ts"], unit="ms") < split_date]
    trades_oos = df_all[pd.to_datetime(df_all["exit_ts"], unit="ms") >= split_date]

    print(f"\n{'='*68}")
    print(f"樣本外驗證（切分點: {split_date.date()}，IS={IS_RATIO*100:.0f}% / OOS={(1-IS_RATIO)*100:.0f}%）")
    print(f"{'='*68}")
    _print_stats_block("樣本內 In-Sample", equity_is, trades_is, symbol_close_caches)
    _print_stats_block("樣本外 Out-of-Sample", equity_oos, trades_oos, symbol_close_caches)
    print(f"\n[判讀] 若OOS的Sharpe/勝率遠低於IS（尤其翻負），代表IS階段可能是雜訊被"
          f"參數/邏輯選擇fit上去的假象，不能當作策略有真實edge的證據。")

    # [v3新增] 1倍 vs 2倍(加碼)倉位的績效對照，直接看加碼規則本身有沒有加分
    print(f"\n{'='*68}")
    print("加碼規則診斷：1倍倉位 vs 第2次交叉加碼(2倍)倉位")
    print(f"{'='*68}")
    for mult_label, sub in [("1倍(一般)", df_all[df_all["size_mult"] == 1.0]),
                             ("2倍(第2次交叉加碼)", df_all[df_all["size_mult"] == 2.0])]:
        if len(sub) == 0:
            continue
        n = len(sub)
        wr = (sub["net_pnl"] > 0).sum() / n
        avg_r = sub["R_Multiple"].mean()
        pnl = sub["net_pnl"].sum()
        print(f"  {mult_label:<20} n={n:>6}  勝率={wr*100:5.1f}%  平均R={avg_r:+.4f}  總淨盈虧={pnl:+9.2f} USDT")
    if len(df_all[df_all["size_mult"] == 2.0]) > 30 and len(df_all[df_all["size_mult"] == 1.0]) > 30:
        from scipy import stats as _stats
        t_stat, p_val = _stats.ttest_ind(
            df_all[df_all["size_mult"] == 2.0]["R_Multiple"],
            df_all[df_all["size_mult"] == 1.0]["R_Multiple"], equal_var=False)
        print(f"  Welch's t-test（2倍組 vs 1倍組 R_Multiple均值）: t={t_stat:.3f}, p={p_val:.4g}")
        if p_val < 0.05:
            better = "2倍加碼組" if df_all[df_all["size_mult"]==2.0]["R_Multiple"].mean() > \
                                      df_all[df_all["size_mult"]==1.0]["R_Multiple"].mean() else "1倍一般組"
            print(f"  → 顯著差異，「{better}」的R_Multiple顯著較高")
        else:
            print(f"  → 無顯著差異，加碼規則本身沒有額外的方向性資訊，"
                  f"只是把同樣的(負)期望值放大部位在賭，會放大虧損而非改善它")

    equity_csv = f"equity_curve_v15sg_merged_v3_macd7146{run_label}.csv"
    out = f"backtest_result_v15sg_merged_v3_macd7146{run_label}.csv"
    equity_full.to_csv(equity_csv, header=["equity"])
    try:
        df_all.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n交易明細已存至 {out}")
        print(f"權益曲線已存至 {equity_csv}")
    except PermissionError:
        print(f"\n[!] {out} 被佔用，請關閉 Excel")

# ----------------------------------------------
# 主程序
# ----------------------------------------------

# [v3新增] 要對照的MIN_TP1_RR候選值。1.8是原本設定，1.6是這次想測試的
# "放寬TP1門檻、讓更多交易撐到TP1"假說。資料只抓/算指標一次，backtest_symbol
# (唯一吃到min_tp1_rr的部分)針對每個候選值各跑一次，避免重複抓取浪費時間。
MIN_TP1_RR_CANDIDATES = [1.8, 1.6]

if __name__ == "__main__":
    dfs = {}
    df15s = {}

    for sym in SYMBOLS:
        print(f"\n=== {sym}（抓取資料+算指標，只做一次）===")
        df5  = fetch_all_klines(sym, "5m",  PAGES_5M)
        df5  = add_indicators(df5)
        df15 = fetch_all_klines(sym, "15m", PAGES_15M)
        dfs[sym] = df5
        df15s[sym] = df15

    for min_tp1_rr in MIN_TP1_RR_CANDIDATES:
        run_label = f"_tp1rr{str(min_tp1_rr).replace('.', 'p')}"
        print(f"\n\n{'#'*68}")
        print(f"# 回測 MIN_TP1_RR = {min_tp1_rr}")
        print(f"{'#'*68}")

        all_trades = []
        daily_records_by_symbol = {}
        dfs_out = {}

        for sym in SYMBOLS:
            print(f"  {sym} 回測中...")
            trades, df5_out, daily_records = backtest_symbol(
                sym, dfs[sym], df15s[sym], min_tp1_rr=min_tp1_rr)
            all_trades.extend(trades)
            dfs_out[sym] = df5_out
            daily_records_by_symbol[sym] = daily_records
            print(f"  {sym} 完成，{len(trades)} 筆交易")

        print_results(all_trades, dfs_out, daily_records_by_symbol,
                      run_label=run_label, min_tp1_rr=min_tp1_rr)