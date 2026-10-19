"""
bitunix_backtest_v20sb.py
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
資金費率    : 固定 0.0574%/8h，按「真實UTC結算時間點」(00:00/08:00/16:00) 計算
              跨越次數，而非用持倉時長整除
R倍數       : RISK = |Entry-Stop|×Size，R_Multiple = Result÷Risk
套保        : TP1觸發後 Hedged=Y，視為已鎖定不虧本金
統計        : 累積PnL / 勝率 / 平均R(不含套保) / 平均盈利單 / 平均虧損單 /
              平均盈虧比 / 夏普比率 / 最大回撤 / 套保策略數 / 套保後平均R

────────────────────────────────────────────────────────────────
v21 修正紀錄（相對 v20sa）
────────────────────────────────────────────────────────────────
[修正1] 輸出CSV缺少「滾倉」欄位卻寫在cols清單裡，會在最後輸出時 KeyError 崩潰。
        → _make_trade() 補上 "滾倉" 欄位（目前策略無滾倉機制，固定填 "N"）。

[修正2] 同一根5m K棒內，SL 和 TP1 有可能同時被觸及（K棒波動夠大時），
        原代碼永遠「先判斷SL、SL觸發就直接continue」，等於無條件假設SL先發生，
        沒有任何依據，且會讓策略看起來比實際更差（漏掉了本來能吃到TP1套保的單子）。
        → 改為用開盤價(open)判斷本根K棒「大機率的路徑方向」：
          若 open 離 SL 更近 → 假設SL先觸發；若 open 離 TP 更近 → 假設TP先觸發。
          這仍是一個近似假設（OHLC無法還原K棒內部真實路徑），但比「無條件SL優先」
          更合理，且在程式碼與注釋中明確記錄此為近似處理，不假裝是精確模擬。

[修正3] TP1觸發後，SL 改為 entry_price（套保），但原代碼在同一次迴圈迭代中
        不會重新檢查「新SL」是否也被本根K棒觸及 —— 只有下一根K棒才會用新SL判斷。
        如果一根大陽/陰線同時衝過TP1又打回entry，這根K棒的模擬會不準確
        （實際應該套保出場，但代碼要等到下一根才處理）。
        → 改為在 TP1/TP2/TP3 判斷後，於同一根K棒內用「新SL」重新檢查一次是否觸發，
          若觸發則立即以新SL出場，不留到下一根。

[修正4] 資金費率計算原本用 int((exit-entry)/8h) 整除持倉時長，
        任何持倉不滿8小時的交易資費永遠算0，嚴重低估短線策略的資費成本
        （實盤只要跨過UTC 00:00/08:00/16:00其中一個結算點就要付一次）。
        → 改為 calc_funding_fee_v2()：直接計算 entry_time 到 exit_time 之間
          實際跨越了幾個UTC 00:00/08:00/16:00 結算時間點。
────────────────────────────────────────────────────────────────
"""

import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timezone

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
# 數據抓取（與 v20sa 相同，未變更）
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
    """建 VP，lows/highs/vols 為 numpy array"""
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
# [修正4] 資金費率：改為按「真實UTC結算時間點」計算跨越次數
# 結算時間點固定為每天 UTC 00:00 / 08:00 / 16:00
# ----------------------------------------------

FUNDING_HOURS_UTC = [0, 8, 16]

def _funding_settlements_before_or_at(ts_ms):
    """
    回傳「小於等於 ts_ms 的最近一個資金費結算時間點」的毫秒數。
    用於計算 entry~exit 之間總共跨越了幾個結算點。
    """
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    # 該日 00/08/16 三個結算點，找出 <= dt 的最後一個
    candidates = [
        dt.replace(hour=h, minute=0, second=0, microsecond=0)
        for h in FUNDING_HOURS_UTC
    ]
    valid = [c for c in candidates if c <= dt]
    if valid:
        last = max(valid)
    else:
        # dt 比當天 00:00 還早，不會發生（00:00 是當天最早的結算點），保留防呆
        last = (dt.replace(hour=0, minute=0, second=0, microsecond=0))
    return last

def calc_funding_fee_v2(entry_ms, exit_ms, notional):
    """
    直接計算 entry_time 到 exit_time 之間實際跨越了幾個 UTC 00:00/08:00/16:00
    結算時間點（而非用持倉時長整除8小時），修正短線交易資費被低估的問題。
    """
    if exit_ms <= entry_ms:
        return 0.0

    entry_dt = datetime.fromtimestamp(entry_ms / 1000, tz=timezone.utc)
    exit_dt  = datetime.fromtimestamp(exit_ms  / 1000, tz=timezone.utc)

    # 用小時網格窮舉所有可能結算點，數落在 (entry_dt, exit_dt] 區間內的個數
    # 效能考量：直接用日期範圍生成候選點，而非逐小時迴圈
    count = 0
    day = entry_dt.date()
    last_day = exit_dt.date()
    while day <= last_day:
        for h in FUNDING_HOURS_UTC:
            point = datetime(day.year, day.month, day.day, h, 0, 0, tzinfo=timezone.utc)
            if entry_dt < point <= exit_dt:
                count += 1
        day = day.fromordinal(day.toordinal() + 1)

    return count * FUND_RATE * notional

# 保留舊函式名稱以防其他地方引用，但內部改呼叫新邏輯
def calc_funding_fee(entry_ms, exit_ms, notional):
    return calc_funding_fee_v2(entry_ms, exit_ms, notional)

# ----------------------------------------------
# [修正3再修正] 判斷同一根K棒內，「新SL」與「下一階段TP」誰先發生
# 用途：TP1觸發後SL改entry、TP2觸發後SL改tp1，這兩種情況都要重新判斷
# 本根K棒是否同時也碰到了下一個TP（例如TP1觸發後同棒也碰到TP2甚至TP3）。
# 若無條件讓新SL優先判定，等於重新引入了跟修正2一樣的悲觀偏誤。
# → 統一用open價位置判斷哪個「較可能先發生」，跟修正2用同一套近似邏輯，
#   避免同一類問題在不同分支被不一致地處理。
# ----------------------------------------------

def resolve_same_bar_sl_vs_tp(side, o, high, low, sl_price, tp_price):
    """
    回傳 (sl_hit, tp_hit)：本根K棒最終判定為哪一個先發生。
    - 若只有一個被觸及，直接回傳該結果。
    - 若兩個同時被觸及，用open價離哪個更近來近似判斷先後。
    - 若都沒被觸及，兩者皆False。
    這仍是近似（OHLC無法還原K棒內部真實路徑），非精確模擬。
    """
    sl_hit_raw = (side == "BUY"  and low  <= sl_price) or \
                 (side == "SELL" and high >= sl_price)
    tp_hit_raw = False
    if tp_price is not None:
        tp_hit_raw = (side == "BUY"  and high >= tp_price) or \
                     (side == "SELL" and low  <= tp_price)

    if sl_hit_raw and tp_hit_raw:
        sl_dist = abs(o - sl_price)
        tp_dist = abs(o - tp_price)
        if sl_dist <= tp_dist:
            return True, False
        else:
            return False, True

    return sl_hit_raw, tp_hit_raw


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
        o    = float(row["open"])
        high = float(row["high"])
        low  = float(row["low"])
        t    = int(row["time"])
        close = float(row["close"])

        # ── 更新所有持倉 ──
        closed_idx = []
        for pi, pos in enumerate(positions):
            side = pos["side"]

            # [修正2] 用開盤價判斷本根K棒「較可能先發生」的事件方向，
            # 而非無條件假設SL優先。這仍是近似（OHLC無法還原K棒內部真實路徑），
            # 但比「永遠SL先」更合理，且不隱瞞這是一個假設。
            tp_price_for_order = None
            if pos["tp_hit"] == 0 and pos["tp1"]:
                tp_price_for_order = pos["tp1"]
            elif pos["tp_hit"] == 1 and pos["tp2"]:
                tp_price_for_order = pos["tp2"]
            elif pos["tp_hit"] == 2 and pos["tp3"]:
                tp_price_for_order = pos["tp3"]

            sl_hit, _tp_hit_ignored = resolve_same_bar_sl_vs_tp(
                side, o, high, low, pos["sl"], tp_price_for_order
            )

            if sl_hit:
                qty_r    = pos["remaining_qty"]
                notional = pos["entry_price"] * qty_r
                pnl_raw  = (pos["sl"] - pos["entry_price"]) * qty_r
                if side == "SELL":
                    pnl_raw = -pnl_raw
                fee     = notional * FEE_RATE
                funding = calc_funding_fee_v2(pos["entry_time"], t, notional)
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
                funding = calc_funding_fee_v2(pos["entry_time"], t, notional)
                net     = pnl_raw - fee - funding
                pos["realized_pnl"]  += net
                pos["fee_paid"]      += fee
                pos["fund_paid"]     += funding
                pos["remaining_qty"] -= qty_c
                pos["tp_hit"]         = 1
                pos["sl"]             = pos["entry_price"]
                pos["hedged"]         = True

                # [修正3] TP1觸發後SL改為entry，需在同一根K棒內重新檢查新SL
                # 是否也被本根K棒觸及（例如大陽線衝過TP1後又打回entry）。
                # [修正3再修正] 同時要考慮本根K棒是否也碰到了TP2——
                # 不能無條件讓新SL優先，否則等於重新引入修正2要解決的悲觀偏誤。
                # 用open價位置判斷新SL跟TP2誰先發生。
                new_sl_hit, _tp2_also_hit = resolve_same_bar_sl_vs_tp(
                    side, o, high, low, pos["sl"], pos["tp2"]
                )
                if new_sl_hit:
                    qty_r    = pos["remaining_qty"]
                    notional = pos["entry_price"] * qty_r
                    pnl_raw  = 0.0  # SL == entry_price，此段無盈虧
                    fee      = notional * FEE_RATE
                    funding  = calc_funding_fee_v2(pos["entry_time"], t, notional)
                    net      = pnl_raw - fee - funding + pos["realized_pnl"]

                    trades.append(_make_trade(symbol, pos, t, pos["sl"], "撞到止損(已套保)",
                                              net, fee + pos["fee_paid"], funding + pos["fund_paid"]))
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
                pos["sl"]             = pos["tp1"]

                # [修正3] TP2觸發後SL改為tp1，同樣需要同根K棒內重新檢查
                # [修正3再修正] 同時要考慮本根K棒是否也碰到了TP3，
                # 用open價位置判斷新SL跟TP3誰先發生，而非無條件讓新SL優先。
                new_sl_hit, _tp3_also_hit = resolve_same_bar_sl_vs_tp(
                    side, o, high, low, pos["sl"], pos["tp3"]
                )
                if new_sl_hit:
                    qty_r    = pos["remaining_qty"]
                    notional = pos["entry_price"] * qty_r
                    pnl_raw  = abs(pos["sl"] - pos["entry_price"]) * qty_r  # SL==tp1，仍有盈利
                    fee      = notional * FEE_RATE
                    funding  = calc_funding_fee_v2(pos["entry_time"], t, notional)
                    net      = pnl_raw - fee - funding + pos["realized_pnl"]

                    trades.append(_make_trade(symbol, pos, t, pos["sl"], "撞到止損(已套保)",
                                              net, fee + pos["fee_paid"], funding + pos["fund_paid"]))
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
# 統計計算（與 v20sa 相同，未在本輪修正範圍內）
# ----------------------------------------------

def compute_stats(df_all):
    df_all = df_all.sort_values("Time").reset_index(drop=True)
    df_all["cum_pnl"] = df_all["Result"].cumsum()

    total  = len(df_all)
    wins   = df_all[df_all["Result"] > 0]
    losses = df_all[df_all["Result"] <= 0]
    win_rate = len(wins) / total * 100 if total else 0

    not_hedged = df_all[df_all["Hedged?"] == "N"]
    avg_r_no_hedge = not_hedged["R_Multiple"].mean() if len(not_hedged) else 0

    hedged = df_all[df_all["Hedged?"] == "Y"]
    hedged_count = len(hedged)
    avg_r_hedged = hedged["R_Multiple"].mean() if hedged_count else 0

    avg_win  = wins["Result"].mean()   if len(wins)   else 0
    avg_loss = losses["Result"].mean() if len(losses) else 0
    avg_rr   = abs(avg_win / avg_loss) if avg_loss != 0 else float("nan")

    returns = df_all["Result"]
    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(len(returns))
    else:
        sharpe = float("nan")

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
    print(f"{' / '.join(SYMBOLS)}  5m進場(POC突破)  15m VP TP+SL  [v21 邏輯修正版]")
    print(f"倉位：{NOTIONAL} USDT/筆  手續費：VIP0 {FEE_RATE*100}%  資金費率：{FUND_RATE*100}%/8h（按真實結算點計算）")
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

    out = "backtest_result_v20sb.csv"
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