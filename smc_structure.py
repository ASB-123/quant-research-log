"""
smc_structure.py
====================
移植自 LuxAlgo Smart Money Concepts 的核心結構判斷邏輯：
1. Pivot（Swing High / Swing Low）偵測，帶確認延遲（跟 Pine `leg()` 邏輯一致）
2. BOS（Break of Structure）/ CHoCH（Change of Character）判斷
3. Order Block 綁定結構轉折點（跟隨 LuxAlgo storeOrderBlock 的抓取方式：
   在 pivot 到突破的區間內，找「波動過濾後的極值K棒」作為OB）
4. 依 liquidity_zones.get_entry_timeframe 決定：紐約盤用5m結構，其餘用15m結構

輸入資料格式：df 需含 ["time"(ms), "open","high","low","close"]，由舊到新排序。
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from liquidity_zones import get_entry_timeframe, TF_WEIGHT

BULLISH = 1
BEARISH = -1
BOS = "BOS"
CHOCH = "CHoCH"


# ----------------------------------------------
# 1. 波動過濾（跟 LuxAlgo 一致：高波動K棒的 high/low 互換，避免插針干擾OB極值）
# ----------------------------------------------

def _parsed_high_low(df: pd.DataFrame, atr_period: int = 200):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period, min_periods=1).mean().values
    high_vol_bar = (h - l) >= (2 * atr)
    parsed_high = np.where(high_vol_bar, l, h)
    parsed_low = np.where(high_vol_bar, h, l)
    return parsed_high, parsed_low


def _atr(df: pd.DataFrame, period: int = 14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean().values


# ----------------------------------------------
# 2. Pivot 偵測（leg邏輯：確認延遲 = length 根K棒）
# ----------------------------------------------

def compute_pivots(df: pd.DataFrame, length: int) -> pd.DataFrame:
    """
    回傳每個被確認的 pivot：columns = [bar_index, time, price, type]
    type: "high" 或 "low"
    確認延遲 length 根K棒（跟即時走勢一致，不偷看未來）
    """
    highs = df["high"].values
    lows = df["low"].values
    times = df["time"].values
    n = len(df)

    cur_leg = 0  # 0=bearish leg, 1=bullish leg（跟Pine常數對齊）
    pivots = []

    for i in range(length, n):
        window_high = highs[i - length + 1: i + 1].max()
        window_low = lows[i - length + 1: i + 1].min()
        new_leg_high = highs[i - length] > window_high
        new_leg_low = lows[i - length] < window_low

        prev_leg = cur_leg
        if new_leg_high:
            cur_leg = 0
        elif new_leg_low:
            cur_leg = 1

        if cur_leg != prev_leg:
            if cur_leg == 1:  # 剛轉成 bullish leg → 代表剛確認一個 low pivot
                pivots.append({
                    "bar_index": i - length, "time": int(times[i - length]),
                    "price": float(lows[i - length]), "type": "low",
                })
            else:  # 剛轉成 bearish leg → 代表剛確認一個 high pivot
                pivots.append({
                    "bar_index": i - length, "time": int(times[i - length]),
                    "price": float(highs[i - length]), "type": "high",
                })

    return pd.DataFrame(pivots)


# ----------------------------------------------
# 3. BOS / CHoCH 判斷 + 綁定OB
# ----------------------------------------------

def compute_structure_events(df: pd.DataFrame, length: int, timeframe_label: str,
                              displacement_atr_mult: float = 0.15) -> pd.DataFrame:
    """
    逐K棒模擬 LuxAlgo displayStructure 的邏輯：
    - 追蹤目前有效的 swing high / swing low pivot（尚未被突破的）
    - close 向上穿越 swing high → 多方 BOS/CHoCH，並在 pivot~突破區間找OB（區間內最高parsed_high的那根K棒）
    - close 向下穿越 swing low  → 空方 BOS/CHoCH，並在 pivot~突破區間找OB（區間內最低parsed_low的那根K棒）
    - [新增] displacement filter：突破幅度必須超過 pivot 價位 + ATR×displacement_atr_mult，
      避免價格剛好多0.01就被判定為BOS/CHoCH的假突破
    - [診斷新增] swing_amplitude / amplitude_atr_ratio：這次突破的pivot跟「對側最近一個pivot」的
      價格差，以及該差值相對當根ATR的倍數。用來事後判斷這是不是盤整區間內的雜訊型轉折。
    回傳事件表：time, type(BOS/CHoCH), direction, level, ob_high, ob_low, ob_time, timeframe, weight,
              swing_amplitude, amplitude_atr_ratio
    """
    pivots = compute_pivots(df, length)
    parsed_high, parsed_low = _parsed_high_low(df)
    atr = _atr(df)
    closes = df["close"].values
    times = df["time"].values
    n = len(df)

    events = []
    trend_bias = 0  # 0=未定, BULLISH, BEARISH

    active_high = None  # dict: bar_index, time, price, crossed
    active_low = None

    pivot_ptr = 0
    pivots_list = pivots.to_dict("records")

    for i in range(n):
        # 有新confirmed pivot就更新 active_high / active_low
        while pivot_ptr < len(pivots_list) and pivots_list[pivot_ptr]["bar_index"] <= i:
            p = pivots_list[pivot_ptr]
            if p["type"] == "high":
                active_high = {"bar_index": p["bar_index"], "time": p["time"], "price": p["price"], "crossed": False}
            else:
                active_low = {"bar_index": p["bar_index"], "time": p["time"], "price": p["price"], "crossed": False}
            pivot_ptr += 1

        c = closes[i]

        if active_high is not None and not active_high["crossed"] and \
                c > active_high["price"] + atr[i] * displacement_atr_mult:
            tag = CHOCH if trend_bias == BEARISH else BOS
            active_high["crossed"] = True
            trend_bias = BULLISH

            # 向上突破：跟LuxAlgo一致，OB取區間內「最低parsed_low」的那根K棒
            seg_lo = parsed_low[active_high["bar_index"]: i + 1]
            ob_local_idx = int(np.argmin(seg_lo))
            ob_idx = active_high["bar_index"] + ob_local_idx

            swing_amp = abs(active_high["price"] - active_low["price"]) if active_low is not None else np.nan
            amp_atr_ratio = swing_amp / atr[i] if (atr[i] and not np.isnan(swing_amp)) else np.nan

            events.append({
                "time": int(times[i]), "bar_index": i, "type": tag, "direction": "bullish",
                "level": active_high["price"], "timeframe": timeframe_label,
                "weight": TF_WEIGHT.get(timeframe_label, 1),
                "ob_time": int(times[ob_idx]),
                "ob_high": float(df["high"].iloc[ob_idx]), "ob_low": float(df["low"].iloc[ob_idx]),
                "swing_amplitude": float(swing_amp) if not np.isnan(swing_amp) else np.nan,
                "amplitude_atr_ratio": float(amp_atr_ratio) if not np.isnan(amp_atr_ratio) else np.nan,
            })

        if active_low is not None and not active_low["crossed"] and \
                c < active_low["price"] - atr[i] * displacement_atr_mult:
            tag = CHOCH if trend_bias == BULLISH else BOS
            active_low["crossed"] = True
            trend_bias = BEARISH

            # 向下突破：跟LuxAlgo一致，OB取區間內「最高parsed_high」的那根K棒
            seg_hi = parsed_high[active_low["bar_index"]: i + 1]
            ob_local_idx = int(np.argmax(seg_hi))
            ob_idx = active_low["bar_index"] + ob_local_idx

            swing_amp = abs(active_high["price"] - active_low["price"]) if active_high is not None else np.nan
            amp_atr_ratio = swing_amp / atr[i] if (atr[i] and not np.isnan(swing_amp)) else np.nan

            events.append({
                "time": int(times[i]), "bar_index": i, "type": tag, "direction": "bearish",
                "level": active_low["price"], "timeframe": timeframe_label,
                "weight": TF_WEIGHT.get(timeframe_label, 1),
                "ob_time": int(times[ob_idx]),
                "ob_high": float(df["high"].iloc[ob_idx]), "ob_low": float(df["low"].iloc[ob_idx]),
                "swing_amplitude": float(swing_amp) if not np.isnan(swing_amp) else np.nan,
                "amplitude_atr_ratio": float(amp_atr_ratio) if not np.isnan(amp_atr_ratio) else np.nan,
            })

    events_df = pd.DataFrame(events)
    if not events_df.empty:
        events_df = events_df.sort_values("bar_index").reset_index(drop=True)
        # [診斷新增] 用K棒數量算「距離上一個同時框事件」的間隔，取代固定毫秒窗口
        # （固定4小時窗口在 length=50 時連一個pivot週期都裝不滿，測不出真實的密集程度）
        events_df["bars_since_prev_event"] = events_df["bar_index"].diff()
        prev_direction = events_df["direction"].shift(1)
        flip = events_df["direction"].ne(prev_direction).astype("object")
        flip[events_df["bars_since_prev_event"].isna()] = np.nan
        events_df["direction_flip"] = flip
    return events_df


# ----------------------------------------------
# 4. 時段切換：依 get_entry_timeframe 選 5m 或 15m 的結構事件
# ----------------------------------------------

def build_dual_timeframe_structure(df5: pd.DataFrame, df15: pd.DataFrame,
                                    length5: int = 50, length15: int = 50) -> dict:
    """
    分別在 5m / 15m 上計算結構事件，回傳兩張表供之後依時段挑選使用。
    """
    ev5 = compute_structure_events(df5, length5, "5m")
    ev15 = compute_structure_events(df15, length15, "15m")
    return {"5m": ev5, "15m": ev15}


def get_active_structure_event(ts_ms: int, dual_events: dict) -> pd.Series | None:
    """
    依 ts_ms 所在時段（紐約盤用5m，其餘用15m），
    回傳「該時框上，時間 <= ts_ms 的最近一個結構事件」。
    """
    tf = get_entry_timeframe(ts_ms)
    ev = dual_events[tf]
    if ev.empty:
        return None
    valid = ev[ev["time"] <= ts_ms]
    if valid.empty:
        return None
    return valid.iloc[-1]