"""
liquidity_zones.py
====================
ICT/SMC 潛在支撐壓制區（流動性）偵測模組

涵蓋範圍：
1. 關鍵開盤價位：1H / 4H / 8H / 日 / 週 / 月 開盤
2. 前日高低點
3. 三大盤別（亞盤/倫敦盤/紐約盤）判斷，自動處理冬夏令（zoneinfo）
4. 各盤別內爆量K棒時間點記錄（MAD 穩健Z分數）
5. Order Block 偵測，依時間週期給予不同權重
6. FVG 偵測（僅限 1H 以上週期）
7. 依時段自動切換進場週期（紐盤5m／其餘15m）
8. OB 止損緩衝計算（預設 2%）

輸入資料格式假設：df 需含欄位 ["time"(ms), "open","high","low","close","baseVol"]，
與現有 bitunix 回測框架（v21）抓下來的資料格式一致。
"""

from __future__ import annotations
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

# ----------------------------------------------
# 1. 盤別時區與時段定義（標準外匯盤別，本地時間，自動含夏令）
# ----------------------------------------------

SESSION_TZ = {
    "asia":   ZoneInfo("Asia/Tokyo"),
    "london": ZoneInfo("Europe/London"),
    "ny":     ZoneInfo("America/New_York"),
}

# (start_hour, end_hour) 皆為該盤別「當地時間」的小時範圍，跨零點的話 end < start
SESSION_HOURS = {
    "asia":   (9, 18),
    "london": (8, 17),
    "ny":     (8, 17),
}

TF_WEIGHT = {
    "1M": 6, "1w": 5, "1d": 4, "8h": 3.5, "4h": 3, "1h": 2, "15m": 1, "5m": 0.5,
}

FVG_MIN_TF = {"1h", "4h", "8h", "1d", "1w", "1M"}  # FVG 只在這些週期偵測


def _to_local(ts_ms: int, tz: ZoneInfo) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(tz)


def classify_session(ts_ms: int) -> list[str]:
    """回傳該時間點落在哪些盤別內（可能同時屬於多個，例如倫紐盤重疊）"""
    active = []
    for name, tz in SESSION_TZ.items():
        local = _to_local(ts_ms, tz)
        h = local.hour
        start, end = SESSION_HOURS[name]
        if start <= end:
            in_session = start <= h < end
        else:  # 跨零點
            in_session = h >= start or h < end
        if in_session:
            active.append(name)
    return active if active else ["off"]


def get_entry_timeframe(ts_ms: int) -> str:
    """紐約盤用 5m，其餘（亞盤/倫敦盤/盤外）用 15m"""
    sessions = classify_session(ts_ms)
    return "5m" if "ny" in sessions else "15m"


# ----------------------------------------------
# 2. 關鍵開盤價位（1H/4H/8H/日/週/月）
# ----------------------------------------------

def compute_key_opens(df: pd.DataFrame) -> pd.DataFrame:
    """
    輸入基礎K線（建議用最細週期，如5m或15m），
    回傳每根K棒對應的「當前所處週期」開盤價（用該週期已收線的最新一根，避免用未來資料）。
    """
    out = df.copy()
    dt_utc = pd.to_datetime(out["time"], unit="ms", utc=True)

    periods = {
        "open_1h": "1h", "open_4h": "4h", "open_8h": "8h",
        "open_1d": "1D", "open_1w": "1W", "open_1M": "1MS",
    }
    for col, freq in periods.items():
        bucket = dt_utc.dt.floor(freq) if freq not in ("1W", "1MS") else dt_utc.dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period(
            "W" if freq == "1W" else "M").dt.start_time.dt.tz_localize("UTC")
        # 每個 bucket 內第一根K棒的 open 即為該週期開盤價
        first_open = out.groupby(bucket)["open"].transform("first")
        out[col] = first_open

    return out


def compute_prev_day_high_low(df: pd.DataFrame) -> pd.DataFrame:
    """回傳每根K棒對應的「前一個已收線UTC日」高低點"""
    out = df.copy()
    dt_utc = pd.to_datetime(out["time"], unit="ms", utc=True)
    day = dt_utc.dt.floor("1D")
    daily = out.groupby(day).agg(day_high=("high", "max"), day_low=("low", "min"))
    daily_shifted = daily.shift(1)  # 前一日
    out["prev_day_high"] = day.map(daily_shifted["day_high"])
    out["prev_day_low"] = day.map(daily_shifted["day_low"])
    return out


# ----------------------------------------------
# 3. 各盤別爆量時間點（MAD 穩健Z分數）
# ----------------------------------------------

def detect_volume_spikes(df: pd.DataFrame, threshold: float = 3.5) -> pd.DataFrame:
    """
    對每根K棒依所屬盤別分組，用 MAD (Median Absolute Deviation) 計算穩健Z分數，
    標記爆量K棒。回傳新增欄位：session_tags(list), vol_mad_z, is_vol_spike
    """
    out = df.copy()
    out["session_tags"] = out["time"].apply(classify_session)

    # 展開成多對多方便分組（同一根K棒可能屬於多個盤別，如倫紐重疊）
    exploded = out.explode("session_tags").rename(columns={"session_tags": "session"})

    def mad_z(vol_series: pd.Series) -> pd.Series:
        median = vol_series.median()
        mad = (vol_series - median).abs().median()
        if mad == 0:
            return pd.Series(np.zeros(len(vol_series)), index=vol_series.index)
        # 0.6745 為常態分布下 MAD 轉換為標準差的係數
        return 0.6745 * (vol_series - median) / mad

    exploded["vol_mad_z"] = exploded.groupby("session")["baseVol"].transform(mad_z)

    # 同一根K棒若在多盤別都算，取最大 z 分數代表
    z_per_bar = exploded.groupby(exploded.index)["vol_mad_z"].max()
    out["vol_mad_z"] = z_per_bar
    out["is_vol_spike"] = out["vol_mad_z"] >= threshold
    return out


# ----------------------------------------------
# 4. Order Block 偵測（依週期給權重）
# ----------------------------------------------

def detect_order_blocks(df: pd.DataFrame, timeframe_label: str, impulse_atr_mult: float = 1.5) -> pd.DataFrame:
    """
    簡化版 OB 偵測：
    - Bullish OB：上漲衝量（該根K棒漲幅 >= impulse_atr_mult * ATR）前的「最後一根收黑K」
    - Bearish OB：下跌衝量前的「最後一根收紅K」
    回傳只含被判定為 OB 的列，附上 timeframe / weight 欄位。
    """
    out = df.copy()
    c, h, l = out["close"], out["high"], out["low"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    body = out["close"] - out["open"]

    obs = []
    for i in range(1, len(out)):
        impulse_up = body.iloc[i] > 0 and body.iloc[i] >= impulse_atr_mult * atr.iloc[i]
        impulse_down = body.iloc[i] < 0 and abs(body.iloc[i]) >= impulse_atr_mult * atr.iloc[i]

        if impulse_up:
            j = i - 1
            while j >= 0 and body.iloc[j] >= 0:
                j -= 1
            if j >= 0:
                obs.append({
                    "type": "bullish_ob", "timeframe": timeframe_label,
                    "weight": TF_WEIGHT.get(timeframe_label, 1),
                    "ob_time": int(out["time"].iloc[j]),
                    "ob_high": float(out["high"].iloc[j]), "ob_low": float(out["low"].iloc[j]),
                    "impulse_time": int(out["time"].iloc[i]),
                })
        elif impulse_down:
            j = i - 1
            while j >= 0 and body.iloc[j] <= 0:
                j -= 1
            if j >= 0:
                obs.append({
                    "type": "bearish_ob", "timeframe": timeframe_label,
                    "weight": TF_WEIGHT.get(timeframe_label, 1),
                    "ob_time": int(out["time"].iloc[j]),
                    "ob_high": float(out["high"].iloc[j]), "ob_low": float(out["low"].iloc[j]),
                    "impulse_time": int(out["time"].iloc[i]),
                })

    return pd.DataFrame(obs)


# ----------------------------------------------
# 5. FVG 偵測（僅限 1H 以上週期）
# ----------------------------------------------

def detect_fvg(df: pd.DataFrame, timeframe_label: str) -> pd.DataFrame:
    """
    標準三根K棒 FVG：
    - Bullish FVG: 第1根 high < 第3根 low → 缺口 = (第1根high, 第3根low)
    - Bearish FVG: 第1根 low > 第3根 high → 缺口 = (第3根high, 第1根low)
    只在 timeframe_label 屬於 FVG_MIN_TF 時才執行，否則回傳空表。
    """
    if timeframe_label not in FVG_MIN_TF:
        return pd.DataFrame(columns=["type", "timeframe", "weight", "gap_low", "gap_high", "time"])

    out = df.reset_index(drop=True)
    fvgs = []
    for i in range(2, len(out)):
        h1, l1 = out["high"].iloc[i - 2], out["low"].iloc[i - 2]
        h3, l3 = out["high"].iloc[i], out["low"].iloc[i]
        t3 = int(out["time"].iloc[i])

        if h1 < l3:
            fvgs.append({"type": "bullish_fvg", "timeframe": timeframe_label,
                         "weight": TF_WEIGHT.get(timeframe_label, 1),
                         "gap_low": float(h1), "gap_high": float(l3), "time": t3})
        elif l1 > h3:
            fvgs.append({"type": "bearish_fvg", "timeframe": timeframe_label,
                         "weight": TF_WEIGHT.get(timeframe_label, 1),
                         "gap_low": float(h3), "gap_high": float(l1), "time": t3})

    return pd.DataFrame(fvgs)


# ----------------------------------------------
# 6. OB 止損緩衝
# ----------------------------------------------

def sl_from_ob(ob_low: float, ob_high: float, side: str, buffer_pct: float = 0.02) -> float:
    """
    多單：SL 設在 ob_low 下方 buffer_pct
    空單：SL 設在 ob_high 上方 buffer_pct
    """
    if side == "BUY":
        return ob_low * (1 - buffer_pct)
    else:
        return ob_high * (1 + buffer_pct)


def sl_from_ob_range(ob_low: float, ob_high: float, side: str, buffer_ratio: float = 0.15) -> float:
    """
    緩衝跟著OB區間本身的大小走（OB區間 × buffer_ratio），而非固定價格百分比。
    5m的OB通常較窄，緩衝自然跟著變窄；15m/1h的OB較寬，緩衝也跟著變寬——
    比固定2%價格緩衝更合理，避免5m進場時SL被拉得比OB本身還遠。
    """
    rng = ob_high - ob_low
    buf = rng * buffer_ratio
    if side == "BUY":
        return ob_low - buf
    else:
        return ob_high + buf