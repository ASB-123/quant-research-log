"""
30sa_backtesting.py
====================
[全新假說 —— 多時框 + ADX盤整濾網 + composite score進場 + 上一階時框20ma出場]

這是三個既有元件的全新組合，不是任何一個元件的重跑：

  1. 進場訊號本體：沿用 bitunix_composite_score_backtest.py 的五分量
     composite score（MACD 7/14/6、KDJ 9/3、RSI 14、量能代理、資金費率
     z-score，等權重0.2，門檻候選[0.4, 0.5]），未改動計算方式。

  2. ADX(14)盤整濾網：ADX < 25 時，該bar完全跳過composite score訊號判斷
     （不進場，不管composite score數值多少）；ADX >= 25 才看訊號。

     ADX(14)>=25這個濾網，在交接文件的分析5裡，套用在原本EMA/MACD訊號上
     已經被完全否證：五個ADX區間平均R全部擠在-0.37~-0.39窄帶，ADX與
     R_Multiple相關係數-0.007（p=0.445不顯著），標準二分法Welch's t-test
     p=0.529不顯著——虧損在所有市場狀態下均勻分布，不集中在盤整區。
     這次的差異：(a)套用對象是composite score，不是EMA/MACD；
     (b)這次疊加多時框（5m~1d），不是單一5m。這兩點讓這次測試不算
     同一假說重跑，值得測。但composite score五個分量裡，MACD跟資金費率
     這兩塊你們已經在別的腳本摸過底，KDJ/RSI又高度共線（同一類均值回歸
     訊號），如果這次結果又是負的，問題很可能出在composite本身而非
     ADX濾網沒生效——判讀時這兩種可能要分開看，不要混為一談。

  3. 多時框：5m/15m/1h/4h/8h/1d，共6個時框各自獨立跑進場+出場（1w因為
     沒有更高一階時框可看20ma，此版跳過不做）。每個時框的訊號、ADX、
     composite score全部用該時框自己的K線資料計算，彼此獨立、不互相
     過濾也不加碼疊加。

  4. 出場邏輯 —— 全新設計，取代composite score版原本的
     SCORE_EXIT/TIMEOUT機制：
       - ATR(14)×1.5 SL：保留，當硬停損（風控底線，跟composite score版
         一致的機制，未改動）
       - 20ma出場：看「上一階時框」的EMA20，若該上一階K棒的
         實體中點 (open+close)/2 收在20ma之下（多單）或之上（空單），
         視為趨勢轉向，出場。實體中點只看K棒本體，不看影線。
       - 兩者何者先觸發，依先後順序判定；ATR SL用bar內high/low判斷，
         20ma出場用上一階K棒收盤後才能確認，因此檢查頻率受限於上一階
         時框的K棒週期（例如5m倉位要等下一根15m收盤才能檢查一次
         15m的20ma條件）。

     時框→上一階對應表：
       5m→15m, 15m→1h, 1h→4h, 4h→8h, 8h→1d, 1d→1w
     （1d倉位看1w的20ma，1w本身不開倉但仍需要抓1w K線來算EMA20）

  [權重政策延續]：composite score五個分量權重固定0.2×5，不依資料回頭調整。

  [統計配套延續]：equity curve、Sharpe/Sortino/Calmar/MaxDD、stationary
  bootstrap CI、IS/OOS切分（70/30）、Beta/Alpha，全部沿用既有寫法，
  未改動計算邏輣，只是這次每個時框、每個ENTRY_THRESHOLD都各自輸出一份。
"""

import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timezone

BASE_URL = "https://fapi.bitunix.com"

SYMBOLS   = ["ETHUSDT", "BTCUSDT", "SOLUSDT"]
FEE_RATE  = 0.0006
SLIPPAGE_BPS = 2.0

ATR_PERIOD     = 14
ATR_MULTIPLIER = 1.5
ADX_PERIOD     = 14
ADX_THRESHOLD  = 25

TARGET_RISK_USDT  = 0.12
MAX_NOTIONAL_USDT = 500.0
MIN_NOTIONAL_USDT = 10.0
FEE_PCT_OF_RISK_CEILING = 0.30
MIN_RISK_PCT_OF_PRICE   = 2 * FEE_RATE / FEE_PCT_OF_RISK_CEILING

# ── composite score 分量參數（與 bitunix_composite_score_backtest.py 一致）──
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 7, 14, 6
KDJ_PERIOD, KDJ_SMOOTH = 9, 3
RSI_PERIOD = 14
ROLL_NORM_WINDOW = 200
VOLUME_RETURN_LOOKBACK = 3
FUNDING_ZSCORE_WINDOW = 90
WEIGHTS = {"macd": 0.2, "kdj": 0.2, "rsi": 0.2, "volume": 0.2, "funding": 0.2}
ENTRY_THRESHOLD_CANDIDATES = [0.4, 0.5]

EXIT_MA_PERIOD = 20   # [新增] 上一階時框出場用的EMA週期

INITIAL_CAPITAL = 10000.0
IS_RATIO = 0.70
BOOTSTRAP_N = 2000
BOOTSTRAP_BLOCK = 5

# ── [新增] 多時框設定：每個時框 → (K線抓取用的interval字串, 需要抓取的頁數, 上一階時框)──
TIMEFRAMES = {
    "5m":  {"interval": "5m",  "pages": 3000, "parent": "15m"},
    "15m": {"interval": "15m", "pages": 1500, "parent": "1h"},
    "1h":  {"interval": "1h",  "pages": 800,  "parent": "4h"},
    "4h":  {"interval": "4h",  "pages": 400,  "parent": "8h"},
    "8h":  {"interval": "8h",  "pages": 300,  "parent": "1d"},
    "1d":  {"interval": "1d",  "pages": 250,  "parent": "1w"},
    # 1w只當parent用，不獨立開倉（沒有更高一階可看20ma出場）
    "1w":  {"interval": "1w",  "pages": 60,   "parent": None},
}
TRADEABLE_TIMEFRAMES = ["5m", "15m", "1h", "4h", "8h", "1d"]

# ----------------------------------------------
# 資料抓取：K線（沿用composite score版本邏輯，interval改為參數化）
# ----------------------------------------------

def fetch_klines(symbol, interval, limit=200, end_time=None):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time:
        params["endTime"] = end_time
    for attempt in range(5):
        try:
            res = requests.get(f"{BASE_URL}/api/v1/futures/market/kline", params=params, timeout=30)
            return res.json().get("data", [])
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


def fetch_funding_rate_history(symbol, max_pages=40):
    cache = f"cache_funding_{symbol}.csv"
    if os.path.exists(cache):
        df_c = pd.read_csv(cache)
        df_c["fundingTime"] = df_c["fundingTime"].astype(int)
        all_data = df_c.to_dict("records")
        end_time = df_c["fundingTime"].min()
        print(f"  [{symbol} funding] 快取 {len(all_data)} 筆，從 {end_time} 往回繼續")
    else:
        all_data, end_time = [], None

    for p in range(max_pages):
        params = {"symbol": symbol, "limit": 200}
        if end_time:
            params["endTime"] = end_time
        batch = []
        for attempt in range(5):
            try:
                res = requests.get(
                    f"{BASE_URL}/api/v1/futures/market/get_funding_rate_history",
                    params=params, timeout=30)
                batch = res.json().get("data", [])
                break
            except Exception as e:
                wait = (attempt + 1) * 5
                print(f"\n  重試 {attempt+1}，等待 {wait}s：{e}")
                time.sleep(wait)
        if not batch:
            break
        all_data.extend(batch)
        times = [int(b["fundingTime"]) for b in batch]
        new_end = min(times) - 1
        if end_time is not None and new_end >= end_time:
            break
        end_time = new_end
        print(f"  [{symbol} funding] 第{p+1}頁，累積 {len(all_data)} 筆", end="\r")
        time.sleep(0.15)

    df = pd.DataFrame(all_data).drop_duplicates("fundingTime")
    df["fundingTime"] = df["fundingTime"].astype(int)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["markPrice"]   = pd.to_numeric(df["markPrice"], errors="coerce")
    df = df.sort_values("fundingTime").reset_index(drop=True)
    df.to_csv(cache, index=False)
    print(f"\n  [{symbol} funding] 完成，共 {len(df)} 筆")
    return df

# ----------------------------------------------
# 指標：composite score 五分量 / ADX(14)濾網 / 出場用EMA20
# 包成class，取代原本各自獨立的function，方便日後單獨測試/替換其中一塊。
# 回測迴圈本身（backtest_symbol_timeframe）仍維持function寫法，未改動風格。
# ----------------------------------------------

class CompositeScoreCalculator:
    """計算composite score五分量，邏輯與bitunix_composite_score_backtest.py
    的add_component_scores完全一致，未改動計算方式，只是改用class封裝，
    參數可在建構時覆寫，方便日後做參數敏感度測試而不用改全域變數。"""

    def __init__(self, macd_fast=MACD_FAST, macd_slow=MACD_SLOW, macd_signal=MACD_SIGNAL,
                 kdj_period=KDJ_PERIOD, kdj_smooth=KDJ_SMOOTH, rsi_period=RSI_PERIOD,
                 roll_norm_window=ROLL_NORM_WINDOW, volume_return_lookback=VOLUME_RETURN_LOOKBACK,
                 funding_zscore_window=FUNDING_ZSCORE_WINDOW, weights=None,
                 atr_period=ATR_PERIOD, atr_multiplier=ATR_MULTIPLIER):
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.kdj_period = kdj_period
        self.kdj_smooth = kdj_smooth
        self.rsi_period = rsi_period
        self.roll_norm_window = roll_norm_window
        self.volume_return_lookback = volume_return_lookback
        self.funding_zscore_window = funding_zscore_window
        self.weights = weights or dict(WEIGHTS)
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def _score_macd(self, df):
        c = df["close"]
        ema_fast = c.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = c.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        hist = macd_line - signal
        hist_std = hist.rolling(self.roll_norm_window, min_periods=50).std()
        return np.tanh(hist / hist_std.replace(0, np.nan))

    def _score_kdj(self, df):
        c, h, l = df["close"], df["high"], df["low"]
        low_min  = l.rolling(self.kdj_period, min_periods=self.kdj_period).min()
        high_max = h.rolling(self.kdj_period, min_periods=self.kdj_period).max()
        rsv = (c - low_min) / (high_max - low_min).replace(0, np.nan) * 100
        k_line = rsv.ewm(alpha=1 / self.kdj_smooth, adjust=False).mean()
        d_line = k_line.ewm(alpha=1 / self.kdj_smooth, adjust=False).mean()
        j_line = 3 * k_line - 2 * d_line
        return ((50 - j_line) / 50).clip(-1, 1)

    def _score_rsi(self, df):
        c = df["close"]
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        return ((50 - rsi) / 50).clip(-1, 1)

    def _score_volume(self, df):
        c = df["close"]
        vol = df["baseVol"].astype(float)
        vol_mean = vol.rolling(self.roll_norm_window, min_periods=50).mean()
        vol_std  = vol.rolling(self.roll_norm_window, min_periods=50).std()
        vol_z = (vol - vol_mean) / vol_std.replace(0, np.nan)
        ret_sign = np.sign(c.pct_change(self.volume_return_lookback))
        return np.tanh(vol_z * ret_sign / 2)

    def _score_funding(self, df, df_funding):
        if df_funding is None or not len(df_funding):
            return pd.Series(np.nan, index=df.index), df
        fdf = df_funding.copy()
        min_p = max(30, self.funding_zscore_window // 3)
        fdf["roll_mean"] = fdf["fundingRate"].rolling(self.funding_zscore_window, min_periods=min_p).mean()
        fdf["roll_std"]  = fdf["fundingRate"].rolling(self.funding_zscore_window, min_periods=min_p).std()
        fdf["funding_z"] = (fdf["fundingRate"] - fdf["roll_mean"]) / fdf["roll_std"].replace(0, np.nan)
        fdf = fdf.sort_values("fundingTime")
        merged = pd.merge_asof(
            df.sort_values("time"), fdf[["fundingTime", "funding_z"]],
            left_on="time", right_on="fundingTime", direction="backward",
        )
        return -np.tanh(merged["funding_z"] / 2), merged

    def _atr_and_risk(self, df):
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=self.atr_period, adjust=False).mean()
        return atr, atr * self.atr_multiplier

    def add_scores(self, df, df_funding):
        """計算五分量 + composite + ATR/risk，回傳附加欄位後的df。"""
        df = df.copy()
        df["score_macd"]   = self._score_macd(df)
        df["score_kdj"]    = self._score_kdj(df)
        df["score_rsi"]    = self._score_rsi(df)
        df["score_volume"] = self._score_volume(df)
        df["atr"], df["risk"] = self._atr_and_risk(df)

        score_funding, df = self._score_funding(df, df_funding)
        df["score_funding"] = score_funding

        # [修正過的bug] 原本直接加總會讓任一分量NaN時整個composite變NaN
        # （NaN傳播）。改成：每個分量各自乘上權重後skipna加總，並依「實際
        # 有值的分量權重和」重新正規化，讓某個分量缺值（尤其資金費率歷史
        # 涵蓋範圍可能短於K線歷史，1d/1w時框最容易遇到）時，composite仍
        # 能用其餘分量計算，而不是整根bar直接靜默作廢。
        score_cols = {
            "macd": df["score_macd"], "kdj": df["score_kdj"], "rsi": df["score_rsi"],
            "volume": df["score_volume"], "funding": df["score_funding"],
        }
        weighted = pd.DataFrame({k: score_cols[k] * self.weights[k] for k in self.weights})
        weight_mask = pd.DataFrame({k: (~score_cols[k].isna()).astype(float) * self.weights[k]
                                     for k in self.weights})
        weight_sum = weight_mask.sum(axis=1)
        df["composite"] = weighted.sum(axis=1, skipna=True) / weight_sum.replace(0, np.nan)
        df["composite_n_components"] = (weight_mask > 0).sum(axis=1)
        return df

class ADXFilter:
    """Wilder's ADX(14)，標準公式，未依資料調整參數。
    is_tradeable()供回測迴圈逐bar判斷是否該看composite score訊號。"""

    def __init__(self, period=ADX_PERIOD, threshold=ADX_THRESHOLD):
        self.period = period
        self.threshold = threshold

    def add_adx(self, df):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        up_move   = h.diff()
        down_move = -l.diff()
        plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr_w = tr.ewm(alpha=1 / self.period, adjust=False).mean()

        plus_dm_s  = pd.Series(plus_dm, index=df.index).ewm(alpha=1 / self.period, adjust=False).mean()
        minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=1 / self.period, adjust=False).mean()

        plus_di  = 100 * plus_dm_s / atr_w.replace(0, np.nan)
        minus_di = 100 * minus_dm_s / atr_w.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df["adx"] = dx.ewm(alpha=1 / self.period, adjust=False).mean()
        return df

    def is_tradeable(self, adx_value):
        """單一bar層級判斷，供回測迴圈逐bar呼叫。"""
        return not pd.isna(adx_value) and adx_value >= self.threshold


class ExitSignalCalculator:
    """出場訊號：本時框的EMA20/實體中點（供上一階時框查詢用），
    以及把上一階時框的出場訊號前向對齊到本時框每一根K棒。"""

    def __init__(self, ma_period=EXIT_MA_PERIOD):
        self.ma_period = ma_period

    def add_exit_ema(self, df):
        """[新增] 出場用EMA20，以及該K棒的實體中點(不看影線)。"""
        df = df.copy()
        df["ema_exit"] = df["close"].ewm(span=self.ma_period, adjust=False).mean()
        df["body_mid"] = (df["open"] + df["close"]) / 2.0
        return df

    def align_parent_exit_signal(self, df_child, df_parent):
        """將「上一階時框」的body_mid/ema_exit，前向對齊到本時框每一根K棒。
        用time-1做merge_asof(backward)，確保只會拿到「已經收盤」的上一階
        K棒，避免用到同一時刻剛好收盤的那根造成未來函數。"""
        if df_parent is None or len(df_parent) == 0:
            df_child = df_child.copy()
            df_child["parent_body_mid"] = np.nan
            df_child["parent_ema_exit"] = np.nan
            return df_child

        parent = df_parent[["time", "body_mid", "ema_exit"]].rename(
            columns={"body_mid": "parent_body_mid", "ema_exit": "parent_ema_exit"}
        ).sort_values("time")
        child = df_child.sort_values("time").copy()
        child["_lookup_time"] = child["time"] - 1
        merged = pd.merge_asof(
            child, parent, left_on="_lookup_time", right_on="time",
            direction="backward", suffixes=("", "_parent_dup"),
        )
        merged = merged.drop(columns=["_lookup_time"])
        if "time_parent_dup" in merged.columns:
            merged = merged.drop(columns=["time_parent_dup"])
        return merged.reset_index(drop=True)

    def check_exit(self, side, parent_body_mid, parent_ema_exit):
        """單一bar層級判斷：這個持倉該不該因上一階時框20ma反轉而出場。"""
        if pd.isna(parent_body_mid) or pd.isna(parent_ema_exit):
            return False
        if side == "BUY":
            return parent_body_mid < parent_ema_exit
        else:
            return parent_body_mid > parent_ema_exit

# ----------------------------------------------
# 滑價 / 同根K棒判斷
# ----------------------------------------------

def apply_slippage(price, side, is_entry):
    s = SLIPPAGE_BPS / 10000.0
    if side == "BUY":
        return price * (1 + s) if is_entry else price * (1 - s)
    else:
        return price * (1 - s) if is_entry else price * (1 + s)


def resolve_same_bar_sl_first(side, low, high, sl_price):
    return (side == "BUY" and low <= sl_price) or (side == "SELL" and high >= sl_price)

# ----------------------------------------------
# [新增] 將「上一階時框」的資料，前向對齊到「本時框」的每一根K棒
# ----------------------------------------------

def align_parent_exit_signal(df_child, df_parent):
    if df_parent is None or len(df_parent) == 0:
        df_child = df_child.copy()
        df_child["parent_body_mid"] = np.nan
        df_child["parent_ema_exit"] = np.nan
        return df_child

    parent = df_parent[["time", "body_mid", "ema_exit"]].rename(
        columns={"body_mid": "parent_body_mid", "ema_exit": "parent_ema_exit"}
    ).sort_values("time")
    child = df_child.sort_values("time").copy()
    child["_lookup_time"] = child["time"] - 1   # 嚴格小於，避免同時間點洩漏未來資訊
    merged = pd.merge_asof(
        child, parent, left_on="_lookup_time", right_on="time",
        direction="backward", suffixes=("", "_parent_dup"),
    )
    merged = merged.drop(columns=["_lookup_time"])
    if "time_parent_dup" in merged.columns:
        merged = merged.drop(columns=["time_parent_dup"])
    return merged.reset_index(drop=True)

# ----------------------------------------------
# 單幣單時框回測
# ----------------------------------------------

def backtest_symbol_timeframe(symbol, timeframe, df, entry_threshold,
                               adx_filter=None, exit_calc=None):
    """回測迴圈本身維持function寫法（未改用class），只是逐bar判斷改呼叫
    adx_filter.is_tradeable() / exit_calc.check_exit()，取代原本直接讀
    欄位再手刻條件式，讓判斷邏輯集中在對應的class裡維護。"""
    adx_filter = adx_filter or ADXFilter()
    exit_calc  = exit_calc or ExitSignalCalculator()

    positions = []
    trades = []
    daily_records = []
    running_realized = 0.0
    current_date = None

    df = df.reset_index(drop=True)
    warmup = max(ROLL_NORM_WINDOW, ATR_PERIOD * 3, ADX_PERIOD * 3, EXIT_MA_PERIOD * 3)
    warmup = min(warmup, len(df) - 1) if len(df) > 1 else 0

    for i in range(warmup, len(df)):
        row = df.iloc[i]
        o = float(row["open"]); high = float(row["high"]); low = float(row["low"])
        t = int(row["time"])
        composite = row["composite"]
        adx = row.get("adx", np.nan)
        parent_body_mid = row.get("parent_body_mid", np.nan)
        parent_ema_exit = row.get("parent_ema_exit", np.nan)

        row_date = datetime.fromtimestamp(t / 1000, tz=timezone.utc).date()
        if current_date is None:
            current_date = row_date
        elif row_date != current_date:
            mark_price = float(df.iloc[i - 1]["close"])
            unreal = 0.0
            for pos in positions:
                diff = (mark_price - pos["entry_price"]) * pos["qty"]
                if pos["side"] == "SELL":
                    diff = -diff
                unreal += diff + pos["realized_pnl"]
            daily_records.append({"date": current_date, "unrealized_incl_partial": unreal,
                                   "realized_closed_cum": running_realized})
            current_date = row_date

        # ── 檢查現有持倉：ATR SL 或 上一階時框20ma實體中點反轉 ──
        closed_idx = []
        for pi, pos in enumerate(positions):
            side = pos["side"]
            sl_hit = resolve_same_bar_sl_first(side, low, high, pos["sl"])
            ma_exit = exit_calc.check_exit(side, parent_body_mid, parent_ema_exit)

            if sl_hit:
                exit_fill = apply_slippage(pos["sl"], side, is_entry=False)
                result = "SL"
            elif ma_exit:
                exit_fill = apply_slippage(o, side, is_entry=False)
                result = "MA_EXIT"
            else:
                continue

            notional = pos["entry_price"] * pos["qty"]
            pnl_raw = (exit_fill - pos["entry_price"]) * pos["qty"]
            if side == "SELL":
                pnl_raw = -pnl_raw
            fee = notional * FEE_RATE
            net = pnl_raw - fee + pos["realized_pnl"]
            running_realized += net

            trades.append(_make_trade(symbol, timeframe, pos, t, exit_fill, result, net,
                                      fee + pos["fee_paid"], i))
            closed_idx.append(pi)

        for pi in sorted(closed_idx, reverse=True):
            positions.pop(pi)

        if pd.isna(composite) or not adx_filter.is_tradeable(adx):
            continue

        if len(positions) == 0:
            risk_dist = float(row["risk"])
            if risk_dist <= 0 or (risk_dist / float(row["close"])) < MIN_RISK_PCT_OF_PRICE:
                continue
            side = None
            if composite >= entry_threshold:
                side = "BUY"
            elif composite <= -entry_threshold:
                side = "SELL"
            if side:
                entry = float(row["close"])
                entry_fill = apply_slippage(entry, side, is_entry=True)
                sl = entry_fill - risk_dist if side == "BUY" else entry_fill + risk_dist
                qty = TARGET_RISK_USDT / risk_dist
                notional = qty * entry_fill
                if MIN_NOTIONAL_USDT <= notional <= MAX_NOTIONAL_USDT:
                    open_fee = notional * FEE_RATE
                    risk_usdt = risk_dist * qty
                    positions.append({
                        "symbol": symbol, "side": side, "entry_price": entry_fill, "sl": sl,
                        "qty": qty, "realized_pnl": -open_fee, "fee_paid": open_fee,
                        "entry_time": t, "entry_bar": i, "risk_usdt": risk_usdt,
                        "entry_adx": float(adx), "entry_composite": float(composite),
                    })

    if current_date is not None:
        mark_price = float(df.iloc[-1]["close"])
        unreal = 0.0
        for pos in positions:
            diff = (mark_price - pos["entry_price"]) * pos["qty"]
            if pos["side"] == "SELL":
                diff = -diff
            unreal += diff + pos["realized_pnl"]
        daily_records.append({"date": current_date, "unrealized_incl_partial": unreal,
                               "realized_closed_cum": running_realized})

    return trades, daily_records


def _make_trade(symbol, timeframe, pos, exit_time, exit_price, result, net_pnl, fee_total, exit_bar):
    risk = pos["risk_usdt"] if pos["risk_usdt"] > 0 else 1e-9
    r_multiple = net_pnl / risk
    return {
        "symbol": symbol, "timeframe": timeframe, "side": pos["side"],
        "entry_ts": int(pos["entry_time"]), "exit_ts": int(exit_time),
        "entry_time": datetime.fromtimestamp(pos["entry_time"] / 1000).strftime("%Y-%m-%d %H:%M"),
        "exit_time": datetime.fromtimestamp(exit_time / 1000).strftime("%Y-%m-%d %H:%M"),
        "entry": round(pos["entry_price"], 4), "exit": round(exit_price, 4),
        "result": result, "net_pnl": round(net_pnl, 4), "fee_total": round(fee_total, 4),
        "risk_usdt": round(risk, 4), "R_Multiple": round(r_multiple, 4),
        "bars_held": exit_bar - pos["entry_bar"],
        "entry_adx": round(pos["entry_adx"], 2),
        "entry_composite": round(pos["entry_composite"], 4),
    }

# ----------------------------------------------
# 績效工具（沿用composite score版本邏輯，未改動）
# ----------------------------------------------

def build_portfolio_equity(daily_records_by_symbol, initial_capital=INITIAL_CAPITAL):
    all_dates = sorted(set(rec["date"] for recs in daily_records_by_symbol.values() for rec in recs))
    if not all_dates:
        return pd.Series(dtype=float)
    per_symbol_series = {}
    for sym, recs in daily_records_by_symbol.items():
        recs_sorted = sorted(recs, key=lambda r: r["date"])
        s_realized = pd.Series({r["date"]: r["realized_closed_cum"] for r in recs_sorted}) \
            .reindex(all_dates).ffill().fillna(0.0)
        s_unreal = pd.Series({r["date"]: r["unrealized_incl_partial"] for r in recs_sorted}) \
            .reindex(all_dates).fillna(0.0)
        per_symbol_series[sym] = s_realized + s_unreal
    combined = sum(per_symbol_series.values())
    equity = initial_capital + combined
    equity.index = pd.to_datetime(equity.index)
    return equity.sort_index()


def compute_perf_stats(equity, trades_df=None, periods_per_year=365, initial_capital=INITIAL_CAPITAL):
    stats = {}
    ret = equity.diff().dropna() / initial_capital
    if len(ret) < 2 or ret.std() == 0:
        stats.update({"Sharpe": np.nan, "Sortino": np.nan, "Calmar": np.nan,
                       "MaxDD_pct": np.nan, "AnnReturn_pct": np.nan})
        return stats, ret
    ann_return = ret.mean() * periods_per_year
    ann_vol = ret.std() * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    downside = ret[ret < 0]
    down_vol = downside.std() * np.sqrt(periods_per_year) if len(downside) > 1 else np.nan
    sortino = ann_return / down_vol if down_vol and down_vol > 0 else np.nan
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan
    stats.update({"Sharpe": sharpe, "Sortino": sortino, "Calmar": calmar,
                   "MaxDD_pct": max_dd * 100, "AnnReturn_pct": ann_return * 100,
                   "AnnVol_pct": ann_vol * 100})
    if trades_df is not None and len(trades_df):
        wins = trades_df[trades_df["net_pnl"] > 0]["net_pnl"]
        losses = trades_df[trades_df["net_pnl"] <= 0]["net_pnl"]
        gross_win = wins.sum(); gross_loss = abs(losses.sum())
        stats["ProfitFactor"] = gross_win / gross_loss if gross_loss > 0 else np.nan
        stats["Expectancy_USDT"] = trades_df["net_pnl"].mean()
    if (equity <= 0).any():
        stats["_capital_inadequate_warning"] = True
    return stats, ret


def stationary_bootstrap_sharpe(ret, n_boot=BOOTSTRAP_N, avg_block=BOOTSTRAP_BLOCK,
                                 periods_per_year=365, seed=42):
    rng = np.random.default_rng(seed)
    r = ret.values; n = len(r)
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
        sharpes[b] = np.nan if sd == 0 else (mu * periods_per_year) / (sd * np.sqrt(periods_per_year))
    valid = sharpes[~np.isnan(sharpes)]
    if len(valid) == 0:
        return np.nan, np.nan, valid
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return lo, hi, valid


def compute_beta_alpha(strat_ret, symbol_close_caches, periods_per_year=365):
    daily_closes = {}
    for sym, df in symbol_close_caches.items():
        s = df.copy()
        s["date"] = pd.to_datetime(s["time"], unit="ms").dt.date
        daily_closes[sym] = s.groupby("date")["close"].last()
    df_close = pd.DataFrame(daily_closes).sort_index().ffill()
    basket_ret = df_close.pct_change().mean(axis=1).dropna()
    basket_ret.index = pd.to_datetime(basket_ret.index)
    aligned = pd.concat([strat_ret.rename("strat"), basket_ret.rename("bench")], axis=1, sort=True).dropna()
    if len(aligned) < 10 or aligned["bench"].var() == 0:
        return np.nan, np.nan, len(aligned)
    beta = aligned["strat"].cov(aligned["bench"]) / aligned["bench"].var()
    alpha_daily = aligned["strat"].mean() - beta * aligned["bench"].mean()
    return beta, alpha_daily * periods_per_year, len(aligned)


def _print_stats_block(label, equity, trades_df, symbol_close_caches):
    stats, ret = compute_perf_stats(equity, trades_df)
    print(f"\n【{label}】區間: {equity.index.min().date()} ~ {equity.index.max().date()}（{len(equity)} 天）")
    if stats.get("_capital_inadequate_warning"):
        print(f"  [!] 權益曲線曾經<=0，最大回撤%數字會失真")
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
    print(f"  Sharpe 95% CI  : [{lo:.3f}, {hi:.3f}]  (stationary bootstrap, n={BOOTSTRAP_N}, avg_block={BOOTSTRAP_BLOCK}天)")
    if not np.isnan(lo) and lo < 0 < hi:
        print(f"  → CI涵蓋0，樣本數不足以證明Sharpe顯著大於0")
    beta, alpha, n_days = compute_beta_alpha(ret, symbol_close_caches)
    print(f"  Beta (vs三幣等權basket) : {beta:.3f}  （對齊天數={n_days}）")
    print(f"  Alpha (年化)            : {alpha:+.4f}")


def print_results_for_timeframe(timeframe, all_trades, dfs, daily_records_by_symbol,
                                 run_label="", entry_threshold=None):
    if not all_trades:
        print(f"\n[{timeframe}] 無交易信號（可能是ADX濾網太嚴或ENTRY_THRESHOLD太嚴格）")
        return
    df_all = pd.DataFrame(all_trades)
    sym_counts = df_all.groupby("symbol").size().to_dict()
    total = len(df_all)
    wins = df_all[df_all["net_pnl"] > 0]
    losses = df_all[df_all["net_pnl"] <= 0]
    total_net = df_all["net_pnl"].sum()
    total_fee = df_all["fee_total"].sum()
    avg_win = wins["net_pnl"].mean() if len(wins) else 0
    avg_loss = losses["net_pnl"].mean() if len(losses) else 0
    avg_r = df_all["R_Multiple"].mean()
    long_trades = df_all[df_all["side"] == "BUY"]
    short_trades = df_all[df_all["side"] == "SELL"]

    print(f"\n{'='*68}")
    print(f"[時框: {timeframe}]  {' / '.join(SYMBOLS)}  "
          f"composite score + ADX(14)>={ADX_THRESHOLD}濾網 + 上一階時框EMA{EXIT_MA_PERIOD}實體中點出場 "
          f"[30sa{run_label}]  ENTRY_THRESHOLD={entry_threshold}")
    print(f"風控：固定風險 {TARGET_RISK_USDT} USDT/筆  SL：ATR({ATR_PERIOD})×{ATR_MULTIPLIER}（硬停損，與20ma出場並存）")
    print(f"手續費：{FEE_RATE*100}%  滑價：{SLIPPAGE_BPS}bps")
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
    result_dist = df_all["result"].value_counts().to_dict()
    print(f"出場分佈    : {result_dist}")
    print(f"總淨盈虧    : {total_net:+.4f} USDT")
    print(f"總手續費    : -{total_fee:.4f} USDT")
    print(f"平均盈利    : {avg_win:+.4f} USDT")
    print(f"平均虧損    : {avg_loss:+.4f} USDT")
    print(f"平均R倍數   : {avg_r:+.4f}")
    if avg_loss != 0:
        print(f"盈虧比      : {abs(avg_win/avg_loss):.2f}")

    equity_full = build_portfolio_equity(daily_records_by_symbol)
    if len(equity_full) < 5:
        print(f"\n[!] [{timeframe}] 天數過少，無法計算有意義的權益曲線指標")
        return
    symbol_close_caches = {sym: dfs[sym][["time", "close"]] for sym in SYMBOLS}

    print(f"\n{'='*68}")
    print(f"[{timeframe}] 風險調整指標（基於逐日Mark-to-Market權益曲線）")
    print(f"{'='*68}")
    _print_stats_block("全樣本 (Full Sample)", equity_full, df_all, symbol_close_caches)

    split_idx = int(len(equity_full) * IS_RATIO)
    if split_idx < 1 or split_idx >= len(equity_full) - 1:
        print(f"\n[!] [{timeframe}] 天數不足以做有意義的IS/OOS切分，略過")
    else:
        split_date = equity_full.index[split_idx]
        equity_is = equity_full.iloc[:split_idx + 1]
        equity_oos = equity_full.iloc[split_idx:]
        trades_is = df_all[pd.to_datetime(df_all["exit_ts"], unit="ms") < split_date]
        trades_oos = df_all[pd.to_datetime(df_all["exit_ts"], unit="ms") >= split_date]

        print(f"\n{'='*68}")
        print(f"[{timeframe}] 樣本外驗證（切分點: {split_date.date()}，IS={IS_RATIO*100:.0f}% / OOS={(1-IS_RATIO)*100:.0f}%）")
        print(f"{'='*68}")
        _print_stats_block("樣本內 In-Sample", equity_is, trades_is, symbol_close_caches)
        _print_stats_block("樣本外 Out-of-Sample", equity_oos, trades_oos, symbol_close_caches)
        print(f"\n[判讀] 若OOS遠低於IS，代表ENTRY_THRESHOLD等設定可能是雜訊被揀選出來的假象。")

    equity_full.to_csv(f"equity_curve_30sa_{timeframe}{run_label}.csv", header=["equity"])
    df_all.to_csv(f"backtest_result_30sa_{timeframe}{run_label}.csv", index=False, encoding="utf-8-sig")
    print(f"\n[{timeframe}] 交易明細已存至 backtest_result_30sa_{timeframe}{run_label}.csv")
    print(f"[{timeframe}] 權益曲線已存至 equity_curve_30sa_{timeframe}{run_label}.csv")


def print_cross_timeframe_summary(summary_rows):
    """[新增] 跨時框總覽表：一眼看出哪個時框在哪個門檻下有沒有東西，
    避免要逐一往上翻六個時框×兩個門檻共12份報告。"""
    if not summary_rows:
        return
    df = pd.DataFrame(summary_rows)
    print(f"\n\n{'#'*68}")
    print("# 跨時框總覽（Sharpe全樣本 / Sharpe 95% CI / OOS Sharpe / 交易數）")
    print(f"{'#'*68}")
    cols = ["timeframe", "entry_threshold", "n_trades", "win_rate_pct",
            "sharpe_full", "sharpe_ci_lo", "sharpe_ci_hi", "sharpe_oos"]
    df = df[cols]
    with pd.option_context("display.width", 140, "display.max_columns", None):
        print(df.to_string(index=False))
    print(f"\n[判讀提醒] 這是6個時框×2個門檻=12組獨立測試，若要嚴格解讀顯著性，"
          f"記得用Bonferroni校正（α=0.05/12≈0.0042才算通過），不要挑其中最好看的"
          f"一組時框/門檻就當作策略有效——那樣做本質上是在12次嘗試裡挑一次最佳結果，"
          f"是資料窺探的一種形式。")

# ----------------------------------------------
# 主程序
# ----------------------------------------------

if __name__ == "__main__":
    score_calc  = CompositeScoreCalculator()
    adx_filter  = ADXFilter()
    exit_calc   = ExitSignalCalculator()

    raw_dfs = {sym: {} for sym in SYMBOLS}
    funding_cache = {}

    for sym in SYMBOLS:
        print(f"\n=== {sym}：抓取資金費率歷史（供composite score用）===")
        funding_cache[sym] = fetch_funding_rate_history(sym)

        for tf, cfg in TIMEFRAMES.items():
            print(f"\n=== {sym} {tf}：抓取K線 + 計算指標 ===")
            df = fetch_all_klines(sym, cfg["interval"], cfg["pages"])
            df = score_calc.add_scores(df, funding_cache[sym])
            df = adx_filter.add_adx(df)
            df = exit_calc.add_exit_ema(df)
            raw_dfs[sym][tf] = df
            valid = df["composite"].notna().sum()
            print(f"  [{sym} {tf}] composite有效筆數: {valid} / {len(df)}")

    aligned_dfs = {sym: {} for sym in SYMBOLS}
    for sym in SYMBOLS:
        for tf in TRADEABLE_TIMEFRAMES:
            parent_tf = TIMEFRAMES[tf]["parent"]
            df_child = raw_dfs[sym][tf]
            df_parent = raw_dfs[sym].get(parent_tf) if parent_tf else None
            aligned_dfs[sym][tf] = exit_calc.align_parent_exit_signal(df_child, df_parent)

    summary_rows = []
    for tf in TRADEABLE_TIMEFRAMES:
        for entry_threshold in ENTRY_THRESHOLD_CANDIDATES:
            run_label = f"_th{str(entry_threshold).replace('.', 'p')}"
            print(f"\n\n{'#'*68}")
            print(f"# 回測 時框={tf}  ENTRY_THRESHOLD={entry_threshold}")
            print(f"{'#'*68}")

            all_trades = []
            daily_records_by_symbol = {}
            dfs_for_beta = {}
            for sym in SYMBOLS:
                print(f"  {sym} {tf} 回測中...")
                trades, daily_records = backtest_symbol_timeframe(
                    sym, tf, aligned_dfs[sym][tf], entry_threshold=entry_threshold,
                    adx_filter=adx_filter, exit_calc=exit_calc)
                all_trades.extend(trades)
                daily_records_by_symbol[sym] = daily_records
                dfs_for_beta[sym] = raw_dfs[sym][tf]
                print(f"  {sym} {tf} 完成，{len(trades)} 筆交易")

            print_results_for_timeframe(tf, all_trades, dfs_for_beta, daily_records_by_symbol,
                                        run_label=run_label, entry_threshold=entry_threshold)

            if all_trades:
                df_all = pd.DataFrame(all_trades)
                equity_full = build_portfolio_equity(daily_records_by_symbol)
                row = {"timeframe": tf, "entry_threshold": entry_threshold,
                       "n_trades": len(df_all),
                       "win_rate_pct": round((df_all["net_pnl"] > 0).mean() * 100, 1),
                       "sharpe_full": np.nan, "sharpe_ci_lo": np.nan, "sharpe_ci_hi": np.nan,
                       "sharpe_oos": np.nan}
                if len(equity_full) >= 5:
                    stats, ret = compute_perf_stats(equity_full, df_all)
                    row["sharpe_full"] = round(stats.get("Sharpe", np.nan), 3)
                    lo, hi, _ = stationary_bootstrap_sharpe(ret)
                    row["sharpe_ci_lo"] = round(lo, 3) if not np.isnan(lo) else np.nan
                    row["sharpe_ci_hi"] = round(hi, 3) if not np.isnan(hi) else np.nan
                    split_idx = int(len(equity_full) * IS_RATIO)
                    if 0 < split_idx < len(equity_full) - 1:
                        equity_oos = equity_full.iloc[split_idx:]
                        stats_oos, _ = compute_perf_stats(equity_oos)
                        row["sharpe_oos"] = round(stats_oos.get("Sharpe", np.nan), 3)
                summary_rows.append(row)

    print_cross_timeframe_summary(summary_rows)