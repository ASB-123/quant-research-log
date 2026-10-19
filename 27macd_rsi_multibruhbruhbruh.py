"""
bitunix_composite_score_backtest.py
====================
[策略] 綜合加權多指標評分（Composite Weighted Score）

四個指標，各自代表不同性質的訊號來源，等權重（各0.25）合成一個介於
[-1, +1]的總分，總分超過門檻才進場：
  1. MACD(12,26,9) 直方圖 → 趨勢跟隨（用標準週期，不是v3測過的7/14/6，
     因為那組已經被證明沒有額外優勢，這裡用最少調參的通用版本）
  2. RSI(14) → 均值回歸
  3. 成交量z-score × 近期報酬方向 → [⚠] 這是「量能代理」，不是真正的
     訂單流/微觀結構指標。真正的訂單流需要逐筆成交/深度歷史資料，一般
     交易所REST API不提供這種歷史回溯，這裡用K線自帶的成交量欄位做近似，
     解讀時請記得這個限制。
  4. 資金費率 z-score（沿用 funding_reversal 腳本的邏輯與正負號慣例，
     擁擠度極端時給出反向分數）→ 另類資料

[權重政策 —— 這是這次實驗設計最重要的一點]
  四個指標權重固定為 0.25 / 0.25 / 0.25 / 0.25，在看到這批資料的回測結果
  之前就決定，不依照這批資料的表現回頭調整。這是你們討論後選定的路線：
  乾淨、可以直接當confirmatory測試，代價是如果某個指標的權重"憑經驗給錯"，
  策略表現會變差——但那是假說本身的問題，不是資料污染的問題，兩者要分清楚。

[⚠ 誠實揭露：這不是一個全新的假說組合]
  MACD這個分量本身，你們已經在原始版本跟v3版本測過類似的邏輯，沒有找到
  方向性edge。資金費率分量，你們在funding_reversal腳本裡也已經用同一套
  z-score邏輯測過一次（那次是完全用資金費率單獨決定方向，這次只是它的
  25%權重）。真正沒被測過的只有RSI和量能代理這兩塊。
  這代表：如果這次的綜合分數結果是負的，那不特別意外——因為四個分量裡
  至少一半的類型你已經摸過底。如果結果是正的，你會需要拆解過（下面的
  「分量診斷」區塊）之後才能判斷，是新加入的RSI/量能真的在貢獻，還是
  純粹運氣或MACD/資金費率剛好那段時間表現不差。

[分量診斷 —— 沿用你們既有的加碼規則診斷精神]
  每筆交易記錄下進場當下四個分量各自的分數。回測結束後拆解：當某個
  分量的方向跟交易方向一致（"agree"）時，績效是否顯著優於不一致
  （"disagree"）的組別。這個拆解直接借用你們原本"1倍vs2倍加碼"的
  Welch's t-test做法，只是這次是套用在四個分量各自身上。
"""

import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timezone

BASE_URL = "https://fapi.bitunix.com"

SYMBOLS   = ["ETHUSDT", "BTCUSDT", "SOLUSDT"]
PAGES_5M  = 3000
FEE_RATE  = 0.0006
SLIPPAGE_BPS = 2.0

ATR_PERIOD     = 14
ATR_MULTIPLIER = 1.5

TARGET_RISK_USDT  = 0.12
MAX_NOTIONAL_USDT = 500.0
MIN_NOTIONAL_USDT = 10.0
FEE_PCT_OF_RISK_CEILING = 0.30
MIN_RISK_PCT_OF_PRICE   = 2 * FEE_RATE / FEE_PCT_OF_RISK_CEILING

# ── 分量參數 ──
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 7, 14, 6
RSI_PERIOD = 14
ROLL_NORM_WINDOW = 200   # MACD直方圖 / 成交量 z-score 用的滾動標準化窗口
VOLUME_RETURN_LOOKBACK = 3
FUNDING_ZSCORE_WINDOW = 90

# ── 權重（固定，等權重，不依資料調整） ──
WEIGHTS = {"macd": 0.25, "rsi": 0.25, "volume": 0.25, "funding": 0.25}

# ── 進出場門檻 ──
ENTRY_THRESHOLD_CANDIDATES = [0.4, 0.5]   # sweep兩個候選，比照你們原本MIN_TP1_RR的sweep風格
EXIT_THRESHOLD = 0.15                      # 總分回落到此絕對值以下 → 出場（訊號自己說沒理由再抱了）
MAX_HOLD_BARS  = 288                        # 24小時（5m×288），逾時強制出場的安全閥

INITIAL_CAPITAL = 10000.0
IS_RATIO = 0.70
BOOTSTRAP_N = 2000
BOOTSTRAP_BLOCK = 5

# ----------------------------------------------
# 資料抓取：5分K（含baseVol，這次量能分量需要用到）
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


# ----------------------------------------------
# 資料抓取：資金費率歷史（與 funding_reversal_backtest.py 相同）
# ----------------------------------------------

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
                batch = []
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
# 四個分量的計算
# ----------------------------------------------

def add_component_scores(df5, df_funding):
    df = df5.copy()
    c = df["close"]; h = df["high"]; l = df["low"]

    # --- 1. MACD 趨勢分量 ---
    ema_fast = c.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = c.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    hist = macd_line - signal
    hist_std = hist.rolling(ROLL_NORM_WINDOW, min_periods=50).std()
    df["score_macd"] = np.tanh(hist / hist_std.replace(0, np.nan))

    # --- 2. RSI 均值回歸分量 ---
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    df["rsi"] = rsi
    df["score_rsi"] = ((50 - rsi) / 50).clip(-1, 1)   # RSI低(超賣)→正分(偏多)；RSI高(超買)→負分(偏空)

    # --- 3. 量能代理分量（非真訂單流，見檔頭說明）---
    vol = df["baseVol"].astype(float)
    vol_mean = vol.rolling(ROLL_NORM_WINDOW, min_periods=50).mean()
    vol_std  = vol.rolling(ROLL_NORM_WINDOW, min_periods=50).std()
    vol_z = (vol - vol_mean) / vol_std.replace(0, np.nan)
    ret_sign = np.sign(c.pct_change(VOLUME_RETURN_LOOKBACK))
    df["score_volume"] = np.tanh(vol_z * ret_sign / 2)

    # --- ATR（風控用，不算分量）---
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    df["atr"]  = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    df["risk"] = df["atr"] * ATR_MULTIPLIER

    # --- 4. 資金費率 z-score 分量（另類資料，8h更新一次，用merge_asof前向對齊到每根5m K）---
    fdf = df_funding.copy()
    fdf["roll_mean"] = fdf["fundingRate"].rolling(
        FUNDING_ZSCORE_WINDOW, min_periods=max(30, FUNDING_ZSCORE_WINDOW // 3)).mean()
    fdf["roll_std"] = fdf["fundingRate"].rolling(
        FUNDING_ZSCORE_WINDOW, min_periods=max(30, FUNDING_ZSCORE_WINDOW // 3)).std()
    fdf["funding_z"] = (fdf["fundingRate"] - fdf["roll_mean"]) / fdf["roll_std"].replace(0, np.nan)
    fdf = fdf.sort_values("fundingTime")

    df = pd.merge_asof(
        df.sort_values("time"), fdf[["fundingTime", "funding_z"]],
        left_on="time", right_on="fundingTime", direction="backward",
    )
    # 沿用funding_reversal的方向慣例：擁擠度極端(z很大)→反向給分（負分＝偏空）
    df["score_funding"] = -np.tanh(df["funding_z"] / 2)

    # --- 合成總分（固定等權重）---
    df["composite"] = (
        WEIGHTS["macd"]    * df["score_macd"] +
        WEIGHTS["rsi"]     * df["score_rsi"] +
        WEIGHTS["volume"]  * df["score_volume"] +
        WEIGHTS["funding"] * df["score_funding"]
    )
    return df


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
# 單幣回測
# ----------------------------------------------

def backtest_symbol(symbol, df, entry_threshold):
    positions = []
    trades = []
    daily_records = []
    running_realized = 0.0
    current_date = None

    df = df.reset_index(drop=True)
    warmup = max(ROLL_NORM_WINDOW, ATR_PERIOD * 3)

    for i in range(warmup, len(df)):
        row = df.iloc[i]
        o = float(row["open"]); high = float(row["high"]); low = float(row["low"])
        t = int(row["time"])
        composite = row["composite"]

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

        if pd.isna(composite):
            continue

        # ── 檢查現有持倉：ATR SL / 總分反轉出場 / 逾時 ──
        closed_idx = []
        for pi, pos in enumerate(positions):
            side = pos["side"]
            sl_hit = resolve_same_bar_sl_first(side, low, high, pos["sl"])
            bars_held = i - pos["entry_bar"]
            score_exit = (side == "BUY" and composite < EXIT_THRESHOLD) or \
                         (side == "SELL" and composite > -EXIT_THRESHOLD)
            timeout = bars_held >= MAX_HOLD_BARS

            if sl_hit:
                exit_fill = apply_slippage(pos["sl"], side, is_entry=False)
                result = "SL"
            elif score_exit:
                exit_fill = apply_slippage(o, side, is_entry=False)
                result = "SCORE_EXIT"
            elif timeout:
                exit_fill = apply_slippage(float(row["close"]), side, is_entry=False)
                result = "TIMEOUT"
            else:
                continue

            notional = pos["entry_price"] * pos["qty"]
            pnl_raw = (exit_fill - pos["entry_price"]) * pos["qty"]
            if side == "SELL":
                pnl_raw = -pnl_raw
            fee = notional * FEE_RATE
            net = pnl_raw - fee + pos["realized_pnl"]
            running_realized += net

            trades.append(_make_trade(symbol, pos, t, exit_fill, result, net, fee + pos["fee_paid"], i))
            closed_idx.append(pi)

        for pi in sorted(closed_idx, reverse=True):
            positions.pop(pi)

        # ── 進場 ──
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
                        "entry_score_macd": float(row["score_macd"]),
                        "entry_score_rsi": float(row["score_rsi"]),
                        "entry_score_volume": float(row["score_volume"]),
                        "entry_score_funding": float(row["score_funding"]) if not pd.isna(row["score_funding"]) else np.nan,
                        "entry_composite": float(composite),
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


def _make_trade(symbol, pos, exit_time, exit_price, result, net_pnl, fee_total, exit_bar):
    risk = pos["risk_usdt"] if pos["risk_usdt"] > 0 else 1e-9
    r_multiple = net_pnl / risk
    return {
        "symbol": symbol, "side": pos["side"],
        "entry_ts": int(pos["entry_time"]), "exit_ts": int(exit_time),
        "entry_time": datetime.fromtimestamp(pos["entry_time"] / 1000).strftime("%m-%d %H:%M"),
        "exit_time": datetime.fromtimestamp(exit_time / 1000).strftime("%m-%d %H:%M"),
        "entry": round(pos["entry_price"], 4), "exit": round(exit_price, 4),
        "result": result, "net_pnl": round(net_pnl, 4), "fee_total": round(fee_total, 4),
        "risk_usdt": round(risk, 4), "R_Multiple": round(r_multiple, 4),
        "bars_held": exit_bar - pos["entry_bar"],
        "entry_score_macd": round(pos["entry_score_macd"], 4),
        "entry_score_rsi": round(pos["entry_score_rsi"], 4),
        "entry_score_volume": round(pos["entry_score_volume"], 4),
        "entry_score_funding": round(pos["entry_score_funding"], 4) if not pd.isna(pos["entry_score_funding"]) else np.nan,
        "entry_composite": round(pos["entry_composite"], 4),
    }


# ----------------------------------------------
# 績效工具（沿用既有邏輯）
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


def print_component_diagnostics(df_all):
    """分量診斷：拆解四個分量各自"跟交易方向一致"時的績效，比照
    原本1倍vs2倍加碼規則的Welch's t-test風格。"""
    from scipy import stats as _stats
    print(f"\n{'='*68}")
    print("分量診斷：各指標「方向與最終交易一致」vs「不一致」時的績效對照")
    print(f"{'='*68}")
    side_sign = df_all["side"].map({"BUY": 1, "SELL": -1})
    for comp in ["macd", "rsi", "volume", "funding"]:
        col = f"entry_score_{comp}"
        sub = df_all.dropna(subset=[col])
        if len(sub) < 20:
            print(f"  [{comp}] 樣本不足（{len(sub)}筆），略過")
            continue
        agree_mask = (sub[col] * side_sign.loc[sub.index]) > 0
        agree = sub[agree_mask]
        disagree = sub[~agree_mask]
        if len(agree) < 10 or len(disagree) < 10:
            print(f"  [{comp}] agree={len(agree)} / disagree={len(disagree)}，其中一組樣本太少，略過t-test")
            continue
        r_agree = agree["R_Multiple"].mean()
        r_disagree = disagree["R_Multiple"].mean()
        t_stat, p_val = _stats.ttest_ind(agree["R_Multiple"], disagree["R_Multiple"], equal_var=False)
        print(f"  [{comp:<8}] agree n={len(agree):>5} 平均R={r_agree:+.4f}  |  "
              f"disagree n={len(disagree):>5} 平均R={r_disagree:+.4f}  |  "
              f"Welch t={t_stat:+.3f} p={p_val:.4g}"
              f"{'  ← 顯著差異' if p_val < 0.05 else ''}")
    print(f"  [提醒] 這裡測了4個分量，若要嚴格解讀顯著性，記得用Bonferroni校正"
          f"（α=0.05/4=0.0125才算通過），不要看到單一p<0.05就直接下結論。")


def print_results(all_trades, dfs, daily_records_by_symbol, run_label="", entry_threshold=None):
    if not all_trades:
        print("無交易信號（可能是ENTRY_THRESHOLD太嚴格）")
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
    print(f"{' / '.join(SYMBOLS)}  綜合加權評分(MACD+RSI+量能代理+資金費率) "
          f"[composite_score{run_label}]  ENTRY_THRESHOLD={entry_threshold}  權重={WEIGHTS}")
    print(f"風控：固定風險 {TARGET_RISK_USDT} USDT/筆  SL：ATR({ATR_PERIOD})×{ATR_MULTIPLIER}  "
          f"出場：總分回落至|{EXIT_THRESHOLD}|內 或 逾時{MAX_HOLD_BARS}根")
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
        print("\n[!] 天數過少，無法計算有意義的權益曲線指標")
        return
    symbol_close_caches = {sym: dfs[sym][["time", "close"]] for sym in SYMBOLS}

    print(f"\n{'='*68}")
    print("風險調整指標（基於逐日Mark-to-Market權益曲線）")
    print(f"{'='*68}")
    _print_stats_block("全樣本 (Full Sample)", equity_full, df_all, symbol_close_caches)

    split_idx = int(len(equity_full) * IS_RATIO)
    split_date = equity_full.index[split_idx]
    equity_is = equity_full.iloc[:split_idx + 1]
    equity_oos = equity_full.iloc[split_idx:]
    trades_is = df_all[pd.to_datetime(df_all["exit_ts"], unit="ms") < split_date]
    trades_oos = df_all[pd.to_datetime(df_all["exit_ts"], unit="ms") >= split_date]

    print(f"\n{'='*68}")
    print(f"樣本外驗證（切分點: {split_date.date()}，IS={IS_RATIO*100:.0f}% / OOS={(1-IS_RATIO)*100:.0f}%）")
    print(f"{'='*68}")
    _print_stats_block("樣本內 In-Sample", equity_is, trades_is, symbol_close_caches)
    _print_stats_block("樣本外 Out-of-Sample", equity_oos, trades_oos, symbol_close_caches)
    print(f"\n[判讀] 若OOS遠低於IS，代表ENTRY_THRESHOLD這類超參數可能是雜訊被揀選出來的假象。")

    print_component_diagnostics(df_all)

    equity_full.to_csv(f"equity_curve_composite_score{run_label}.csv", header=["equity"])
    df_all.to_csv(f"backtest_result_composite_score{run_label}.csv", index=False, encoding="utf-8-sig")
    print(f"\n交易明細已存至 backtest_result_composite_score{run_label}.csv")
    print(f"權益曲線已存至 equity_curve_composite_score{run_label}.csv")


# ----------------------------------------------
# 主程序
# ----------------------------------------------

if __name__ == "__main__":
    dfs = {}

    for sym in SYMBOLS:
        print(f"\n=== {sym}（抓取5m K線 + 資金費率歷史 + 計算四個分量，只做一次）===")
        df5 = fetch_all_klines(sym, "5m", PAGES_5M)
        df_funding = fetch_funding_rate_history(sym)
        df = add_component_scores(df5, df_funding)
        dfs[sym] = df
        valid = df["composite"].notna().sum()
        print(f"  [{sym}] 總分有效筆數: {valid} / {len(df)}")

    for entry_threshold in ENTRY_THRESHOLD_CANDIDATES:
        run_label = f"_th{str(entry_threshold).replace('.', 'p')}"
        print(f"\n\n{'#'*68}")
        print(f"# 回測 ENTRY_THRESHOLD = {entry_threshold}")
        print(f"{'#'*68}")

        all_trades = []
        daily_records_by_symbol = {}
        for sym in SYMBOLS:
            print(f"  {sym} 回測中...")
            trades, daily_records = backtest_symbol(sym, dfs[sym], entry_threshold=entry_threshold)
            all_trades.extend(trades)
            daily_records_by_symbol[sym] = daily_records
            print(f"  {sym} 完成，{len(trades)} 筆交易")

        print_results(all_trades, dfs, daily_records_by_symbol, run_label=run_label, entry_threshold=entry_threshold)