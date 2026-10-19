"""
bitunix_funding_reversal_backtest.py
====================
[假說] 資金費率反轉(Funding Rate Mean-Reversion)

跟你先前測過的所有版本(原始SMC訊號、反轉訊號、ADX regime、MACD7/14/6+加碼)
最根本的差異：這次的訊號來源不是K線價格形狀，而是資金費率——一個反映「當下
多空雙方擁擠程度」的市場結構性數據，理論上跟"從歷史價格pattern猜方向"是完全
不同性質的資訊來源。

[核心邏輯]
  資金費率結算 = 多空雙方擁擠程度的直接量化。費率相對自身歷史滾動均值的
  z-score 出現極端值時：
    - z 很高（多方付費過熱，代表多方過度擁擠）→ 反向放空
    - z 很低（空方付費過熱，代表空方過度擁擠）→ 反向做多
  這是classic的"擁擠交易反轉"假說，理論基礎跟"資金費率套利者/反向對敲者
  存在"有關——但這只是假說，不是已驗證的結論，這支程式的目的就是去檢驗它。

[⚠ 資料窺探狀態說明 —— 誠實揭露]
  這是條件變數的全新假說：你先前燒過的四輪測試，燒的是「用什麼方式從K線
  形狀定義方向性訊號」這個維度，資金費率從未被當作訊號來源測過，所以以
  「資金費率z-score」為條件的訊號本身還沒被你在這批資料上調過參——這點
  是乾淨的。
  但要老實講：底層的價格資料（用來算ATR、算SL/TP、算滑價）跟你先前四輪
  用的是同一份 2022~2026 快取，這批資料本身作為「完全獨立樣本」的乾淨度
  早就不存在了。差別在於：這次的方向性判斷完全來自資金費率，不是價格
  pattern，所以價格資料被重複使用不會直接污染這次的「訊號有沒有方向性
  資訊」這個核心問題——但如果這版本看起來有效，比照先前的原則，下一步
  仍然是找全新資料或直接紙上交易驗證，不要在同一批資料上反覆調
  Z_THRESHOLD或rolling window長度。

[本版跟v3的關鍵差異]
  1. 完全捨棄EMA/MACD/VP等一切價格形狀類指標。
  2. 資金費率不再像原本calc_funding_fee_v2那樣用固定估計值(FUND_RATE=0.000574)
     去粗估資金費成本，而是逐筆套用「當下實際結算的歷史資金費率」——這既是
     這次策略的訊號來源，也讓資金費損益本身變得精確。
  3. 進場時機綁定在資金費率結算時間點(UTC 0/8/16點)，不是每根5分K都在找訊號。
  4. 出場邏輯：ATR SL / 固定R倍數TP / 最大持倉時間(逾時出場) 三選一，
     沒有VP波動剖面那套TP1/TP2/TP3分批。

[統計把關 —— 沿用你既有的機構級配套]
  equity curve、Sharpe/Sortino/Calmar/MaxDD、stationary bootstrap CI、
  Beta/Alpha（對齊三幣basket）、IS/OOS 70/30切分。全部原封不動沿用，
  因為這套配套本身跟訊號來源無關，是通用的績效驗證工具。
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

FUNDING_HOURS_UTC = [0, 8, 16]

# ── 資金費率訊號參數 ──
FUNDING_ZSCORE_WINDOW = 90     # 滾動窗口（結算筆數，8h一筆，90筆≈30天）
Z_THRESHOLD           = 2.0    # z-score絕對值超過此門檻才算「極端擁擠」
MIN_ABS_FUNDING_RATE  = 0.0003 # 額外濾網：即使z極端，費率本身太小也不進場（避免早期資料雜訊）
MAX_HOLD_SETTLEMENTS  = 3      # 最長持倉幾個結算週期（3 × 8h = 24h）就強制出場

# TP用固定R倍數（跟v3的VP剖面TP完全不同邏輯），這裡sweep兩個候選值做對照
TP_RR_CANDIDATES = [1.5, 2.5]

INITIAL_CAPITAL = 10000.0
IS_RATIO        = 0.70
BOOTSTRAP_N     = 2000
BOOTSTRAP_BLOCK = 5

# ----------------------------------------------
# 資料抓取：5分K（沿用你原本的快取模式）
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
    df = df.sort_values("time").reset_index(drop=True)
    df.to_csv(cache, index=False)
    print(f"\n  [{symbol} {interval}] 完成，共 {len(df)} 根")
    return df


# ----------------------------------------------
# 資料抓取：資金費率歷史
# 端點：GET /api/v1/futures/market/get_funding_rate_history
# 參數：symbol, starTime(官方文件原文如此拼寫), endTime, limit(最大200)
# 回傳：markPrice, fundingRate, fundingTime(ms)
# 結算週期固定8小時，往回抓約4年資料 = 365*4*3 ≈ 4380筆 → 22頁足夠，抓寬一點防漏
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
            break  # 沒有更舊的資料了，避免無限迴圈
        end_time = new_end
        print(f"  [{symbol} funding] 第{p+1}頁，累積 {len(all_data)} 筆", end="\r")
        time.sleep(0.15)

    df = pd.DataFrame(all_data).drop_duplicates("fundingTime")
    df["fundingTime"] = df["fundingTime"].astype(int)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["markPrice"]   = pd.to_numeric(df["markPrice"], errors="coerce")
    df = df.sort_values("fundingTime").reset_index(drop=True)
    df.to_csv(cache, index=False)
    print(f"\n  [{symbol} funding] 完成，共 {len(df)} 筆結算記錄，"
          f"期間 {datetime.fromtimestamp(df['fundingTime'].min()/1000, tz=timezone.utc).date()} "
          f"~ {datetime.fromtimestamp(df['fundingTime'].max()/1000, tz=timezone.utc).date()}")
    return df


def compute_funding_zscore(df_funding, window=FUNDING_ZSCORE_WINDOW):
    df = df_funding.copy()
    df["roll_mean"] = df["fundingRate"].rolling(window, min_periods=max(30, window // 3)).mean()
    df["roll_std"]  = df["fundingRate"].rolling(window, min_periods=max(30, window // 3)).std()
    df["z"] = (df["fundingRate"] - df["roll_mean"]) / df["roll_std"]
    df.loc[df["roll_std"] <= 0, "z"] = np.nan
    return df


# ----------------------------------------------
# 指標：只需要ATR（風控用），不需要EMA/MACD
# ----------------------------------------------

def add_atr(df):
    c = df["close"]; h = df["high"]; l = df["low"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    df["atr"]  = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    df["risk"] = df["atr"] * ATR_MULTIPLIER
    return df


# ----------------------------------------------
# 滑價 / 同根K棒判斷（沿用）
# ----------------------------------------------

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


# ----------------------------------------------
# 單幣回測
# ----------------------------------------------

def backtest_symbol(symbol, df5, df_funding, tp_rr):
    positions = []
    trades = []
    daily_records = []
    running_realized = 0.0
    current_date = None

    # 建立「結算時間戳 → 資金費率列」查表，5m K棒的time跟結算時間會精準對齊
    funding_lookup = {int(r.fundingTime): r for r in df_funding.itertuples()}

    df5 = df5.copy().reset_index(drop=True)

    for i in range(max(ATR_PERIOD * 3, 50), len(df5)):
        row = df5.iloc[i]
        o = float(row["open"]); high = float(row["high"]); low = float(row["low"])
        t = int(row["time"])

        row_date = datetime.fromtimestamp(t / 1000, tz=timezone.utc).date()
        if current_date is None:
            current_date = row_date
        elif row_date != current_date:
            mark_price = float(df5.iloc[i - 1]["close"])
            unreal = 0.0
            for pos in positions:
                diff = (mark_price - pos["entry_price"]) * pos["qty"]
                if pos["side"] == "SELL":
                    diff = -diff
                unreal += diff + pos["realized_pnl"]
            daily_records.append({"date": current_date, "unrealized_incl_partial": unreal,
                                   "realized_closed_cum": running_realized})
            current_date = row_date

        # ── 若此bar恰好是資金費率結算時間點，對所有持倉套用「真實歷史資金費率」 ──
        if t in funding_lookup and positions:
            frow = funding_lookup[t]
            fr = float(frow.fundingRate)
            for pos in positions:
                notional = pos["entry_price"] * pos["qty"]
                side_sign = 1.0 if pos["side"] == "BUY" else -1.0
                # 費率為正時多方付給空方：多單付費（負值），空單收費（正值）
                funding_pnl = -side_sign * notional * fr
                pos["realized_pnl"] += funding_pnl
                pos["funding_paid"] += funding_pnl
                pos["settlements_held"] += 1

        # ── 檢查現有持倉：SL / TP / 逾時出場 ──
        closed_idx = []
        for pi, pos in enumerate(positions):
            side = pos["side"]
            sl_hit, tp_hit = resolve_same_bar_sl_vs_tp(side, o, high, low, pos["sl"], pos["tp"])

            timeout = pos["settlements_held"] >= MAX_HOLD_SETTLEMENTS

            if sl_hit or tp_hit:
                exit_raw = pos["sl"] if sl_hit else pos["tp"]
                exit_fill = apply_slippage(exit_raw, side, is_entry=False)
                result = "SL" if sl_hit else "TP"
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

            trades.append(_make_trade(symbol, pos, t, exit_fill, result, net, fee + pos["fee_paid"]))
            closed_idx.append(pi)

        for pi in sorted(closed_idx, reverse=True):
            positions.pop(pi)

        # ── 進場：僅在資金費率結算時間點評估訊號，且同時間只允許一個持倉 ──
        if t in funding_lookup and len(positions) == 0:
            frow = funding_lookup[t]
            z = frow.z
            fr = float(frow.fundingRate)
            mark_price = float(frow.markPrice)
            risk_dist = float(row["risk"])

            signal_side = None
            if pd.isna(z) or risk_dist <= 0 or abs(fr) < MIN_ABS_FUNDING_RATE:
                pass
            elif (risk_dist / mark_price) < MIN_RISK_PCT_OF_PRICE:
                pass
            elif z >= Z_THRESHOLD:
                signal_side = "SELL"   # 多方過度擁擠付費 → 反向做空
            elif z <= -Z_THRESHOLD:
                signal_side = "BUY"    # 空方過度擁擠付費 → 反向做多

            if signal_side:
                side = signal_side
                entry_fill = apply_slippage(mark_price, side, is_entry=True)
                sl = entry_fill - risk_dist if side == "BUY" else entry_fill + risk_dist
                tp = entry_fill + risk_dist * tp_rr if side == "BUY" else entry_fill - risk_dist * tp_rr

                qty = TARGET_RISK_USDT / risk_dist
                notional = qty * entry_fill
                if MIN_NOTIONAL_USDT <= notional <= MAX_NOTIONAL_USDT:
                    open_fee = notional * FEE_RATE
                    risk_usdt = risk_dist * qty
                    positions.append({
                        "symbol": symbol, "side": side, "entry_price": entry_fill,
                        "sl": sl, "tp": tp, "qty": qty, "realized_pnl": -open_fee,
                        "fee_paid": open_fee, "funding_paid": 0.0, "entry_time": t,
                        "risk_usdt": risk_usdt, "settlements_held": 0,
                        "entry_z": float(z), "entry_funding_rate": fr,
                    })

    if current_date is not None:
        mark_price = float(df5.iloc[-1]["close"])
        unreal = 0.0
        for pos in positions:
            diff = (mark_price - pos["entry_price"]) * pos["qty"]
            if pos["side"] == "SELL":
                diff = -diff
            unreal += diff + pos["realized_pnl"]
        daily_records.append({"date": current_date, "unrealized_incl_partial": unreal,
                               "realized_closed_cum": running_realized})

    return trades, daily_records


def _make_trade(symbol, pos, exit_time, exit_price, result, net_pnl, fee_total):
    risk = pos["risk_usdt"] if pos["risk_usdt"] > 0 else 1e-9
    r_multiple = net_pnl / risk
    return {
        "symbol": symbol, "side": pos["side"],
        "entry_ts": int(pos["entry_time"]), "exit_ts": int(exit_time),
        "entry_time": datetime.fromtimestamp(pos["entry_time"] / 1000).strftime("%m-%d %H:%M"),
        "exit_time": datetime.fromtimestamp(exit_time / 1000).strftime("%m-%d %H:%M"),
        "entry": round(pos["entry_price"], 4), "exit": round(exit_price, 4),
        "result": result, "net_pnl": round(net_pnl, 4),
        "fee_total": round(fee_total, 4), "funding_paid": round(pos["funding_paid"], 4),
        "risk_usdt": round(risk, 4), "R_Multiple": round(r_multiple, 4),
        "settlements_held": pos["settlements_held"],
        "entry_z": round(pos["entry_z"], 3), "entry_funding_rate": pos["entry_funding_rate"],
    }


# ----------------------------------------------
# 組合層權益曲線 / 績效指標（與v3完全相同，通用邏輯，直接沿用）
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
    for sym, df5 in symbol_close_caches.items():
        s = df5.copy()
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
        print(f"  [!] 權益曲線曾經<=0，最大回撤%數字會失真，建議調高INITIAL_CAPITAL重跑")
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
        print(f"  → CI涵蓋0，樣本數不足以證明Sharpe顯著大於0，結論保留")
    beta, alpha, n_days = compute_beta_alpha(ret, symbol_close_caches)
    print(f"  Beta (vs三幣等權basket) : {beta:.3f}  （對齊天數={n_days}）")
    print(f"  Alpha (年化)            : {alpha:+.4f}")


def print_results(all_trades, dfs, daily_records_by_symbol, run_label="", tp_rr=None):
    if not all_trades:
        print("無交易信號（可能是Z_THRESHOLD太嚴格，或MIN_ABS_FUNDING_RATE濾掉太多）")
        return
    df_all = pd.DataFrame(all_trades)
    sym_counts = df_all.groupby("symbol").size().to_dict()
    total = len(df_all)
    wins = df_all[df_all["net_pnl"] > 0]
    losses = df_all[df_all["net_pnl"] <= 0]
    total_net = df_all["net_pnl"].sum()
    total_fee = df_all["fee_total"].sum()
    total_funding = df_all["funding_paid"].sum()
    avg_win = wins["net_pnl"].mean() if len(wins) else 0
    avg_loss = losses["net_pnl"].mean() if len(losses) else 0
    avg_r = df_all["R_Multiple"].mean()
    long_trades = df_all[df_all["side"] == "BUY"]
    short_trades = df_all[df_all["side"] == "SELL"]

    print(f"\n{'='*68}")
    print(f"{' / '.join(SYMBOLS)}  資金費率反轉(z-score) [funding_reversal{run_label}]  "
          f"Z_THRESHOLD={Z_THRESHOLD}  TP_RR={tp_rr}  MAX_HOLD={MAX_HOLD_SETTLEMENTS}結算")
    print(f"風控：固定風險 {TARGET_RISK_USDT} USDT/筆  SL：ATR({ATR_PERIOD})×{ATR_MULTIPLIER}")
    print(f"手續費：{FEE_RATE*100}%  滑價：{SLIPPAGE_BPS}bps  資金費：逐筆套用真實歷史結算值")
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
    print(f"總資金費損益: {total_funding:+.4f} USDT   （正值代表策略淨收到資金費，負值代表淨付出）")
    print(f"平均盈利    : {avg_win:+.4f} USDT")
    print(f"平均虧損    : {avg_loss:+.4f} USDT")
    print(f"平均R倍數   : {avg_r:+.4f}")
    if avg_loss != 0:
        print(f"盈虧比      : {abs(avg_win/avg_loss):.2f}")

    # Wilson CI 檢驗勝率是否顯著偏離隨機
    from math import sqrt
    n = total; p_hat = len(wins) / n
    z = 1.96
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = (z * sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    print(f"勝率 Wilson 95% CI : [{(center-half)*100:.1f}%, {(center+half)*100:.1f}%]")

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
    print(f"\n[判讀] 若OOS遠低於IS（尤其翻負），代表IS階段可能是雜訊被z-score門檻/持倉"
          f"時限這些超參數揀選出來的假象。")

    equity_full.to_csv(f"equity_curve_funding_reversal{run_label}.csv", header=["equity"])
    try:
        df_all.to_csv(f"backtest_result_funding_reversal{run_label}.csv", index=False, encoding="utf-8-sig")
        print(f"\n交易明細已存至 backtest_result_funding_reversal{run_label}.csv")
        print(f"權益曲線已存至 equity_curve_funding_reversal{run_label}.csv")
    except PermissionError:
        print(f"\n[!] 檔案被佔用，請關閉 Excel")


# ----------------------------------------------
# 主程序
# ----------------------------------------------

if __name__ == "__main__":
    dfs = {}
    funding_dfs = {}

    for sym in SYMBOLS:
        print(f"\n=== {sym}（抓取5m K線 + 資金費率歷史，只做一次）===")
        df5 = fetch_all_klines(sym, "5m", PAGES_5M)
        df5 = add_atr(df5)
        dfs[sym] = df5

        df_funding = fetch_funding_rate_history(sym)
        df_funding = compute_funding_zscore(df_funding)
        funding_dfs[sym] = df_funding

        n_valid_z = df_funding["z"].notna().sum()
        n_extreme = (df_funding["z"].abs() >= Z_THRESHOLD).sum()
        print(f"  [{sym}] 有效z-score筆數: {n_valid_z} / {len(df_funding)}，"
              f"其中極端(|z|>={Z_THRESHOLD})筆數: {n_extreme} "
              f"({n_extreme/max(n_valid_z,1)*100:.1f}%)")

    for tp_rr in TP_RR_CANDIDATES:
        run_label = f"_tprr{str(tp_rr).replace('.', 'p')}"
        print(f"\n\n{'#'*68}")
        print(f"# 回測 TP_RR = {tp_rr}")
        print(f"{'#'*68}")

        all_trades = []
        daily_records_by_symbol = {}

        for sym in SYMBOLS:
            print(f"  {sym} 回測中...")
            trades, daily_records = backtest_symbol(sym, dfs[sym], funding_dfs[sym], tp_rr=tp_rr)
            all_trades.extend(trades)
            daily_records_by_symbol[sym] = daily_records
            print(f"  {sym} 完成，{len(trades)} 筆交易")

        print_results(all_trades, dfs, daily_records_by_symbol, run_label=run_label, tp_rr=tp_rr)