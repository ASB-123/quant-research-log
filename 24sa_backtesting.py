"""
bitunix_funding_cash_carry_backtest.py
====================
[策略] 現貨+永續 Cash-and-Carry（資金費率套利，Delta中性）

跟 funding_reversal 版本是同一份資金費率資料，但玩法完全不同性質：
  反轉版賭「方向」（極端擁擠時反向下注，本質仍是方向性交易，只是訊號源換了）。
  這一版不賭方向：同時買進等量現貨 + 放空等量永續，兩腿的價格曝險理論上互相
  抵銷，唯一的獲利來源是「資金費率」本身——當費率夠正、且正到能覆蓋開平倉
  四筆手續費與basis(現貨-永續價差)波動風險時，才進場吃這筆費率。

[⚠ 重大工程限制 —— 誠實揭露，這點必須先讀]
  1. 現貨API的base URL我沒辦法實測確認。期貨用 fapi.bitunix.com 是文件明確
     寫的、也是你原本程式碼實際在用的，這點沒問題。但現貨的REST base domain
     官方文件（api-doc.bitunix.com）並未直接寫明，只有WebSocket寫了
     wss://openapi.bitunix.com。本程式碼用 SPOT_BASE_URL_CANDIDATES 做自動
     偵測（依序嘗試，抓第一個回應正常的），第一次跑的時候務必先看
     「[偵測]」那行印出來的實際base URL是不是合理，不要盲跑。如果全部
     偵測失敗，需要你自己去 Bitunix API 後台或文件確認正確網域，改
     SPOT_BASE_URL_CANDIDATES 常數。
  2. 現貨手續費假設跟期貨一樣是0.06%——這只是假設，不是查證過的真實費率。
     Bitunix現貨費率可能不同（甚至可能有VIP分級），套利策略對手續費極度
     敏感，正式使用前務必去帳戶頁面確認你自己的實際現貨/合約費率並修改
     FEE_RATE_SPOT / FEE_RATE_PERP。
  3. 沒有考慮永續合約保證金/爆倉風險。真實的cash-and-carry空頭腿是有槓桿
     部位，如果basis劇烈反向擴大，可能觸及維持保證金而被強平，這裡完全
     沒有模擬保證金機制——這是本回測「風險被低估」的最大來源，正式上線
     前這塊一定要另外評估（用多低的槓桿、多少保證金緩衝）。
  4. 出場判斷只在每次資金費結算時間點檢查（8小時一次），不是逐筆K線監控
     basis stop，所以如果basis在兩次結算之間劇烈反向又拉回，這裡看不到。

[統計配套] equity curve、Sharpe/Sortino/Calmar、bootstrap CI、IS/OOS。但這裡的Beta/Alpha基準改成「現貨本身」而不是
三幣籃子——因為這是delta中性策略，比較對象應該是「如果你單純持有現貨」，
而不是「三幣種一起做多」的市場基準。
"""

import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timezone

FUTURES_BASE_URL = "https://fapi.bitunix.com"

# [⚠ 需要驗證] 依序嘗試，抓第一個回應正常的當作現貨API base URL
SPOT_BASE_URL_CANDIDATES = [
    "https://openapi.bitunix.com",
    "https://api.bitunix.com",
    "https://fapi.bitunix.com",
]

SYMBOLS = ["ETHUSDT", "BTCUSDT", "SOLUSDT"]

FEE_RATE_SPOT = 0.0010   # 現貨手續費
FEE_RATE_PERP = 0.0006   # 永續手續費，沿用你原本程式碼的假設

NOTIONAL_USDT = 1000.0   # 每次進場的每腿名目本金（現貨腿以此換算數量，永續腿數量與現貨腿相同單位數）

# ── 進出場門檻（年化資金費率，用來跟持有成本比較）──
FUNDING_SMOOTH_SETTLEMENTS = 3      # 平滑窗口：近3次結算(=24h)的平均費率
ENTRY_ANNUALIZED_THRESHOLD = 0.15   # 年化資金費率需超過15%才考慮進場（涵蓋開平倉4次手續費+合理利潤空間）
EXIT_ANNUALIZED_THRESHOLD  = 0.03   # 平滑後年化費率跌破3%就出場（費率吸引力消失）
BASIS_STOP_PCT             = 0.015  # basis相對現貨價格的不利變動超過1.5%，視為dislocation，緊急出場
MAX_HOLD_DAYS               = 90    # 安全閥：最長持倉90天，避免回測中出現不合理的無限持倉

SETTLEMENTS_PER_YEAR = 365 * 3      # 每8小時一次，一年3*365次

INITIAL_CAPITAL = 10000.0
IS_RATIO = 0.70
BOOTSTRAP_N = 2000
BOOTSTRAP_BLOCK = 5

# ----------------------------------------------
# 現貨API base URL 自動偵測
# ----------------------------------------------

def detect_spot_base_url():
    test_params = {"symbol": "BTCUSDT", "interval": "60", "limit": "5"}
    for base in SPOT_BASE_URL_CANDIDATES:
        try:
            res = requests.get(f"{base}/api/spot/v1/market/kline/history", params=test_params, timeout=10)
            data = res.json()
            if isinstance(data, dict) and data.get("code") in (0, "0") and data.get("data"):
                print(f"  [偵測] 現貨API base URL確認可用: {base}")
                print(f"  [偵測] 樣本回傳: {str(data['data'])[:200]}")
                return base
        except Exception as e:
            print(f"  [偵測] {base} 失敗：{e}")
            continue
    raise RuntimeError(
        "[!] 無法自動偵測現貨API的base URL。請至 https://api-doc.bitunix.com 確認"
        "正確的REST base domain，手動設定 SPOT_BASE_URL_CANDIDATES 後重跑。"
        "（也可能是本機網路無法連到bitunix.com，先用curl手動測一次連線）"
    )


# ----------------------------------------------
# 資金費率歷史（與 funding_reversal_backtest.py 完全相同邏輯）
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
                    f"{FUTURES_BASE_URL}/api/v1/futures/market/get_funding_rate_history",
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
# 現貨小時K線（用來對齊每次資金費結算時間點的現貨價格）
# ----------------------------------------------

def _parse_ts_flexible(ts_val):
    """[⚠] 現貨API回傳的ts格式官方文件寫ISO8601，但history端點沒明確示範，
    這裡做寬鬆解析：能轉int（epoch ms或s）就轉，不行就當ISO8601字串解析。
    第一次跑一定要對照 print 出來的樣本資料人工核對一次。"""
    try:
        v = int(ts_val)
        if v > 10**12:      # ms
            return v
        elif v > 10**9:     # s
            return v * 1000
        else:
            return v
    except (ValueError, TypeError):
        return int(pd.to_datetime(ts_val, utc=True).timestamp() * 1000)


def fetch_spot_hourly_klines(symbol, base_url, max_pages=200):
    cache = f"cache_spot_hourly_{symbol}.csv"
    if os.path.exists(cache):
        df_c = pd.read_csv(cache)
        all_data = df_c.to_dict("records")
        end_time_s = int(df_c["time_ms"].min() / 1000)
        print(f"  [{symbol} spot-1h] 快取 {len(all_data)} 根，從 {end_time_s}(s) 往回繼續")
    else:
        all_data, end_time_s = [], None

    for p in range(max_pages):
        params = {"symbol": symbol, "interval": "60", "limit": 500}
        if end_time_s:
            params["endTime"] = end_time_s
        try:
            res = requests.get(f"{base_url}/api/spot/v1/market/kline/history", params=params, timeout=30)
            data = res.json().get("data", [])
        except Exception as e:
            print(f"\n  [{symbol} spot-1h] 請求失敗：{e}")
            time.sleep(5)
            continue
        if not data:
            break
        parsed = []
        for row in data:
            t_ms = _parse_ts_flexible(row.get("ts"))
            parsed.append({
                "time_ms": t_ms,
                "open": float(row["open"]), "high": float(row["high"]),
                "low": float(row["low"]), "close": float(row["close"]),
            })
        all_data.extend(parsed)
        oldest_s = min(r["time_ms"] for r in parsed) // 1000
        if end_time_s is not None and oldest_s >= end_time_s:
            break
        end_time_s = oldest_s
        print(f"  [{symbol} spot-1h] 第{p+1}頁，累積 {len(all_data)} 根", end="\r")
        time.sleep(0.15)

    df = pd.DataFrame(all_data).drop_duplicates("time_ms").sort_values("time_ms").reset_index(drop=True)
    df.to_csv(cache, index=False)
    print(f"\n  [{symbol} spot-1h] 完成，共 {len(df)} 根")
    if len(df):
        print(f"  [{symbol} spot-1h] 樣本前3筆：\n{df.head(3)}")
    return df


# ----------------------------------------------
# 合併資金費率結算時間點 + 對應現貨價格
# ----------------------------------------------

def build_settlement_table(df_funding, df_spot_hourly):
    """把每筆資金費結算時間點對應到當下的現貨小時K收盤價（用<=結算時間的最近一根）。"""
    df_spot_hourly = df_spot_hourly.sort_values("time_ms")
    merged = pd.merge_asof(
        df_funding.sort_values("fundingTime"),
        df_spot_hourly.rename(columns={"close": "spot_price"})[["time_ms", "spot_price"]],
        left_on="fundingTime", right_on="time_ms", direction="backward",
        tolerance=3 * 3600 * 1000,  # 最多容忍3小時對不齊
    )
    merged = merged.dropna(subset=["spot_price"]).reset_index(drop=True)
    merged["roll_funding"] = merged["fundingRate"].rolling(
        FUNDING_SMOOTH_SETTLEMENTS, min_periods=FUNDING_SMOOTH_SETTLEMENTS).mean()
    merged["ann_funding"] = merged["roll_funding"] * SETTLEMENTS_PER_YEAR
    return merged


# ----------------------------------------------
# 單幣回測（只在結算時間點上動作，不需要5m K線）
# ----------------------------------------------

def backtest_symbol_carry(symbol, settle_df):
    trades = []
    daily_records = []
    running_realized = 0.0
    position = None  # None 或 dict

    settle_df = settle_df.reset_index(drop=True)

    for idx, row in settle_df.iterrows():
        t = int(row["fundingTime"])
        perp_price = float(row["markPrice"])
        spot_price = float(row["spot_price"])
        fr = float(row["fundingRate"])
        ann_funding = row["ann_funding"]

        row_date = datetime.fromtimestamp(t / 1000, tz=timezone.utc).date()

        # ── 若有持倉：先套用這次結算的資金費（我們空永續，費率為正時收錢） ──
        if position is not None:
            funding_pnl = position["qty"] * perp_price * fr
            position["realized_pnl"] += funding_pnl
            position["funding_received"] += funding_pnl
            position["settlements_held"] += 1

            basis_now = (perp_price - spot_price) / spot_price
            basis_entry = position["entry_basis"]
            basis_move_adverse = basis_now - basis_entry  # 對我們不利的方向：perp相對spot漲多（我們空perp）

            days_held = (t - position["entry_time"]) / (1000 * 3600 * 24)
            exit_reason = None
            if not pd.isna(ann_funding) and ann_funding < EXIT_ANNUALIZED_THRESHOLD:
                exit_reason = "FUNDING_DECAY"
            elif basis_move_adverse > BASIS_STOP_PCT:
                exit_reason = "BASIS_STOP"
            elif days_held >= MAX_HOLD_DAYS:
                exit_reason = "MAX_HOLD"

            if exit_reason:
                spot_pnl = (spot_price - position["entry_spot"]) * position["qty"]
                perp_pnl = (position["entry_perp"] - perp_price) * position["qty"]
                close_fee = (spot_price * position["qty"]) * FEE_RATE_SPOT + \
                            (perp_price * position["qty"]) * FEE_RATE_PERP
                net = spot_pnl + perp_pnl - close_fee + position["realized_pnl"]
                running_realized += net

                trades.append({
                    "symbol": symbol, "entry_time": position["entry_time_str"],
                    "exit_time": datetime.fromtimestamp(t / 1000).strftime("%m-%d %H:%M"),
                    "days_held": round(days_held, 2), "settlements_held": position["settlements_held"],
                    "entry_spot": round(position["entry_spot"], 4), "exit_spot": round(spot_price, 4),
                    "entry_perp": round(position["entry_perp"], 4), "exit_perp": round(perp_price, 4),
                    "entry_basis_pct": round(basis_entry * 100, 4), "exit_basis_pct": round(basis_now * 100, 4),
                    "funding_received": round(position["funding_received"], 4),
                    "basis_pnl": round(spot_pnl + perp_pnl, 4),
                    "fee_total": round(close_fee + position["open_fee"], 4),
                    "net_pnl": round(net, 4), "exit_reason": exit_reason,
                })
                position = None

        # ── 無持倉時，檢查是否符合進場門檻 ──
        if position is None and not pd.isna(ann_funding) and ann_funding > ENTRY_ANNUALIZED_THRESHOLD:
            qty = NOTIONAL_USDT / spot_price
            open_fee = (spot_price * qty) * FEE_RATE_SPOT + (perp_price * qty) * FEE_RATE_PERP
            position = {
                "entry_time": t, "entry_time_str": datetime.fromtimestamp(t / 1000).strftime("%m-%d %H:%M"),
                "entry_spot": spot_price, "entry_perp": perp_price, "qty": qty,
                "entry_basis": (perp_price - spot_price) / spot_price,
                "realized_pnl": -open_fee, "open_fee": open_fee,
                "funding_received": 0.0, "settlements_held": 0,
            }

        # ── 逐日 mark-to-market 快照 ──
        if idx == len(settle_df) - 1 or \
           datetime.fromtimestamp(int(settle_df.iloc[idx + 1]["fundingTime"]) / 1000, tz=timezone.utc).date() != row_date:
            unreal = 0.0
            if position is not None:
                spot_pnl = (spot_price - position["entry_spot"]) * position["qty"]
                perp_pnl = (position["entry_perp"] - perp_price) * position["qty"]
                unreal = spot_pnl + perp_pnl + position["realized_pnl"]
            daily_records.append({"date": row_date, "unrealized_incl_partial": unreal,
                                   "realized_closed_cum": running_realized})

    return trades, daily_records


# ----------------------------------------------
# 績效工具（與 funding_reversal_backtest.py 相同邏輯，直接沿用）
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


def compute_beta_alpha_vs_spot(strat_ret, spot_close_caches, periods_per_year=365):
    """基準改成現貨本身（等權平均），因為這是delta中性策略，該比較的是
    「不做這個策略、單純持有現貨」的機會成本，而不是三幣一起做多的市場基準。"""
    daily_closes = {}
    for sym, df in spot_close_caches.items():
        s = df.copy()
        s["date"] = pd.to_datetime(s["time_ms"], unit="ms").dt.date
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


def _print_stats_block(label, equity, trades_df, spot_close_caches):
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
    beta, alpha, n_days = compute_beta_alpha_vs_spot(ret, spot_close_caches)
    print(f"  Beta (vs現貨等權basket) : {beta:.3f}  （對齊天數={n_days}，理論上delta中性策略beta應接近0）")
    print(f"  Alpha (年化)            : {alpha:+.4f}")


def print_results(all_trades, spot_dfs, daily_records_by_symbol, symbols):
    if not all_trades:
        print("無任何進場（可能是ENTRY_ANNUALIZED_THRESHOLD太嚴格，或這段期間資金費率長期不夠正）")
        return
    df_all = pd.DataFrame(all_trades)
    total = len(df_all)
    wins = df_all[df_all["net_pnl"] > 0]
    losses = df_all[df_all["net_pnl"] <= 0]
    total_net = df_all["net_pnl"].sum()
    total_funding = df_all["funding_received"].sum()
    total_basis_pnl = df_all["basis_pnl"].sum()
    total_fee = df_all["fee_total"].sum()

    print(f"\n{'='*68}")
    print(f"{' / '.join(symbols)}  現貨+永續 Cash-and-Carry [funding_cash_carry]")
    print(f"進場門檻：年化資金費率 > {ENTRY_ANNUALIZED_THRESHOLD*100:.0f}%  "
          f"出場門檻：< {EXIT_ANNUALIZED_THRESHOLD*100:.0f}%  Basis stop：{BASIS_STOP_PCT*100:.1f}%")
    print(f"每腿名目本金：{NOTIONAL_USDT} USDT  手續費：現貨{FEE_RATE_SPOT*100}% / 永續{FEE_RATE_PERP*100}%")
    print(f"{'='*68}")
    for sym in symbols:
        n = (df_all["symbol"] == sym).sum()
        sym_net = df_all[df_all["symbol"] == sym]["net_pnl"].sum()
        print(f"  {sym:<10} 進出場 {n:>4} 次  淨盈虧 {sym_net:>+.4f} USDT")
    print(f"{'─'*68}")
    print(f"總進出場次數: {total}")
    print(f"勝率        : {len(wins)/total*100:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"總淨盈虧    : {total_net:+.4f} USDT")
    print(f"  其中資金費收入  : {total_funding:+.4f} USDT")
    print(f"  其中basis損益   : {total_basis_pnl:+.4f} USDT   （理論上應接近0，越大代表basis風險越不可忽略）")
    print(f"  其中手續費成本  : -{total_fee:.4f} USDT")
    print(f"平均持倉天數: {df_all['days_held'].mean():.1f} 天")
    exit_dist = df_all["exit_reason"].value_counts().to_dict()
    print(f"出場原因分佈: {exit_dist}")

    equity_full = build_portfolio_equity(daily_records_by_symbol)
    if len(equity_full) < 5:
        print("\n[!] 天數過少，無法計算有意義的權益曲線指標")
        return
    spot_close_caches = {sym: spot_dfs[sym] for sym in symbols}

    print(f"\n{'='*68}")
    print("風險調整指標（基於逐日Mark-to-Market權益曲線）")
    print(f"{'='*68}")
    _print_stats_block("全樣本 (Full Sample)", equity_full, df_all, spot_close_caches)

    split_idx = int(len(equity_full) * IS_RATIO)
    split_date = equity_full.index[split_idx]
    equity_is = equity_full.iloc[:split_idx + 1]
    equity_oos = equity_full.iloc[split_idx:]

    print(f"\n{'='*68}")
    print(f"樣本外驗證（切分點: {split_date.date()}）")
    print(f"{'='*68}")
    _print_stats_block("樣本內 In-Sample", equity_is, None, spot_close_caches)
    _print_stats_block("樣本外 Out-of-Sample", equity_oos, None, spot_close_caches)

    equity_full.to_csv("equity_curve_funding_cash_carry.csv", header=["equity"])
    df_all.to_csv("backtest_result_funding_cash_carry.csv", index=False, encoding="utf-8-sig")
    print(f"\n交易明細已存至 backtest_result_funding_cash_carry.csv")
    print(f"權益曲線已存至 equity_curve_funding_cash_carry.csv")


# ----------------------------------------------
# 主程序
# ----------------------------------------------

if __name__ == "__main__":
    print("=== 偵測現貨API base URL ===")
    spot_base = detect_spot_base_url()

    settle_tables = {}
    spot_dfs = {}

    for sym in SYMBOLS:
        print(f"\n=== {sym}（抓取資金費率歷史 + 現貨小時K線）===")
        df_funding = fetch_funding_rate_history(sym)
        df_spot = fetch_spot_hourly_klines(sym, spot_base)
        spot_dfs[sym] = df_spot

        settle_df = build_settlement_table(df_funding, df_spot)
        settle_tables[sym] = settle_df
        n_eligible = (settle_df["ann_funding"] > ENTRY_ANNUALIZED_THRESHOLD).sum()
        print(f"  [{sym}] 對齊後結算筆數: {len(settle_df)}，"
              f"其中年化費率>{ENTRY_ANNUALIZED_THRESHOLD*100:.0f}%的筆數: {n_eligible} "
              f"({n_eligible/max(len(settle_df),1)*100:.1f}%)")

    all_trades = []
    daily_records_by_symbol = {}
    for sym in SYMBOLS:
        print(f"  {sym} 回測中...")
        trades, daily_records = backtest_symbol_carry(sym, settle_tables[sym])
        all_trades.extend(trades)
        daily_records_by_symbol[sym] = daily_records
        print(f"  {sym} 完成，{len(trades)} 次進出場")

    print_results(all_trades, spot_dfs, daily_records_by_symbol, SYMBOLS)