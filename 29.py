"""
cross_sectional_funding_factor.py
====================
[定位] 橫截面多因子（Cross-Sectional）骨架 —— 基於 funding_reversal_backtest.py 改寫

跟你原本 funding_reversal_backtest.py 最根本的差異：
  原版是「單一幣種、時間序列」邏輯 —— 每個幣自己看自己的 funding z-score 夠不夠
  極端，觸發就進場。三個幣、4年資料只湊出40筆交易，樣本量不足以判斷有沒有edge。

  這支改成「同一時間點，橫向比較所有幣種」邏輯 —— 每次結算時間點，對所有幣種
  各算一個因子分數，排序後做多因子分數最高的一籃子、做空最低的一籃子。
  樣本量不再取決於「觸發次數」，而是「幣種數 × 結算次數」，同樣邏輯，
  樣本量可以從40筆直接放大到幾千筆等級。

[核心機制]
  1. 每個資金費率結算時間點（UTC 0/8/16），對 SYMBOLS 裡每個幣計算一個
     複合因子分數（目前預設：funding z-score 的負值 + 動量因子，見下方權重）。
     用「負值」是延續原假說：funding過熱(z高)代表擁擠，預期偏弱，所以
     factor分數跟z-score反向。
  2. 對所有有效分數的幣種做排名（rank），分數最高的前 TOP_N 做多，
     最低的後 TOP_N 做空 —— 這是市場中性(market-neutral)結構的核心：
     同時做多做空，對大盤方向的曝險互相抵消。
  3. 部位大小：每一腿(多籃子/空籃子)分配相同的總風險預算，
     籃子內部依因子分數強度加權，而不是原版的固定風險/筆。
  4. 持有到下一次結算時間點做 rebalance（滿倉換手），不是原版的
     SL/TP/逾時出場邏輯 —— 這是橫截面策略跟事件驅動策略在結構上的
     根本差異：橫截面策略關心的是「排序有沒有預測力」，不是「單筆
     交易的進出場時機」。

[⚠ 這是骨架，不是成品]
  - 目前只實作了兩個因子（funding z-score 反轉 + 動量），且用最簡單的
    等權重組合(FACTOR_WEIGHTS)。因子選擇、權重、TOP_N 全部沒調過參，
    第一輪先確認「這個框架能不能跑出有意義的樣本量」，不要一開始就
    急著多因子疊加或調權重 —— 跟你先前的教訓一樣：一次只驗證一個維度。
  - SYMBOLS 清單目前只放 3 個幣做示範（沿用原本三幣），你要做的第一件事
    是把這個清單擴充到 Bitunix 上有足夠歷史資料、流動性夠的所有 USDT
    永續合約 —— 樣本量的放大主要來自這裡，不是因子本身。
  - 沒有做「因子中性化」（比如剔除跟大盤beta的相關性、產業/類別中性化）
    這種更嚴謹的橫截面因子工程，先看陽春版本有沒有訊號，再考慮要不要加。
  - 假設所有幣種的資金費率結算時間點完全對齊在同一個 UTC 時間戳
    （0/8/16點）—— 這在 Bitunix 多數合約應該成立，但拿到新幣種時
    務必先驗證，不要假設。

[統計把關 —— 沿用你既有的機構級配套]
  equity curve、Sharpe/Sortino/Calmar/MaxDD、stationary bootstrap CI、
  IS/OOS 70/30切分。這套跟訊號來源無關，直接沿用不動。
"""

import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timezone

BASE_URL = "https://fapi.bitunix.com"

# ── 第一步要做的事：把這個清單擴充成 Bitunix 上流動性夠、歷史夠長的幣種 ──
# 樣本量 = 幣種數 × 結算次數，這個清單的長度直接決定了統計檢定力。
SYMBOLS = ["ETHUSDT", "BTCUSDT", "SOLUSDT"]  # TODO: 擴充到 20~50 個幣

PAGES_5M = 3000
FEE_RATE = 0.0006
SLIPPAGE_BPS = 2.0

FUNDING_ZSCORE_WINDOW = 90       # 沿用原版：約30天滾動窗口
MOMENTUM_LOOKBACK_SETTLEMENTS = 9  # 動量因子回看幾個結算週期（9×8h=3天）

# 因子權重：目前是最陽春的等權組合，之後要調權重就改這裡
FACTOR_WEIGHTS = {
    "funding_reversal": 0.5,   # -z_score（擁擠度越高，分數越低）
    "momentum": 0.5,           # 過去N期報酬率（動量越強，分數越高）
}

TOP_N = 1              # 每籃子選幾個幣做多/做空（幣種數少時先設1，擴充清單後再調大）
TOTAL_RISK_USDT = 0.24  # 整個組合（多腿+空腿合計）的目標風險預算
MIN_VALID_SYMBOLS = 2 * TOP_N  # 至少要有這麼多幣種同時有有效分數，才觸發這次rebalance

INITIAL_CAPITAL = 10000.0
IS_RATIO = 0.70
BOOTSTRAP_N = 2000
BOOTSTRAP_BLOCK = 5


# ----------------------------------------------
# 資料抓取（跟原版 funding_reversal_backtest.py 完全相同，直接沿用）
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
        all_data = df_c.to_dict("records")
        end_time = df_c["time"].min()
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
    df["markPrice"] = pd.to_numeric(df["markPrice"], errors="coerce")
    df = df.sort_values("fundingTime").reset_index(drop=True)
    df.to_csv(cache, index=False)
    print(f"\n  [{symbol} funding] 完成，共 {len(df)} 筆")
    return df


# ----------------------------------------------
# 因子計算（這裡是骨架的核心，之後要加新因子就在這裡擴充）
# ----------------------------------------------

def compute_funding_zscore(df_funding, window=FUNDING_ZSCORE_WINDOW):
    df = df_funding.copy()
    df["roll_mean"] = df["fundingRate"].rolling(window, min_periods=max(30, window // 3)).mean()
    df["roll_std"] = df["fundingRate"].rolling(window, min_periods=max(30, window // 3)).std()
    df["z"] = (df["fundingRate"] - df["roll_mean"]) / df["roll_std"]
    df.loc[df["roll_std"] <= 0, "z"] = np.nan
    return df


def compute_momentum(df_funding, lookback=MOMENTUM_LOOKBACK_SETTLEMENTS):
    """用 markPrice 在結算時間點上的報酬率當動量因子，避免額外對齊5分K的麻煩。"""
    df = df_funding.copy()
    df["momentum"] = df["markPrice"].pct_change(lookback)
    return df


def build_factor_table(funding_dfs):
    """
    把每個幣種的 funding_reversal 分數 + 動量分數，合併成一張
    「結算時間點 × 幣種」的因子表，方便後續逐時間點做橫截面排序。
    回傳: {fundingTime: {symbol: {"funding_reversal": x, "momentum": y}}}
    """
    per_time = {}
    for sym, df in funding_dfs.items():
        for row in df.itertuples():
            t = int(row.fundingTime)
            per_time.setdefault(t, {})[sym] = {
                "funding_reversal": -row.z if not pd.isna(row.z) else np.nan,  # 反向：擁擠度高分數低
                "momentum": row.momentum if not pd.isna(row.momentum) else np.nan,
            }
    return per_time


def cross_sectional_score(raw_scores_this_time):
    """
    把某個時間點、所有幣種的原始因子值，各自做橫截面z-score標準化
    （同一因子在不同幣種間比大小才有意義，不能直接把funding_reversal
    和momentum的原始尺度加在一起），再依 FACTOR_WEIGHTS 合成總分。
    """
    factor_names = list(FACTOR_WEIGHTS.keys())
    df = pd.DataFrame(raw_scores_this_time).T  # index=symbol, columns=factor_names
    combined = pd.Series(0.0, index=df.index)
    valid_mask = pd.Series(True, index=df.index)
    for f in factor_names:
        col = df[f]
        mu, sd = col.mean(), col.std()
        if pd.isna(sd) or sd == 0:
            valid_mask &= False
            continue
        z = (col - mu) / sd
        combined += FACTOR_WEIGHTS[f] * z.fillna(0)
        valid_mask &= col.notna()
    combined = combined[valid_mask]
    return combined.sort_values(ascending=False)


# ----------------------------------------------
# 組合層回測：每個結算時間點 rebalance 一次
# ----------------------------------------------

def apply_slippage(price, side, is_entry):
    s = SLIPPAGE_BPS / 10000.0
    if side == "BUY":
        return price * (1 + s) if is_entry else price * (1 - s)
    else:
        return price * (1 - s) if is_entry else price * (1 + s)


def run_cross_sectional_backtest(funding_dfs, top_n=TOP_N):
    factor_table = build_factor_table(funding_dfs)
    sorted_times = sorted(factor_table.keys())

    mark_price_lookup = {
        sym: {int(r.fundingTime): float(r.markPrice) for r in df.itertuples()}
        for sym, df in funding_dfs.items()
    }

    daily_records = []
    running_realized = 0.0
    current_positions = {}  # {symbol: {"side":..., "entry_price":..., "notional":...}}

    trades = []

    for idx, t in enumerate(sorted_times):
        raw_scores = factor_table[t]
        # 只保留這個時間點「所有因子都有值」的幣種
        clean_scores = {
            sym: v for sym, v in raw_scores.items()
            if all(not pd.isna(v[f]) for f in FACTOR_WEIGHTS)
        }
        if len(clean_scores) < MIN_VALID_SYMBOLS:
            continue

        ranked = cross_sectional_score(clean_scores)
        if len(ranked) < 2 * top_n:
            continue

        long_syms = list(ranked.index[:top_n])
        short_syms = list(ranked.index[-top_n:])

        # ── 結算已存在的部位（rebalance：全部平倉再開新倉，先求邏輯簡單） ──
        realized_this_round = 0.0
        for sym, pos in current_positions.items():
            if t not in mark_price_lookup.get(sym, {}):
                continue
            exit_raw = mark_price_lookup[sym][t]
            exit_fill = apply_slippage(exit_raw, pos["side"], is_entry=False)
            pnl_raw = (exit_fill - pos["entry_price"]) * pos["qty"]
            if pos["side"] == "SELL":
                pnl_raw = -pnl_raw
            fee = pos["notional"] * FEE_RATE
            net = pnl_raw - fee
            realized_this_round += net
            trades.append({
                "entry_ts": pos["entry_ts"], "exit_ts": t, "symbol": sym,
                "side": pos["side"], "net_pnl": net,
            })
        running_realized += realized_this_round
        current_positions = {}

        # ── 開新倉：多籃子 + 空籃子，各分一半風險預算，籃內等權 ──
        risk_per_leg = TOTAL_RISK_USDT / 2
        risk_per_symbol = risk_per_leg / top_n
        for sym in long_syms + short_syms:
            side = "BUY" if sym in long_syms else "SELL"
            entry_raw = mark_price_lookup[sym].get(t)
            if entry_raw is None:
                continue
            entry_fill = apply_slippage(entry_raw, side, is_entry=True)
            notional = risk_per_symbol * 20  # TODO: 用ATR或波動率動態決定槓桿/名目部位，這裡先用固定倍數示範
            qty = notional / entry_fill
            current_positions[sym] = {
                "side": side, "entry_price": entry_fill, "qty": qty,
                "notional": notional, "entry_ts": t,
            }

        dt = datetime.fromtimestamp(t / 1000, tz=timezone.utc)
        daily_records.append({"date": dt.date(), "realized_closed_cum": running_realized})

    return trades, daily_records


# ----------------------------------------------
# 績效統計（跟原版完全相同，通用邏輯直接沿用）
# ----------------------------------------------

def build_equity_curve(daily_records, initial_capital=INITIAL_CAPITAL):
    if not daily_records:
        return pd.Series(dtype=float)
    df = pd.DataFrame(daily_records).groupby("date").last().reset_index()
    s = pd.Series(df["realized_closed_cum"].values,
                  index=pd.to_datetime(df["date"]))
    equity = initial_capital + s
    return equity.sort_index()


def compute_perf_stats(equity, periods_per_year=365, initial_capital=INITIAL_CAPITAL):
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
    return stats, ret


def stationary_bootstrap_sharpe(ret, n_boot=BOOTSTRAP_N, avg_block=BOOTSTRAP_BLOCK,
                                 periods_per_year=365, seed=42):
    rng = np.random.default_rng(seed)
    r = ret.values
    n = len(r)
    if n < 10:
        return np.nan, np.nan
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
        return np.nan, np.nan
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return lo, hi


def print_results(trades, daily_records):
    if not trades:
        print("無交易 —— 可能是 MIN_VALID_SYMBOLS 太嚴格，或幣種數不夠形成排序")
        return
    df = pd.DataFrame(trades)
    total = len(df)
    wins = df[df["net_pnl"] > 0]
    total_net = df["net_pnl"].sum()

    print(f"\n{'='*68}")
    print(f"橫截面多因子回測  幣種數={len(SYMBOLS)}  TOP_N={TOP_N}  因子={list(FACTOR_WEIGHTS.keys())}")
    print(f"{'='*68}")
    print(f"總交易次數 : {total}")
    print(f"勝率        : {len(wins)/total*100:.1f}%")
    print(f"總淨盈虧        : {total_net:+.4f} USDT")
    print(f"平均每腿盈虧    : {df['net_pnl'].mean():+.4f} USDT")

    equity = build_equity_curve(daily_records)
    if len(equity) < 5:
        print("\n[!] rebalance次數過少，無法計算有意義的權益曲線指標")
        return

    stats, ret = compute_perf_stats(equity)
    print(f"\n區間: {equity.index.min().date()} ~ {equity.index.max().date()}（{len(equity)} 個rebalance點）")
    print(f"  Sharpe   : {stats.get('Sharpe', float('nan')):.3f}")
    print(f"  Sortino  : {stats.get('Sortino', float('nan')):.3f}")
    print(f"  Calmar   : {stats.get('Calmar', float('nan')):.3f}")
    print(f"  最大回撤 : {stats.get('MaxDD_pct', float('nan')):.2f}%")
    lo, hi = stationary_bootstrap_sharpe(ret)
    print(f"  Sharpe 95% CI : [{lo:.3f}, {hi:.3f}]")
    if not np.isnan(lo) and lo < 0 < hi:
        print(f"  → CI涵蓋0，樣本仍不足以證明Sharpe顯著大於0")

    split_idx = int(len(equity) * IS_RATIO)
    split_date = equity.index[split_idx]
    equity_is = equity.iloc[:split_idx + 1]
    equity_oos = equity.iloc[split_idx:]
    stats_is, _ = compute_perf_stats(equity_is)
    stats_oos, _ = compute_perf_stats(equity_oos)
    print(f"\nIS/OOS 切分點: {split_date.date()}")
    print(f"  IS  Sharpe: {stats_is.get('Sharpe', float('nan')):.3f}")
    print(f"  OOS Sharpe: {stats_oos.get('Sharpe', float('nan')):.3f}")

    equity.to_csv("equity_curve_cross_sectional.csv", header=["equity"])
    df.to_csv("trades_cross_sectional.csv", index=False, encoding="utf-8-sig")
    print(f"\n交易明細已存至 trades_cross_sectional.csv")
    print(f"權益曲線已存至 equity_curve_cross_sectional.csv")


# ----------------------------------------------
# 主程序
# ----------------------------------------------

if __name__ == "__main__":
    funding_dfs = {}

    for sym in SYMBOLS:
        print(f"\n=== {sym}（抓取資金費率歷史）===")
        df_funding = fetch_funding_rate_history(sym)
        df_funding = compute_funding_zscore(df_funding)
        df_funding = compute_momentum(df_funding)
        funding_dfs[sym] = df_funding

    trades, daily_records = run_cross_sectional_backtest(funding_dfs, top_n=TOP_N)
    print_results(trades, daily_records)