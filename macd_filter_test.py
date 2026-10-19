"""
macd_filter_test.py
====================
快速驗證：在正式把MACD整合進回測引擎之前，先問一個便宜的問題——
「進場當下MACD方向，是否能把BOS/CHoCH訊號的母體，篩出一個有方向性edge的子集合？」

做法：
1. 讀取 backtest_result_smc_v1.csv 裡每一筆已成交的交易
2. 用 cache_{symbol}_{tf}.csv 的K線資料，算出進場當下（結構所在時框，5m或15m）
   最近一根已收線K棒的 MACD 狀態（MACD線 vs 訊號線）
3. 把交易分成「MACD方向與交易方向一致」vs「不一致」兩組
4. 對兩組分別做：
   (a) 實際勝率 + 卡方檢定，看兩組勝率是否顯著不同
   (b) 沿用裁決性方向測試的邏輯：往後看N根K棒，方向準確率是否顯著優於50%
5. 多重比較一律套用 Bonferroni 校正

如果「MACD一致組」在校正後仍顯著優於「不一致組」或優於50%，才值得花時間
把MACD濾網整合進正式回測引擎；否則不用做這個工程。

用法：跟 backtest_result_smc_v1.csv、cache_{SYMBOL}_{5m/15m}.csv 放同資料夾：
    python macd_filter_test.py
需要 scipy（pip install scipy）。
"""

import pandas as pd
import numpy as np
import bisect
import time as time_module
from datetime import datetime
from scipy import stats

TRADES_CSV = "backtest_result_smc_v1.csv"
HORIZONS = [10, 20, 40]  # 只挑3個horizon，避免多重比較檢定數量爆炸

MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9


def load_cache(symbol, tf):
    path = f"cache_{symbol}_{tf}.csv"
    df = pd.read_csv(path)
    df["time"] = df["time"].astype(np.int64)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)

    # 計算標準 MACD：EMA12 - EMA26，訊號線 = MACD的EMA9
    ema_fast = df["close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def time_str_to_ms(time_str):
    dt = datetime.strptime(time_str, "%Y/%m/%d %H:%M")
    return int(time_module.mktime(dt.timetuple()) * 1000)


def get_macd_state_at(cache_df, entry_ms):
    """回傳進場時間點「最近一根已收線」K棒的MACD histogram方向。
    bullish(hist>0) / bearish(hist<0) / None(資料不足或剛好等於0)"""
    times = cache_df["time"].values
    idx = bisect.bisect_right(times, entry_ms) - 1
    if idx < 0:
        return None, None
    hist = cache_df["macd_hist"].iloc[idx]
    if pd.isna(hist) or hist == 0:
        return None, idx
    return ("bullish" if hist > 0 else "bearish"), idx


def forward_close(cache_df, idx, n_bars_ahead):
    target = idx + n_bars_ahead
    if target >= len(cache_df):
        return None
    return cache_df["close"].iloc[target]


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    adj = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    return ((centre - adj) / denom, (centre + adj) / denom)


def run():
    df = pd.read_csv(TRADES_CSV)
    df["entry_ms"] = df["Time"].apply(time_str_to_ms)

    caches = {}
    def get_cache(symbol, tf):
        key = (symbol, tf)
        if key not in caches:
            caches[key] = load_cache(symbol, tf)
        return caches[key]

    # ── 幫每筆交易標記 MACD 狀態、是否與交易方向一致 ──
    macd_states, macd_idxs, agree_flags = [], [], []
    for _, row in df.iterrows():
        cache_df = get_cache(row["Symbol"], row["structure_tf"])
        state, idx = get_macd_state_at(cache_df, row["entry_ms"])
        macd_states.append(state)
        macd_idxs.append(idx)
        if state is None:
            agree_flags.append(None)
        else:
            predicted_bullish = (row["Direction"] == "long")
            macd_bullish = (state == "bullish")
            agree_flags.append(predicted_bullish == macd_bullish)

    df["macd_state"] = macd_states
    df["macd_idx"] = macd_idxs
    df["macd_agree"] = agree_flags

    valid = df[df["macd_agree"].notna()].copy()
    print("=" * 78)
    print(f"MACD方向濾網快速驗證（{len(valid)}/{len(df)} 筆有效，其餘因資料不足或MACD剛好=0被排除）")
    print("=" * 78)

    # ── (a) 實際勝率：一致組 vs 不一致組，卡方檢定 ──
    print("\n[A] 依「MACD方向是否與交易方向一致」拆解實際勝率")
    rows = []
    for agree in [True, False]:
        sub = valid[valid["macd_agree"] == agree]
        n = len(sub)
        if n == 0:
            continue
        wins = (sub["Result"] > 0).sum()
        wr = wins / n * 100
        lo, hi = wilson_ci(wins, n)
        avg_r = sub["R_Multiple"].mean()
        label = "一致(MACD支持進場方向)" if agree else "不一致(MACD反對進場方向)"
        print(f"  {label:<24} 筆數={n:>5}  勝率={wr:5.1f}%  "
              f"95%CI=[{lo*100:.1f}%,{hi*100:.1f}%]  平均R倍數={avg_r:+.3f}")
        rows.append((agree, wins, n - wins))

    if len(rows) == 2:
        contingency = np.array([[r[1], r[2]] for r in rows])
        chi2, p, _, _ = stats.chi2_contingency(contingency)
        print(f"\n  卡方檢定（一致組 vs 不一致組勝率是否顯著不同）: chi2={chi2:.3f}  p值={p:.4f}")

    # ── (b) 裁決性方向測試：只在「一致組」裡看方向準確率是否優於50% ──
    print("\n[B] 「MACD與進場方向一致」的交易，往後看N根K棒的方向準確率")
    agree_sub = valid[valid["macd_agree"] == True]
    test_results = []
    for n_bars in HORIZONS:
        hits, total = 0, 0
        for _, row in agree_sub.iterrows():
            cache_df = get_cache(row["Symbol"], row["structure_tf"])
            fwd = forward_close(cache_df, int(row["macd_idx"]), n_bars)
            if fwd is None:
                continue
            predicted_up = (row["Direction"] == "long")
            actual_up = fwd > row["Entry"]
            hits += int(predicted_up == actual_up)
            total += 1
        if total == 0:
            continue
        acc = hits / total
        result = stats.binomtest(hits, total, 0.5, alternative="greater")
        ci = stats.binomtest(hits, total, 0.5, alternative="two-sided").proportion_ci(0.95)
        flag = "★ 顯著" if result.pvalue < 0.05 else "  不顯著"
        print(f"  N={n_bars:>3}根後  準確率={acc*100:5.1f}%  "
              f"95%CI=[{ci.low*100:.1f}%,{ci.high*100:.1f}%]  p={result.pvalue:.4f}  {flag}  (n={total})")
        test_results.append(result.pvalue)

    if test_results:
        k = len(test_results)
        bonf = 0.05 / k
        n_sig_raw = sum(p < 0.05 for p in test_results)
        n_sig_corrected = sum(p < bonf for p in test_results)
        print(f"\n  [多重比較校正] 共{k}個檢定，Bonferroni門檻 p<{bonf:.4f}")
        print(f"  校正前顯著: {n_sig_raw}/{k}  校正後顯著: {n_sig_corrected}/{k}")

    print("\n" + "=" * 78)
    print("判斷標準：")
    print("- [A]的卡方p值 < 0.05：代表MACD一致與否，勝率有統計上顯著差異，")
    print("  這是「值得整合」的訊號")
    print("- [B]校正後仍有顯著結果：代表在MACD支持的方向上，價格確實更可能")
    print("  往預期方向走，不只是勝率統計上的巧合")
    print("- 兩者都不顯著：代表MACD疊加大概率篩不出有效子集，不建議投入工程")
    print("  時間把它整合進正式回測引擎")
    print("=" * 78)

    valid.to_csv("macd_filter_test_result.csv", index=False)
    print("\n逐筆結果已存至 macd_filter_test_result.csv")


if __name__ == "__main__":
    run()