"""
decisive_direction_test_v15sd_v2_1h4h.py
====================
裁決性測試（1h/4h版）：核心邏輯與 decisive_direction_test_v15sd_v2.py 完全相同，
只改讀取來源改成 backtest_result_v15sd_merged_v2_1h4h.csv / cache_{SYMBOL}_1h.csv，
以及HORIZONS的單位從5m根數改成1h根數（4根=4小時，8根=8小時，24根=1天，48根=2天，
對應原版[5,10,20,40]根5m棒約略等比例放大到「小時尺度」的檢驗窗口）。

[本次新增 —— 機構等級配套]

  1. 直接讀取 entry_ts（原始ms int），不需要「猜年份」的字串解析邏輯，
     v15sd_merged_v2 已經把原始時間戳存進trade record了，這塊複雜度整個砍掉

  2. Block Bootstrap p 值（自相關校正）：
     原版的 binomtest 假設每筆判斷獨立同分布，但同一段趨勢裡連續產生的訊號
     彼此高度相關（同一個母事件被算成好幾筆獨立樣本），標準binomial test的
     p值在這種情況下會系統性低估，容易把「一段趨勢走對」誤判成「訊號普遍有效」。
     這裡額外用時間排序後的區塊重抽樣（moving block bootstrap）算一個校正後的p值，
     跟原本的binomial p值並列報告，兩者出現落差本身就是重要資訊。

  3. 樣本外驗證：依entry_ts中位數切前半/後半，分別跑檢定，
     看方向性edge是不是在整段樣本裡穩定存在，還是只集中在某一段時間

  4. 效果量報告：不只看p值，額外報告 accuracy - 50% 的絕對差距，
     避免「統計顯著但差距小到手續費一吃就沒了」的錯覺
     （這點呼應v15sd_merged_v2算出來的fee_pct_of_risk，可以互相對照）

用法：跟 backtest_result_v15sd_merged_v2.csv、cache_{SYMBOL}_5m.csv 放同資料夾執行：
    python decisive_direction_test_v15sd_v2.py

需要 scipy（pip install scipy）。
"""

import pandas as pd
import numpy as np
import bisect
from scipy import stats

TRADES_CSV = "backtest_result_v15sd_merged_v2_1h4h.csv"
ENTRY_TF   = "1h"
HORIZONS   = [4, 8, 24, 48]   # 小時尺度：4h/8h/1天/2天

BLOCK_BOOTSTRAP_N     = 3000
BLOCK_BOOTSTRAP_AVG_LEN = 8   # 平均區塊長度(筆數)，同一段趨勢常常連續產生數筆訊號


def load_cache(symbol, tf):
    path = f"cache_{symbol}_{tf}.csv"
    df = pd.read_csv(path)
    df["time"] = df["time"].astype(np.int64)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df


def forward_close(cache_times, cache_closes, entry_ms, n_bars_ahead):
    idx = bisect.bisect_left(cache_times, entry_ms)
    target = idx + n_bars_ahead
    if target >= len(cache_closes):
        return None
    return cache_closes[target]


def block_bootstrap_pvalue(hits_sequence, n_boot=BLOCK_BOOTSTRAP_N,
                            avg_block=BLOCK_BOOTSTRAP_AVG_LEN, seed=42):
    """
    hits_sequence: 依時間排序的 0/1 陣列（1=方向判斷正確）。
    虛無假設：真實命中率=0.5。用stationary bootstrap重抽樣出「在虛無假設下，
    保留原本自相關結構」的樣本，看實際觀測到的命中率落在這個分布的哪個百分位。
    做法：先把序列去均值後乘上校正因子，模擬「若真實邊際為0.5」的情境下，
    重抽樣得到的樣本平均命中率之分布，藉此得出比binomial test更保守（穩健）的p值。
    """
    x = np.asarray(hits_sequence, dtype=float)
    n = len(x)
    if n < 10:
        return np.nan, np.nan

    observed_rate = x.mean()

    # 在虛無假設(真實邊際=0.5)下，把觀測序列平移到均值0.5，保留其自相關結構，
    # 再重抽樣看「平移後的序列」用同樣的區塊抽樣法，抽出來的樣本均值離0.5有多遠，
    # 藉此估計「就算真實邊際是0.5，光靠自相關結構本身」能製造出多極端的觀測值。
    x_null = x - x.mean() + 0.5

    rng = np.random.default_rng(seed)
    p = 1.0 / avg_block
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        pos = rng.integers(0, n)
        for k in range(n):
            if k == 0 or rng.random() < p:
                pos = rng.integers(0, n)
            else:
                pos = (pos + 1) % n
            idx[k] = pos
        boot_means[b] = x_null[idx].mean()

    # 單邊p值：虛無假設下抽到 >= observed_rate 的比例
    p_value = (boot_means >= observed_rate).mean()
    return p_value, observed_rate


def run_side_test(sub, caches_getter, label):
    print(f"\n--- {label}（樣本數 {len(sub)}）---")
    rows = []
    sub_sorted = sub.sort_values("entry_ts")

    for n in HORIZONS:
        hits_seq = []
        for _, row in sub_sorted.iterrows():
            symbol = row["symbol"]
            times_list, closes_list = caches_getter(symbol)
            fwd = forward_close(times_list, closes_list, int(row["entry_ts"]), n)
            if fwd is None:
                continue
            entry = row["entry"]
            predicted_up = (row["side"] == "BUY")
            actual_up = fwd > entry
            hits_seq.append(int(predicted_up == actual_up))

        total = len(hits_seq)
        if total == 0:
            continue
        hits = sum(hits_seq)
        acc = hits / total

        binom_result = stats.binomtest(hits, total, 0.5, alternative="greater")
        p_binom = binom_result.pvalue
        ci = stats.binomtest(hits, total, 0.5, alternative="two-sided").proportion_ci(confidence_level=0.95)

        p_block, _ = block_bootstrap_pvalue(hits_seq)

        effect_size = acc - 0.5

        flag_binom = "★" if p_binom < 0.05 else " "
        flag_block = "★" if (not np.isnan(p_block) and p_block < 0.05) else " "

        print(f"  N={n:>3}根後  準確率={acc*100:5.1f}%  效果量(acc-50%)={effect_size*100:+5.1f}pp  "
              f"95%CI=[{ci.low*100:.1f}%,{ci.high*100:.1f}%]  "
              f"p(binomial)={p_binom:.4f}{flag_binom}  p(block-bootstrap,校正自相關)={p_block:.4f}{flag_block}  (n={total})")

        if p_binom < 0.05 and (np.isnan(p_block) or p_block >= 0.05):
            print(f"       ⚠ binomial顯著但block-bootstrap不顯著 → 很可能是自相關訊號"
                  f"（同一段趨勢連續出訊號）造成的假陽性，不是真實方向性edge")

        rows.append({
            "label": label, "horizon_bars": n, "accuracy": acc,
            "effect_size_pp": effect_size * 100,
            "p_binomial": p_binom, "p_block_bootstrap": p_block, "n": total,
        })
    return rows


def run_test():
    df = pd.read_csv(TRADES_CSV)
    if "entry_ts" not in df.columns:
        raise ValueError(
            "找不到 entry_ts 欄位。請確認 backtest_result_v15sd_merged_v2_1h4h.csv 是用 "
            "bitunix_backtest_v15sd_merged_v2.py（新版）產生的，不是舊版v15sd_merged的輸出。"
        )
    df["entry_ts"] = df["entry_ts"].astype(np.int64)

    caches = {}
    def get_cache(symbol):
        if symbol not in caches:
            c = load_cache(symbol, ENTRY_TF)
            caches[symbol] = (c["time"].tolist(), c["close"].tolist())
        return caches[symbol]

    print("=" * 88)
    print("裁決性測試 v2：EMA/MACD方向判斷是否顯著優於隨機丟硬幣（含自相關校正）")
    print("=" * 88)

    all_rows = []

    # ── 全樣本 ──
    print("\n########## 全樣本 (Full Sample) ##########")
    for side_label, sub in [("BUY", df[df["side"] == "BUY"]), ("SELL", df[df["side"] == "SELL"])]:
        if len(sub) == 0:
            continue
        all_rows.extend(run_side_test(sub, get_cache, f"全樣本 side={side_label}"))

    # ── [新增] 樣本外驗證：依entry_ts中位數切前後兩半 ──
    median_ts = df["entry_ts"].median()
    df_first  = df[df["entry_ts"] < median_ts]
    df_second = df[df["entry_ts"] >= median_ts]

    print(f"\n########## 時間切半驗證（切點: entry_ts中位數）##########")
    print(f"前半樣本數={len(df_first)}  後半樣本數={len(df_second)}")
    for half_label, half_df in [("前半段", df_first), ("後半段", df_second)]:
        print(f"\n--- {half_label} ---")
        for side_label, sub in [("BUY", half_df[half_df["side"] == "BUY"]),
                                 ("SELL", half_df[half_df["side"] == "SELL"])]:
            if len(sub) == 0:
                continue
            all_rows.extend(run_side_test(sub, get_cache, f"{half_label} side={side_label}"))

    print("\n" + "=" * 88)
    print("解讀方式：")
    print("- p(binomial) < 0.05 是原始檢定，假設每筆訊號獨立，容易被自相關訊號製造假陽性")
    print("- p(block-bootstrap) < 0.05 才是校正自相關後的結論，較保守但較可信")
    print("- 兩者同時顯著，才建議當作真實edge往下走；只有binomial顯著要高度懷疑")
    print("- 前後半段若結論不一致（例如只有前半顯著），代表edge不穩定，"
          "可能只是特定行情窗口的產物，不是策略普遍有效")
    print("=" * 88)

    if all_rows:
        result_df = pd.DataFrame(all_rows)

        k = len(result_df)
        bonferroni_alpha = 0.05 / k
        result_df["sig_binomial_raw"] = result_df["p_binomial"] < 0.05
        result_df["sig_binomial_bonferroni"] = result_df["p_binomial"] < bonferroni_alpha
        result_df["sig_block_bootstrap_raw"] = result_df["p_block_bootstrap"] < 0.05

        print(f"\n[多重比較校正] 本次共跑了 {k} 個獨立檢定（含全樣本+切半驗證）")
        print(f"  Bonferroni校正後顯著門檻：p < {bonferroni_alpha:.5f}")
        print(f"  binomial校正前顯著筆數  : {result_df['sig_binomial_raw'].sum()} / {k}")
        print(f"  binomial校正後顯著筆數  : {result_df['sig_binomial_bonferroni'].sum()} / {k}")
        print(f"  block-bootstrap顯著筆數 : {result_df['sig_block_bootstrap_raw'].sum()} / {k}"
              f"（未做多重比較校正，僅供參考）")

        result_df.to_csv("decisive_direction_test_v15sd_v2_1h4h_result.csv", index=False)
        print("\n結果已存至 decisive_direction_test_v15sd_v2_1h4h_result.csv")


if __name__ == "__main__":
    run_test()