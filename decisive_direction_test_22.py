"""
decisive_direction_test.py
====================
裁決性測試：完全丟開手續費、滑價、停損停利設計，只驗證一件事——
BOS/CHoCH 確認後，價格往判斷方向走的機率，是否顯著高於50%（丟硬幣基準）。

如果連這個都打不贏隨機，後面所有成本模型/停利停損邏輯的精修都沒有意義，
因為問題不在管理層，在訊號本身沒有方向性預測力。

用法：跟 backtest_result_smc_v1.csv、cache_{SYMBOL}_{5m/15m}.csv 放同資料夾執行：
    python decisive_direction_test.py

需要 scipy（pip install scipy）。
"""

import pandas as pd
import numpy as np
import bisect
import time as time_module
from datetime import datetime
from scipy import stats

TRADES_CSV = "backtest_result_smc_v1.csv"
HORIZONS_5M  = [5, 10, 20, 40]     # 5m情境下往後看的K棒數（25分鐘~200分鐘）
HORIZONS_15M = [5, 10, 20, 40]     # 15m情境下往後看的K棒數（75分鐘~600分鐘）


def load_cache(symbol, tf):
    path = f"cache_{symbol}_{tf}.csv"
    df = pd.read_csv(path)
    df["time"] = df["time"].astype(np.int64)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df


def time_str_to_ms(time_str):
    """反推回 bitunix_backtest_smc_v1.py 裡 datetime.fromtimestamp() 用掉的本地時區，
    盡量還原成當初存檔用的 entry_time（ms）。存在時區誤差風險，但用於往後找N根K棒
    的相對位移，小誤差不影響結論方向。"""
    dt = datetime.strptime(time_str, "%Y/%m/%d %H:%M")
    return int(time_module.mktime(dt.timetuple()) * 1000)


def forward_close(cache_times, cache_closes, entry_ms, n_bars_ahead):
    """找到 entry_ms 之後(含)最近的K棒索引，往後推 n_bars_ahead 根，回傳該根的close。
    找不到（超出資料範圍）回傳 None。"""
    idx = bisect.bisect_left(cache_times, entry_ms)
    target = idx + n_bars_ahead
    if target >= len(cache_closes):
        return None
    return cache_closes[target]


def run_test():
    df = pd.read_csv(TRADES_CSV)
    df["entry_ms"] = df["Time"].apply(time_str_to_ms)

    caches = {}  # (symbol, tf) -> (times_list, closes_list)

    def get_cache(symbol, tf):
        key = (symbol, tf)
        if key not in caches:
            c = load_cache(symbol, tf)
            caches[key] = (c["time"].tolist(), c["close"].tolist())
        return caches[key]

    print("=" * 78)
    print("裁決性測試：核心方向判斷，是否顯著優於隨機丟硬幣（50%）")
    print("=" * 78)

    overall_rows = []

    for tf, horizons in [("5m", HORIZONS_5M), ("15m", HORIZONS_15M)]:
        sub_tf = df[df["structure_tf"] == tf]
        if len(sub_tf) == 0:
            continue
        print(f"\n--- structure_tf = {tf}（樣本數 {len(sub_tf)}）---")

        for n in horizons:
            hits, total = 0, 0
            for _, row in sub_tf.iterrows():
                symbol = row["Symbol"]
                times_list, closes_list = get_cache(symbol, tf)
                fwd = forward_close(times_list, closes_list, row["entry_ms"], n)
                if fwd is None:
                    continue
                entry = row["Entry"]
                predicted_up = (row["Direction"] == "long")
                actual_up = fwd > entry
                hit = (predicted_up == actual_up)
                hits += int(hit)
                total += 1

            if total == 0:
                continue

            acc = hits / total
            # 檢定用單邊（只關心是否顯著優於隨機），但信賴區間一律用雙邊
            # （單邊檢定回傳的proportion_ci上界恆為100%，沒有描述性意義，
            #  這裡改成獨立算一個雙邊Wilson score interval供讀者判斷準確率的合理範圍）
            result = stats.binomtest(hits, total, 0.5, alternative="greater")
            p_val = result.pvalue
            ci_two_sided = stats.binomtest(hits, total, 0.5, alternative="two-sided") \
                .proportion_ci(confidence_level=0.95)

            flag = "★ 顯著優於隨機" if p_val < 0.05 else "  無法拒絕隨機假設"
            print(f"  N={n:>3}根後  方向準確率={acc*100:5.1f}%  "
                  f"95%CI(雙邊)=[{ci_two_sided.low*100:.1f}%, {ci_two_sided.high*100:.1f}%]  "
                  f"p值(單邊,vs50%)={p_val:.4f}  {flag}  (n={total})")

            overall_rows.append({
                "structure_tf": tf, "horizon_bars": n,
                "accuracy": acc, "p_value": p_val, "n": total,
            })

    print("\n" + "=" * 78)
    print("解讀方式：")
    print("- p值 < 0.05 才代表方向準確率顯著高於50%（丟硬幣），是真正的方向性edge")
    print("- 若所有horizon、所有時框的p值都 >= 0.05：核心BOS/CHoCH訊號本身")
    print("  沒有方向預測力，後面不管怎麼調停利/停損/成本模型都無法讓策略獲利，")
    print("  因為問題出在訊號設計，不是參數調校可以解決的層次")
    print("- 若某些horizon顯著、某些不顯著：要看是否有一致的pattern")
    print("  （例如短horizon顯著但長horizon不顯著，代表訊號有短暫的動能延續，")
    print("   但你目前TP設定的距離可能已經超出這個有效窗口，這是可以調整的方向）")
    print("=" * 78)

    if overall_rows:
        result_df = pd.DataFrame(overall_rows)

        # [新增] 多重比較校正：一次跑多個horizon×時框組合，單獨看某一個p<0.05
        # 沒有意義——純雜訊下，跑k個檢定，至少矇到一次p<0.05的機率是
        # 1-(1-0.05)^k，k=8時高達約34%。這裡用Bonferroni校正（保守但簡單透明），
        # 把顯著門檻除以檢定次數，同時印出「至少一次假陽性」的基準機率讓你有感。
        k = len(result_df)
        bonferroni_alpha = 0.05 / k
        chance_of_fluke = 1 - (1 - 0.05) ** k
        result_df["significant_raw(p<0.05)"] = result_df["p_value"] < 0.05
        result_df["significant_bonferroni"] = result_df["p_value"] < bonferroni_alpha

        print(f"\n[多重比較校正] 本次共跑了 {k} 個獨立檢定")
        print(f"  若真實情況是純隨機(50%)，這 {k} 次檢定中至少矇到一次 p<0.05 的機率 ≈ {chance_of_fluke*100:.1f}%")
        print(f"  Bonferroni校正後的顯著門檻：p < {bonferroni_alpha:.5f}")
        n_sig_raw = result_df["significant_raw(p<0.05)"].sum()
        n_sig_corrected = result_df["significant_bonferroni"].sum()
        print(f"  校正前「顯著」筆數：{n_sig_raw} / {k}")
        print(f"  校正後「顯著」筆數：{n_sig_corrected} / {k}")
        if n_sig_raw > 0 and n_sig_corrected == 0:
            print("  → 校正前有看似顯著的結果，但校正後全部不顯著，"
                  "代表那個結果很可能只是多重比較下的假陽性，不是真實edge。")

        result_df.to_csv("decisive_direction_test_result.csv", index=False)
        print("\n結果已存至 decisive_direction_test_result.csv")


if __name__ == "__main__":
    run_test()