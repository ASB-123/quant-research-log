"""
diagnose6.py
====================
在 diagnose5.py 基礎上修正三個問題：

1. [更正] 原本懷疑的「進場當根K棒漏記MAE/MFE」經追蹤程式碼後證實不是bug
   （進場價=當根收盤價+滑價，進場後該根已收盤，沒有剩餘波動可追蹤，
   worst_price/best_price初始化為entry_是正確的）。故本版不修改回測程式碼，
   只針對「診斷腳本本身」的統計方法補強。

2. [新增] 時間切分驗證：把資料依Time切成前後兩段，同樣的診斷分別跑一次，
   避免結論只是對某段歷史雜訊的巧合反應。

3. [新增] 第4、5節加上統計顯著性檢定（二項分布信賴區間 + 卡方檢定），
   避免樣本數不足時把雜訊誤判成結構性差異。

4. [新增] 針對「MFE_R>=1卻還是輸」的交易，多切一次 retest_mode /
   bars_since_prev_event，區分「出場管理沒接住」vs「entry本身太晚
   （追在動能衰竭處進場）」兩種完全不同的病灶。
"""

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("backtest_result_smc_v1.csv")
df["Time"] = pd.to_datetime(df["Time"], format="%Y/%m/%d %H:%M")
df = df.sort_values("Time").reset_index(drop=True)


def wilson_ci(successes, n, z=1.96):
    """Wilson score interval，樣本數小時比常態近似穩健，適合用在<100筆的分組勝率"""
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    adj = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    return ((centre - adj) / denom, (centre + adj) / denom)


def section_header(title):
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


# ----------------------------------------------------------------
# 0) 基本盤
# ----------------------------------------------------------------
section_header("0) 基本盤：手續費占比是否真的被壓下來了")
print(df["fee_pct_of_risk"].describe())
print(f"超過30%上限的筆數: {(df['fee_pct_of_risk'] > 0.30).sum()} / {len(df)}")


# ----------------------------------------------------------------
# 1) 時間切分：前後兩段是否得出一致的結論
# ----------------------------------------------------------------
section_header("1) 時間切分驗證（前60% vs 後40%）")
split_idx = int(len(df) * 0.6)
df_train, df_test = df.iloc[:split_idx], df.iloc[split_idx:]
print(f"前段: {df_train['Time'].min()} ~ {df_train['Time'].max()}  筆數={len(df_train)}")
print(f"後段: {df_test['Time'].min()} ~ {df_test['Time'].max()}  筆數={len(df_test)}")

for name, sub in [("前60%", df_train), ("後40%", df_test)]:
    wr = (sub["Result"] > 0).mean() * 100
    avg_r = sub["R_Multiple"].mean()
    fee_med = sub["fee_pct_of_risk"].median()
    print(f"  {name}: 勝率={wr:5.1f}%  平均R倍數={avg_r:+.3f}  "
          f"手續費占比中位數={fee_med:.3f}")

print("\n→ 若前後兩段的勝率/R倍數/手續費占比落差很大，代表下面的所有分組結論\n"
      "  都可能只是對某一段歷史雜訊的巧合反應，不是穩定的結構性現象。")


# ----------------------------------------------------------------
# 2) MFE_R / MAE_R 分布（沿用 diagnose5，但同時列出前後兩段）
# ----------------------------------------------------------------
section_header("2) 輸家 MFE_R 分布：是否「有機會但沒守住」")

for name, sub in [("全部", df), ("前60%", df_train), ("後40%", df_test)]:
    losers = sub[sub["Result"] <= 0]
    if len(losers) == 0:
        continue
    never_favorable = (losers["MFE_R"] < 0.3).mean() * 100
    some_favorable = ((losers["MFE_R"] >= 0.3) & (losers["MFE_R"] < 1.0)).mean() * 100
    reached_1r = (losers["MFE_R"] >= 1.0).mean() * 100
    print(f"  [{name}] 輸家n={len(losers)}  "
          f"MFE_R<0.3(方向問題)={never_favorable:.1f}%  "
          f"0.3~1(灰色地帶)={some_favorable:.1f}%  "
          f">=1(管理問題?)={reached_1r:.1f}%")


# ----------------------------------------------------------------
# 3) 打平所需盈虧比 vs 實際盈虧比（全部 + 前後兩段）
# ----------------------------------------------------------------
section_header("3) 打平所需盈虧比 vs 實際盈虧比")

def rr_report(sub, name):
    winners, losers = sub[sub["Result"] > 0], sub[sub["Result"] <= 0]
    win_rate = len(winners) / len(sub) if len(sub) else np.nan
    avg_win = winners["Result"].mean() if len(winners) else np.nan
    avg_loss = abs(losers["Result"].mean()) if len(losers) else np.nan
    required_rr = (1 - win_rate) / win_rate if win_rate else np.nan
    actual_rr = avg_win / avg_loss if avg_loss else np.nan
    print(f"  [{name}] 勝率={win_rate*100:5.1f}%  打平需要RR={required_rr:.3f}  "
          f"實際RR={actual_rr:.3f}  缺口={actual_rr - required_rr:+.3f}")

rr_report(df, "全部")
rr_report(df_train, "前60%")
rr_report(df_test, "後40%")


# ----------------------------------------------------------------
# 4) structure_type 拆解 + 卡方檢定顯著性
# ----------------------------------------------------------------
section_header("4) structure_type (BOS vs CHoCH) 拆解 + 顯著性檢定")

types = df["structure_type"].dropna().unique()
rows = []
for st in types:
    sub = df[df["structure_type"] == st]
    n = len(sub)
    wins = (sub["Result"] > 0).sum()
    wr = wins / n * 100
    lo, hi = wilson_ci(wins, n)
    mfe_med = sub["MFE_R"].median()
    print(f"  {st:<8} 筆數={n:>5}  勝率={wr:5.1f}%  "
          f"95%CI=[{lo*100:.1f}%, {hi*100:.1f}%]  "
          f"MFE_R中位數={mfe_med:.3f}  平均R倍數={sub['R_Multiple'].mean():+.3f}")
    rows.append((st, wins, n - wins))

if len(rows) >= 2:
    contingency = np.array([[r[1], r[2]] for r in rows])
    chi2, p, _, _ = stats.chi2_contingency(contingency)
    print(f"\n  卡方檢定（structure_type間勝率是否顯著不同）: chi2={chi2:.3f}  p值={p:.4f}")
    print("  → p < 0.05 才代表不同structure_type之間的勝率差異在統計上顯著，"
          "否則現在看到的差距很可能只是樣本數不足造成的雜訊。")


# ----------------------------------------------------------------
# 5) ob_reused 拆解 + 顯著性檢定
# ----------------------------------------------------------------
section_header("5) ob_reused 拆解 + 顯著性檢定")

rows = []
for reused in [False, True]:
    sub = df[df["ob_reused"] == reused]
    if len(sub) == 0:
        continue
    n = len(sub)
    wins = (sub["Result"] > 0).sum()
    wr = wins / n * 100
    lo, hi = wilson_ci(wins, n)
    print(f"  ob_reused={reused!s:<5} 筆數={n:>5}  勝率={wr:5.1f}%  "
          f"95%CI=[{lo*100:.1f}%, {hi*100:.1f}%]  "
          f"平均R倍數={sub['R_Multiple'].mean():+.3f}")
    rows.append((reused, wins, n - wins))

if len(rows) == 2:
    contingency = np.array([[r[1], r[2]] for r in rows])
    chi2, p, _, _ = stats.chi2_contingency(contingency)
    print(f"\n  卡方檢定（ob_reused=True/False勝率是否顯著不同）: chi2={chi2:.3f}  p值={p:.4f}")
    print("  → 若 p >= 0.05，代表ob_reused目前只是把樣本切成兩份，"
          "沒有真正篩選訊號品質（這跟你先前的觀察一致：只是等比例減少樣本）。")


# ----------------------------------------------------------------
# 6) [新增] MFE_R>=1卻還是輸的交易，到底是「管理問題」還是「entry太晚」
# ----------------------------------------------------------------
section_header("6) MFE_R>=1的輸家：依 retest_mode / bars_since_prev_event 交叉分析")

high_mfe_losers = df[(df["Result"] <= 0) & (df["MFE_R"] >= 1.0)]
print(f"MFE_R>=1的輸家總數: {len(high_mfe_losers)} / 全部輸家 {(df['Result'] <= 0).sum()}")

if len(high_mfe_losers) > 0:
    print("\n依 retest_mode 拆解：")
    for mode in high_mfe_losers["retest_mode"].dropna().unique():
        sub = high_mfe_losers[high_mfe_losers["retest_mode"] == mode]
        all_mode = df[df["retest_mode"] == mode]
        pct_of_mode = len(sub) / len(all_mode) * 100 if len(all_mode) else np.nan
        print(f"  {mode:<8} 筆數={len(sub):>4}  "
          f"佔該retest_mode全部交易的{pct_of_mode:.1f}%")

    print("\n依 bars_since_prev_event 分組（事件密集度，越小代表結構轉折越密集）：")
    bins = [0, 5, 15, 30, 60, np.inf]
    labels = ["<5根(極密集)", "5-15根", "15-30根", "30-60根", ">60根(稀疏)"]
    high_mfe_losers = high_mfe_losers.copy()
    high_mfe_losers["density_bucket"] = pd.cut(
        high_mfe_losers["bars_since_prev_event"], bins=bins, labels=labels)
    bucket_counts = high_mfe_losers["density_bucket"].value_counts().sort_index()
    print(bucket_counts)

    print("\n→ 解讀方式：")
    print("  若 MFE_R>=1 的輸家集中在特定 retest_mode 或特定密集度區間，")
    print("  代表問題出在「特定情境下的entry timing」（追在動能已衰竭的位置進場），")
    print("  而不是單純的停利/停損設計問題——這種情況下調整TP/SL邏輯不會有幫助，")
    print("  該回頭檢視進場條件本身（尤其是 ENABLE_IMMEDIATE_ENTRY 消融測試的結果）。")
    print("  若分布平均、沒有集中在特定情境，才比較支持「單純出場管理沒接住」的假設。")