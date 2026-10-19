"""
plot_smc_structure.py
====================
讀取 K 線 CSV，跑 smc_structure.py 的 BOS/CHoCH + OB 偵測，
畫成可以自己肉眼核對的圖，存成 PNG。

用法：
    python plot_smc_structure.py --csv cache_BTCUSDT_5m.csv --days 7 --tf 5m --length 50

參數說明：
    --csv     K線 CSV 檔路徑（需含 time,open,high,low,close,baseVol 欄位）
    --days    只取最後 N 天的資料來畫圖（避免圖太擠）
    --tf      標籤用，純粹顯示在圖上，不影響計算（計算週期取決於你 CSV 本身是幾分鐘K）
    --length  smc_structure.compute_pivots 的 pivot 確認延遲根數，越大代表結構
              判斷越保守（雜訊少但反應慢），越小越敏感（雜訊多但反應快）
    --out     輸出圖片檔名
"""

import argparse
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

from smc_structure import compute_structure_events


def load_and_filter(csv_path: str, days: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["time"] = df["time"].astype(int)
    df = df.sort_values("time").reset_index(drop=True)

    if days is not None:
        cutoff = df["time"].max() - days * 24 * 60 * 60 * 1000
        df_plot_range = df[df["time"] >= cutoff].reset_index(drop=True)
    else:
        df_plot_range = df

    return df, df_plot_range


def plot_structure(full_df: pd.DataFrame, plot_df: pd.DataFrame,
                    timeframe_label: str, length: int, out_path: str,
                    ob_bars_width: int = 40):
    """
    full_df: 完整歷史資料，用來算 pivot / 結構事件（需要足夠前置資料才能正確判斷）
    plot_df: 只用來決定畫圖的時間範圍
    """
    # 用完整資料算事件，才不會因為資料太短而漏掉真正的 pivot
    events = compute_structure_events(full_df, length=length, timeframe_label=timeframe_label)

    plot_start_ms = plot_df["time"].min()
    plot_end_ms = plot_df["time"].max()
    events_in_range = events[
        (events["time"] >= plot_start_ms) & (events["time"] <= plot_end_ms)
    ].reset_index(drop=True)

    print(f"畫圖範圍內共 {len(events_in_range)} 個結構事件"
          f"（BOS: {(events_in_range['type']=='BOS').sum()}，"
          f"CHoCH: {(events_in_range['type']=='CHoCH').sum()}）")

    df_idx = plot_df.copy()
    df_idx["datetime"] = pd.to_datetime(df_idx["time"], unit="ms", utc=True)
    df_idx = df_idx.set_index("datetime")
    df_ohlcv = df_idx[["open", "high", "low", "close", "baseVol"]].rename(
        columns={"baseVol": "volume"})

    n = len(df_ohlcv)
    time_arr = plot_df["time"].values

    fig, axes = mpf.plot(
        df_ohlcv, type="candle", style="charles", volume=True,
        figsize=(22, 11), returnfig=True, warn_too_much_data=100000,
        title=f"BTCUSDT {timeframe_label} -- BOS/CHoCH + Order Block "
              f"(length={length})"
    )
    ax = axes[0]

    for _, ev in events_in_range.iterrows():
        x_start = int(np.searchsorted(time_arr, ev["time"]))
        color = "green" if ev["direction"] == "bullish" else "red"
        linestyle = "-" if ev["type"] == "BOS" else "--"
        label_text = f"{ev['type']}({ev['direction'][0].upper()})"

        # 水平線：從事件發生的那根K棒開始，畫到圖表最右邊（標準 SMC 畫法）
        ax.hlines(y=ev["level"], xmin=x_start, xmax=n - 1, color=color,
                  linestyle=linestyle, linewidth=1.3, alpha=0.8)

        ax.annotate(label_text, xy=(x_start, ev["level"]),
                    xytext=(5, 9 if ev["direction"] == "bullish" else -14),
                    textcoords="offset points", fontsize=8, color=color,
                    fontweight="bold")

        # OB 區域：只在發生點附近畫一小段，避免整條貫穿造成視覺混亂
        ob_high, ob_low = ev["ob_high"], ev["ob_low"]
        ob_color = "lightgreen" if ev["direction"] == "bullish" else "lightcoral"
        x_ob_end = min(x_start + ob_bars_width, n - 1)
        ax.axhspan(ob_low, ob_high, xmin=x_start / n, xmax=x_ob_end / n,
                   color=ob_color, alpha=0.3)

    bos_g = plt.Line2D([0], [0], color="green", linestyle="-", label="BOS (bullish)")
    choch_g = plt.Line2D([0], [0], color="green", linestyle="--", label="CHoCH (bullish)")
    bos_r = plt.Line2D([0], [0], color="red", linestyle="-", label="BOS (bearish)")
    choch_r = plt.Line2D([0], [0], color="red", linestyle="--", label="CHoCH (bearish)")
    ax.legend(handles=[bos_g, choch_g, bos_r, choch_r], loc="upper left", fontsize=9)

    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"圖已存到: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="繪製 SMC BOS/CHoCH + OB 結構圖，供肉眼核對")
    parser.add_argument("--csv", required=True, help="K線 CSV 路徑")
    parser.add_argument("--days", type=int, default=7, help="只取最後 N 天畫圖（預設7天）")
    parser.add_argument("--tf", default="5m", help="時間週期標籤（僅顯示用）")
    parser.add_argument("--length", type=int, default=50, help="pivot 確認延遲根數")
    parser.add_argument("--out", default="smc_chart.png", help="輸出圖片檔名")
    parser.add_argument("--ob-width", type=int, default=40,
                        help="OB色塊寬度，以K棒根數計（預設40）")
    args = parser.parse_args()

    full_df, plot_df = load_and_filter(args.csv, args.days)
    plot_structure(full_df, plot_df, args.tf, args.length, args.out, args.ob_width)


if __name__ == "__main__":
    main()