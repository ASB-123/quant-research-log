"""
bitunix_backtest_smc_v1.py
====================
ICT/SMC 版回測（取代 v21 的 POC 突破邏輯）

[診斷版修改]：在每筆交易記錄加入 swing_amplitude / amplitude_atr_ratio / event_density_before
三個欄位，用來事後驗證「盤整區間內密集假CHoCH/BOS導致連續虧損」這個假設，
不改變任何進場/出場/資金管理邏輯，純粹是多記錄診斷用欄位。
"""

import requests
import pandas as pd
import numpy as np
import time
import os
import bisect
from datetime import datetime, timezone

from liquidity_zones import (
    compute_key_opens, compute_prev_day_high_low, get_entry_timeframe, sl_from_ob_range,
)
from smc_structure import compute_structure_events

BASE_URL  = "https://fapi.bitunix.com"

SYMBOLS   = ["ETHUSDT", "BTCUSDT", "SOLUSDT"]
PAGES_5M  = 1060
PAGES_15M = 360
FEE_RATE  = 0.0006
FUND_RATE = 0.000574

SL_BUFFER_RATIO   = 0.15  # SL緩衝 = OB區間高度 × 此比例（取代固定價格百分比）
ATR_FLOOR_MULT    = 0.5   # SL距離地板 = ATR × 此倍數，避免OB太窄時止損比雜訊還小
ATR_PERIOD        = 14
STRUCT_LENGTH_5M  = 30   # 由50降低，pivot確認等待時間變短，結構事件密度增加
STRUCT_LENGTH_15M = 30
OB_MAX_AGE_BARS   = 200  # 由96提高，OB存活期拉長，更多事件有機會被retest到

# ---------------------------------------------------------------
# [重構] 風險管理：由「固定NOTIONAL下注」改為「固定風險下注」
# ---------------------------------------------------------------
# 舊版用固定 NOTIONAL=$25 反推qty，risk_usdt = risk_dist * qty 完全隨risk_dist漂移。
# 這是診斷出的結構性病灶：risk_dist越窄，risk_usdt越小，手續費/滑價占比就越誇張，
# 且這批單子的quick_stop比例系統性偏高（雜訊本來就容易掃到窄止損）。
# 上一輪用MIN_RISK_USDT門檻硬濾掉，只是治標；這裡改成機構標準的
# Fixed Fractional Risk Sizing：每筆固定承擔 TARGET_RISK_USDT，
# qty = TARGET_RISK_USDT / risk_dist，risk_usdt因此恆等於TARGET_RISK_USDT
# （除非被下面的槓桿上下限擋掉），從根本解決手續費占比隨機漂移的問題。
TARGET_RISK_USDT  = 0.12   # 每筆固定風險金額，取原本RISK中位數($0.0848)與表現較佳的60-80%分位
                            # 區間($0.1347)之間，作為新基準
MAX_NOTIONAL_USDT = 500.0  # 名目上限（$25本金下約20x槓桿），risk_dist過窄時qty會被放大到
                            # 不合理的槓桿，此時直接跳過該筆交易
MIN_NOTIONAL_USDT = 10.0   # 名目下限，risk_dist過寬時部位會小到手續費占比又失真，一併跳過

# [新增，補上v2遺漏的一塊] 手續費佔風險比例的直接過濾器
# ---------------------------------------------------------------
# 上面的固定風險sizing只解決了「risk_usdt隨risk_dist亂漂移」的問題，
# 但沒解決「手續費佔風險比例」本身——這個比例的公式是：
#     fee_pct_of_risk ≈ 2 × FEE_RATE × entry_price / risk_dist
# 完全不受TARGET_RISK_USDT大小影響（分子分母同比例增減會消掉），
# 純粹由「停損距離佔價格的百分比」決定。實測v2平均risk_dist/entry僅約0.26%，
# 換算下來手續費平均吃掉將近一半的風險預算，這是這次v2結果比v1更差的主因，
# 跟策略方向判斷準不準無關，是費率與停損寬度的結構性不匹配。
# 這裡直接從費率反推「risk_dist至少要佔價格多少百分比」，
# 才能把手續費占比壓在FEE_PCT_OF_RISK_CEILING以下（用最單純的進場+一次性
# 停損出場計算，套保後多次分批出場的實際占比會更低，此為保守估計）。
FEE_PCT_OF_RISK_CEILING = 0.30  # 手續費最多吃掉風險預算的30%（保守但不算太嚴苛的機構常見門檻）
MIN_RISK_PCT_OF_PRICE = 2 * FEE_RATE / FEE_PCT_OF_RISK_CEILING  # 由費率反推，非拍腦袋數字

# [新增] 滑價模型：原本假設零滑價，會高估策略edge，尤其是這次放寬的影線retest+
# 緊停損組合最容易在快速行情吃到滑價。只套用在「市價性質」的成交：進場與
# 未套保的停損出場；套保後的兩平出場經濟意義已由breakeven_price固定，
# 不重複疊加滑價（避免過度複雜化）。
SLIPPAGE_BPS = 2.0  # 單邊滑價，basis points

# [新增] 波動regime過濾：目前ATR相對過去窗口的分位數過低（死水盤整）時不進場，
# 這類環境最容易出現插針型假突破/假retest，是這次放寬條件後最容易滲入的雜訊來源。
ATR_PCTL_WINDOW = 500
ATR_PCTL_FLOOR  = 0.20

# [新增] 消融測試開關：這次一次改了4個地方(STRUCT_LENGTH/OB_MAX_AGE/retest放寬/OB重複使用)，
# 交易數雖然衝上來但整體品質變差，需要拆解是哪個改動在拖累。
# 把「retest放寬」跟「OB重複使用」獨立成開關，方便逐一關掉重跑比較。
ENABLE_WICK_RETEST = True   # True=影線觸及OB即算retest；False=退回收盤價須落在OB內（改動前行為）
ENABLE_OB_REUSE    = True   # True=部位平倉後釋放used_obs可重新進場；False=退回用過就永久封存（改動前行為）

# [新增，依diagnose5發現] OB觸碰次數過濾器
# ---------------------------------------------------------------
# diagnose5的第5節發現：ob_reused=True（曾經在此OB開過倉又平倉、後來又重新進場）
# 的表現明顯優於ob_reused=False（平均R倍數-0.074 vs -0.394，勝率30.0% vs 25.2%）。
# 這跟「重複使用的OB是失敗結構」的假設相反——反而是被驗證過還能守住的位置更紮實，
# 符合技術分析裡「被測試過的關鍵位更強」的邏輯。
# 這裡把它從「事後才知道」的標記，改成「事前」可用的過濾條件：
# 不在OB第一次被retest時就進場，要等它被touch_count次以上才允許進場，
# 用重複觸碰次數本身當作結構強度的代理指標。
ENABLE_OB_TOUCH_FILTER = False  # 已驗證：只是等比例減少樣本，沒有實際篩選效果，退回不啟用
MIN_OB_TOUCHES = 2  # 第1次觸碰只記錄不進場，第2次(含)以後才允許進場

# [新增] 核心訊號消融測試：這幾輪下來，切了symbol/structure_type/direction/
# direction_flip/amplitude_atr_ratio等維度，勝率全部卡在25-27%沒有明顯差異，
# 顯示問題可能不在風控/成本層，而是BOS/CHoCH+OB回測進場這個核心訊號本身。
# 這個開關讓你能對比兩種進場方式：
#   False（原本行為）：結構事件確認後，等OB被retest到、且下一根K棒收盤價
#                     繼續往結構方向走才進場（多一層「回測確認」的延遲）
#   True            ：結構事件（BOS/CHoCH）一確認就直接在當根K棒收盤進場，
#                     跳過retest等待。如果這樣勝率反而更好，代表retest的等待
#                     延遲讓我們買在動能已經衰竭的位置；如果更差，代表retest
#                     確實有過濾雜訊的作用，問題要往更早的結構判斷邏輯找
ENABLE_IMMEDIATE_ENTRY = False

TP1_RATIO = 0.60
TP2_RATIO = 0.25
TP3_RATIO = 0.15

FUNDING_HOURS_UTC = [0, 8, 16]

# ----------------------------------------------
# 數據抓取（與 v21 相同，未變更）
# ----------------------------------------------

def fetch_klines(symbol, interval, limit=200, end_time=None):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time:
        params["endTime"] = end_time
    for attempt in range(5):
        try:
            res  = requests.get(f"{BASE_URL}/api/v1/futures/market/kline",
                                params=params, timeout=30)
            data = res.json().get("data", [])
            return data
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
# 資金費率（與 v21 相同，未變更）
# ----------------------------------------------

def calc_funding_fee_v2(entry_ms, exit_ms, notional):
    if exit_ms <= entry_ms:
        return 0.0
    entry_dt = datetime.fromtimestamp(entry_ms / 1000, tz=timezone.utc)
    exit_dt  = datetime.fromtimestamp(exit_ms  / 1000, tz=timezone.utc)
    count = 0
    day = entry_dt.date()
    last_day = exit_dt.date()
    while day <= last_day:
        for h in FUNDING_HOURS_UTC:
            point = datetime(day.year, day.month, day.day, h, 0, 0, tzinfo=timezone.utc)
            if entry_dt < point <= exit_dt:
                count += 1
        day = day.fromordinal(day.toordinal() + 1)
    return count * FUND_RATE * notional

# ----------------------------------------------
# ATR：用來當SL距離的地板
# ----------------------------------------------

def add_atr(df, period=ATR_PERIOD):
    df = df.copy()
    c, h, l = df["close"], df["high"], df["low"]
    tr = pd.concat([
        h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=period, adjust=False).mean()
    return df


# ----------------------------------------------
# [新增] ATR分位數：用來判斷目前是不是死水盤整regime
# ----------------------------------------------

def add_atr_percentile(df, window=ATR_PCTL_WINDOW):
    """
    對每根K棒計算「目前ATR」在過去window根K棒範圍內的分位數(0~1)，
    只看得到過去資料，不偷看未來。分位數過低代表目前波動異常小，
    容易出現插針型假突破/假retest。
    """
    df = df.copy()
    df["atr_pctl"] = df["atr"].rolling(window, min_periods=50).apply(
        lambda x: (x < x[-1]).mean(), raw=True
    )
    return df


# ----------------------------------------------
# [新增] 滑價：進場與未套保停損出場套用，模擬市價/停損單的不利成交
# ----------------------------------------------

def apply_slippage(price, side, is_entry):
    """
    is_entry=True  → 進場方向的不利滑價（追價成交）
    is_entry=False → 出場方向的不利滑價（停損被打穿後的滑點）
    """
    s = SLIPPAGE_BPS / 10000.0
    if side == "BUY":
        return price * (1 + s) if is_entry else price * (1 - s)
    else:
        return price * (1 - s) if is_entry else price * (1 + s)


# ----------------------------------------------
# 同根K棒 SL/TP 先後判斷（與 v21 相同，未變更）
# ----------------------------------------------

def resolve_same_bar_sl_vs_tp(side, o, high, low, sl_price, tp_price):
    sl_hit_raw = (side == "BUY"  and low  <= sl_price) or \
                 (side == "SELL" and high >= sl_price)
    tp_hit_raw = False
    if tp_price is not None:
        tp_hit_raw = (side == "BUY"  and high >= tp_price) or \
                     (side == "SELL" and low  <= tp_price)
    if sl_hit_raw and tp_hit_raw:
        sl_dist = abs(o - sl_price)
        tp_dist = abs(o - tp_price)
        return (True, False) if sl_dist <= tp_dist else (False, True)
    return sl_hit_raw, tp_hit_raw

# ----------------------------------------------
# 新增：真正兩平價（把殘倉的平倉手續費算進去，避免套保後仍因手續費小虧）
# ----------------------------------------------

def breakeven_price(entry_price, side, fee_rate=FEE_RATE):
    """
    TP1觸發後SL不能只設entry_price——那樣殘倉平倉時還要付一次手續費，
    等於「兩平」實際上是小虧。這裡把手續費補回去，確保殘倉這段價格波動
    至少能cover自己的出場手續費。
    """
    if side == "BUY":
        return entry_price * (1 + fee_rate)
    else:
        return entry_price * (1 - fee_rate)



LEVEL_COLS = ["open_1h", "open_4h", "open_8h", "open_1d", "open_1w", "open_1M",
              "prev_day_high", "prev_day_low"]

MIN_TP1_PCT = 0.003  # TP1至少要有0.3%空間（來回手續費0.12%的2.5倍），太近的流動性位直接跳過
MIN_TP1_RR  = 1.8     # TP1距離至少要是risk_dist的1.8倍，對齊目前約36%命中率所需的打平賺賠比

def build_liquidity_targets(row, side, entry, risk_dist=0.0):
    """取進場方向前方最近的關鍵流動性位，依距離排序，最多回傳3個。
    TP1必須同時滿足：(1)最小百分比 MIN_TP1_PCT (2)最小風報比 MIN_TP1_RR×risk_dist，
    兩者取較大值當門檻——避免抓到「離entry夠遠但相對這筆風險而言還是太近」的流動性位。"""
    levels = []
    for c in LEVEL_COLS:
        v = row.get(c)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if side == "BUY" and v > entry:
            levels.append(float(v))
        elif side == "SELL" and v < entry:
            levels.append(float(v))

    levels = sorted(levels) if side == "BUY" else sorted(levels, reverse=True)

    dedup = []
    for lv in levels:
        if not dedup or abs(lv - dedup[-1]) / entry > 0.001:
            dedup.append(lv)

    min_dist = max(entry * MIN_TP1_PCT, risk_dist * MIN_TP1_RR)
    dedup = [lv for lv in dedup if abs(lv - entry) >= min_dist]
    return dedup[:3]

# ----------------------------------------------
# 單幣回測：結構(BOS/CHoCH) + OB 回測進場
# ----------------------------------------------

def backtest_symbol(symbol, df5, df15):
    positions = []
    trades    = []
    used_obs  = set()   # (timeframe, ob_time) 已進場過的結構事件，避免重複觸發
    seen_ob_keys = set()  # [新增] 記錄「曾經成功開過倉」的ob_key，供ob_reused欄位判斷
    touch_counts = {}     # [新增，依diagnose5發現] 記錄每個ob_key被retest觸碰過幾次，見上方MIN_OB_TOUCHES註解
    pending   = {}       # (timeframe, ob_time) -> 等待下一根反轉確認的retest資訊

    df5 = df5.copy()
    df5["returns"]          = df5["close"].pct_change()
    df5["strategy_returns"] = 0.0
    df5 = compute_key_opens(df5)
    df5 = compute_prev_day_high_low(df5)
    df5 = add_atr(df5)
    df5 = add_atr_percentile(df5)
    df15 = add_atr(df15)
    df15 = add_atr_percentile(df15)

    print(f"    計算結構事件（5m / 15m 雙時框，BOS/CHoCH + 綁定OB）...")
    ev5  = compute_structure_events(df5,  STRUCT_LENGTH_5M,  "5m").sort_values("time").reset_index(drop=True)
    ev15 = compute_structure_events(df15, STRUCT_LENGTH_15M, "15m").sort_values("time").reset_index(drop=True)
    ev5_times  = ev5["time"].tolist()
    ev15_times = ev15["time"].tolist()
    print(f"    5m 結構事件 {len(ev5)} 筆 / 15m 結構事件 {len(ev15)} 筆")

    def get_active_event(t):
        tf = get_entry_timeframe(t)
        ev, times = (ev5, ev5_times) if tf == "5m" else (ev15, ev15_times)
        pos = bisect.bisect_right(times, t) - 1
        if pos < 0:
            return None, tf
        return ev.iloc[pos], tf

    df15_times = df15["time"].tolist()
    df15_atr   = df15["atr"].tolist()
    df15_atr_pctl = df15["atr_pctl"].tolist()

    def get_atr_for(t, tf):
        if tf == "5m":
            return None  # 會用當根5m row["atr"]，不用查表
        pos = bisect.bisect_right(df15_times, t) - 1
        if pos < 0:
            return None
        return df15_atr[pos]

    def get_atr_pctl_for(t, tf):
        if tf == "5m":
            return None  # 會用當根5m row["atr_pctl"]，不用查表
        pos = bisect.bisect_right(df15_times, t) - 1
        if pos < 0:
            return None
        return df15_atr_pctl[pos]

    for i in range(30, len(df5)):
        row  = df5.iloc[i]
        o    = float(row["open"])
        high = float(row["high"])
        low  = float(row["low"])
        t    = int(row["time"])
        close = float(row["close"])

        # ── 更新所有持倉（機制與 v21 完全相同）──
        closed_idx = []
        for pi, pos in enumerate(positions):
            side = pos["side"]

            # [新增] MAE/MFE追蹤：記錄進場以來最不利與最有利的價格，
            # 用來事後判斷停損距離設定是否合理，不用再靠RISK分位數間接猜測
            if side == "BUY":
                pos["worst_price"] = min(pos["worst_price"], low)
                pos["best_price"]  = max(pos["best_price"], high)
            else:
                pos["worst_price"] = max(pos["worst_price"], high)
                pos["best_price"]  = min(pos["best_price"], low)

            tp_price_for_order = None
            if pos["tp_hit"] == 0 and pos["tp1"]:
                tp_price_for_order = pos["tp1"]
            elif pos["tp_hit"] == 1 and pos["tp2"]:
                tp_price_for_order = pos["tp2"]
            elif pos["tp_hit"] == 2 and pos["tp3"]:
                tp_price_for_order = pos["tp3"]

            sl_hit, _ = resolve_same_bar_sl_vs_tp(side, o, high, low, pos["sl"], tp_price_for_order)

            if sl_hit:
                exit_fill = apply_slippage(pos["sl"], side, is_entry=False)
                qty_r    = pos["remaining_qty"]
                notional = pos["entry_price"] * qty_r
                pnl_raw  = (exit_fill - pos["entry_price"]) * qty_r
                if side == "SELL":
                    pnl_raw = -pnl_raw
                fee     = notional * FEE_RATE
                funding = calc_funding_fee_v2(pos["entry_time"], t, notional)
                net     = pnl_raw - fee - funding + pos["realized_pnl"]
                df5.at[i, "strategy_returns"] += \
                    (exit_fill - pos["entry_price"]) / pos["entry_price"] * (1 if side == "BUY" else -1)
                trades.append(_make_trade(symbol, pos, t, exit_fill,
                              "撞到止損" if pos["tp_hit"] == 0 else "撞到止損(已套保)",
                              net, fee + pos["fee_paid"], funding + pos["fund_paid"], exit_bar_index=i))
                closed_idx.append(pi)
                continue

            tp1_hit = (side == "BUY"  and pos["tp_hit"] == 0 and pos["tp1"] and high >= pos["tp1"]) or \
                      (side == "SELL" and pos["tp_hit"] == 0 and pos["tp1"] and low  <= pos["tp1"])
            if tp1_hit:
                qty_c   = pos["initial_qty"] * TP1_RATIO
                notional= pos["entry_price"] * qty_c
                pnl_raw = abs(pos["tp1"] - pos["entry_price"]) * qty_c
                fee     = notional * FEE_RATE
                funding = calc_funding_fee_v2(pos["entry_time"], t, notional)
                net     = pnl_raw - fee - funding
                pos["realized_pnl"]  += net
                pos["fee_paid"]      += fee
                pos["fund_paid"]     += funding
                pos["remaining_qty"] -= qty_c
                pos["tp_hit"]         = 1
                pos["sl"]             = breakeven_price(pos["entry_price"], side)
                pos["hedged"]         = True

                new_sl_hit, _ = resolve_same_bar_sl_vs_tp(side, o, high, low, pos["sl"], pos["tp2"])
                if new_sl_hit:
                    qty_r    = pos["remaining_qty"]
                    notional = pos["entry_price"] * qty_r
                    fee      = notional * FEE_RATE
                    funding  = calc_funding_fee_v2(pos["entry_time"], t, notional)
                    net      = 0.0 - fee - funding + pos["realized_pnl"]
                    trades.append(_make_trade(symbol, pos, t, pos["sl"], "撞到止損(已套保)",
                                              net, fee + pos["fee_paid"], funding + pos["fund_paid"], exit_bar_index=i))
                    closed_idx.append(pi)
                    continue

            tp2_hit = (side == "BUY"  and pos["tp_hit"] == 1 and pos["tp2"] and high >= pos["tp2"]) or \
                      (side == "SELL" and pos["tp_hit"] == 1 and pos["tp2"] and low  <= pos["tp2"])
            if tp2_hit:
                qty_c   = pos["initial_qty"] * TP2_RATIO
                notional= pos["entry_price"] * qty_c
                pnl_raw = abs(pos["tp2"] - pos["entry_price"]) * qty_c
                fee     = notional * FEE_RATE
                funding = calc_funding_fee_v2(pos["entry_time"], t, notional)
                net     = pnl_raw - fee - funding
                pos["realized_pnl"]  += net
                pos["fee_paid"]      += fee
                pos["fund_paid"]     += funding
                pos["remaining_qty"] -= qty_c
                pos["tp_hit"]         = 2
                pos["sl"]             = breakeven_price(pos["tp1"], side)

                new_sl_hit, _ = resolve_same_bar_sl_vs_tp(side, o, high, low, pos["sl"], pos["tp3"])
                if new_sl_hit:
                    exit_fill = apply_slippage(pos["sl"], side, is_entry=False)
                    qty_r    = pos["remaining_qty"]
                    notional = pos["entry_price"] * qty_r
                    pnl_raw  = abs(exit_fill - pos["entry_price"]) * qty_r
                    fee      = notional * FEE_RATE
                    funding  = calc_funding_fee_v2(pos["entry_time"], t, notional)
                    net      = pnl_raw - fee - funding + pos["realized_pnl"]
                    trades.append(_make_trade(symbol, pos, t, exit_fill, "撞到止損(已套保)",
                                              net, fee + pos["fee_paid"], funding + pos["fund_paid"], exit_bar_index=i))
                    closed_idx.append(pi)
                    continue

            tp3_hit = (side == "BUY"  and pos["tp_hit"] == 2 and pos["tp3"] and high >= pos["tp3"]) or \
                      (side == "SELL" and pos["tp_hit"] == 2 and pos["tp3"] and low  <= pos["tp3"])
            if tp3_hit:
                qty_c   = pos["remaining_qty"]
                notional= pos["entry_price"] * qty_c
                pnl_raw = abs(pos["tp3"] - pos["entry_price"]) * qty_c
                fee     = notional * FEE_RATE
                funding = calc_funding_fee_v2(pos["entry_time"], t, notional)
                net     = pnl_raw - fee - funding + pos["realized_pnl"]
                df5.at[i, "strategy_returns"] += abs(pos["tp3"] - pos["entry_price"]) / pos["entry_price"]
                trades.append(_make_trade(symbol, pos, t, pos["tp3"], "撞到止盈",
                                          net, fee + pos["fee_paid"], funding + pos["fund_paid"], exit_bar_index=i))
                closed_idx.append(pi)

        for pi in sorted(closed_idx, reverse=True):
            closed_pos = positions.pop(pi)
            if ENABLE_OB_REUSE:
                ob_key = closed_pos.get("ob_key")
                if ob_key is not None:
                    used_obs.discard(ob_key)

        # ── 進場條件：結構事件(BOS/CHoCH) + OB回測 → 等下一根反轉確認才進場 ──

        def try_open(side_, ob_low_, ob_high_, tf_, event_type_, entry_, atr_val_,
                     swing_amp_=np.nan, amp_ratio_=np.nan,
                     bars_since_prev_=np.nan, direction_flip_=None, ob_key_=None,
                     atr_pctl_=None, ob_reused_=False):
            # [新增] regime過濾：目前波動太低（死水盤整）就不進場
            if atr_pctl_ is not None and not np.isnan(atr_pctl_) and atr_pctl_ < ATR_PCTL_FLOOR:
                return None

            # [新增] 進場滑價
            entry_ = apply_slippage(entry_, side_, is_entry=True)

            sl_ = sl_from_ob_range(ob_low_, ob_high_, side_, SL_BUFFER_RATIO)
            risk_dist_ = abs(entry_ - sl_)
            if atr_val_ is not None and not np.isnan(atr_val_):
                min_risk_dist_ = atr_val_ * ATR_FLOOR_MULT
                if risk_dist_ < min_risk_dist_:
                    risk_dist_ = min_risk_dist_
                    sl_ = entry_ - risk_dist_ if side_ == "BUY" else entry_ + risk_dist_
            if risk_dist_ <= 0:
                return None

            # [新增] 手續費佔比過濾器：見上方 MIN_RISK_PCT_OF_PRICE 註解
            # 這是v2遺漏的一塊——固定風險sizing標準化了risk_usdt，但沒限制
            # risk_dist佔價格的比例，導致手續費占比仍可能系統性過高
            if (risk_dist_ / entry_) < MIN_RISK_PCT_OF_PRICE:
                return None

            targets_ = build_liquidity_targets(row, side_, entry_, risk_dist_)
            while len(targets_) < 3:
                mult_ = [1.8, 2.8, 4.0][len(targets_)]
                targets_.append(entry_ + risk_dist_ * mult_ if side_ == "BUY" else entry_ - risk_dist_ * mult_)
            tp1_, tp2_, tp3_ = targets_[:3]

            # [重構] 固定風險sizing：qty反推自TARGET_RISK_USDT / risk_dist，
            # risk_usdt因此恆等於TARGET_RISK_USDT（除非被槓桿上下限擋掉）
            qty_       = TARGET_RISK_USDT / risk_dist_
            notional_  = qty_ * entry_
            if notional_ > MAX_NOTIONAL_USDT or notional_ < MIN_NOTIONAL_USDT:
                return None  # 槓桿超出合理範圍，跳過（取代舊版的MIN_RISK_USDT硬門檻）

            open_fee_  = notional_ * FEE_RATE
            risk_usdt_ = risk_dist_ * qty_

            return {
                "symbol": symbol, "side": side_, "entry_price": entry_, "sl": sl_,
                "tp1": tp1_, "tp2": tp2_, "tp3": tp3_, "tp_hit": 0, "hedged": False,
                "initial_qty": qty_, "remaining_qty": qty_, "realized_pnl": -open_fee_,
                "fee_paid": open_fee_, "fund_paid": 0.0, "entry_time": t, "risk_usdt": risk_usdt_,
                "structure_tf": tf_, "structure_type": event_type_,
                "worst_price": entry_, "best_price": entry_,
                "ob_reused": ob_reused_, "retest_mode": "wick" if ENABLE_WICK_RETEST else "close",
                # [診斷新增]
                "swing_amplitude": swing_amp_, "amplitude_atr_ratio": amp_ratio_,
                "bars_since_prev_event": bars_since_prev_, "direction_flip": direction_flip_,
                "entry_bar_index": i, "ob_key": ob_key_,
            }

        # 1) 先處理前一根已經retest、正在等這一根確認的pending
        for pkey in list(pending.keys()):
            pinfo = pending[pkey]
            pside = pinfo["side"]
            confirmed = (pside == "BUY"  and close > pinfo["retest_close"]) or \
                        (pside == "SELL" and close < pinfo["retest_close"])
            if confirmed:
                atr_val = float(row["atr"]) if pinfo["tf"] == "5m" else get_atr_for(t, pinfo["tf"])
                atr_pctl_val = float(row["atr_pctl"]) if pinfo["tf"] == "5m" else get_atr_pctl_for(t, pinfo["tf"])
                new_pos = try_open(pside, pinfo["ob_low"], pinfo["ob_high"], pinfo["tf"],
                                   pinfo["event_type"], close, atr_val,
                                   swing_amp_=pinfo.get("swing_amplitude", np.nan),
                                   amp_ratio_=pinfo.get("amplitude_atr_ratio", np.nan),
                                   bars_since_prev_=pinfo.get("bars_since_prev_event", np.nan),
                                   direction_flip_=pinfo.get("direction_flip"),
                                   ob_key_=pkey, atr_pctl_=atr_pctl_val,
                                   ob_reused_=(pkey in seen_ob_keys))
                if new_pos is not None:
                    positions.append(new_pos)
                    used_obs.add(pkey)
                    seen_ob_keys.add(pkey)
            del pending[pkey]  # 不論成功與否，這個pending都消耗掉，要重新retest才會再等一次

        # 2) 再檢查這一根是不是新的retest（進pending，下一根才真正進場）
        event, tf = get_active_event(t)
        if event is None:
            continue

        age_ms = t - int(event["time"])
        bar_min = 5 if tf == "5m" else 15
        max_age_ms = OB_MAX_AGE_BARS * bar_min * 60 * 1000
        key = (tf, int(event["ob_time"]))

        if age_ms > max_age_ms or key in used_obs or key in pending:
            continue

        ob_low, ob_high = float(event["ob_low"]), float(event["ob_high"])
        side = "BUY" if event["direction"] == "bullish" else "SELL"

        # [新增] 即時進場消融分支：見上方 ENABLE_IMMEDIATE_ENTRY 註解。
        # age_ms==0 代表這是這個事件第一次變成「當前有效事件」的那根K棒，
        # 跳過retest+下一根確認的流程，直接在這根K棒收盤進場。
        if ENABLE_IMMEDIATE_ENTRY:
            if age_ms == 0 and key not in used_obs:
                atr_val = float(row["atr"]) if tf == "5m" else get_atr_for(t, tf)
                atr_pctl_val = float(row["atr_pctl"]) if tf == "5m" else get_atr_pctl_for(t, tf)
                new_pos = try_open(side, ob_low, ob_high, tf, event["type"], close, atr_val,
                                   swing_amp_=event.get("swing_amplitude", np.nan),
                                   amp_ratio_=event.get("amplitude_atr_ratio", np.nan),
                                   bars_since_prev_=event.get("bars_since_prev_event", np.nan),
                                   direction_flip_=event.get("direction_flip"),
                                   ob_key_=key, atr_pctl_=atr_pctl_val,
                                   ob_reused_=(key in seen_ob_keys))
                if new_pos is not None:
                    positions.append(new_pos)
                    used_obs.add(key)
                    seen_ob_keys.add(key)
            continue  # 這個模式下完全跳過retest+pending的流程

        # [改為可切換] ENABLE_WICK_RETEST=True 時用影線觸及即算retest；
        # False 時退回原本「收盤價須落在OB區間內」的嚴格版本，方便消融測試比較。
        if ENABLE_WICK_RETEST:
            retest = (low <= ob_high) and (high >= ob_low)
        else:
            retest = ob_low <= close <= ob_high
        if not retest:
            continue

        # [新增，依diagnose5發現] OB觸碰次數過濾：見上方MIN_OB_TOUCHES註解
        touch_counts[key] = touch_counts.get(key, 0) + 1
        if ENABLE_OB_TOUCH_FILTER and touch_counts[key] < MIN_OB_TOUCHES:
            continue  # 這個OB還沒被驗證過，這次只記錄觸碰，不進場

        pending[key] = {
            "side": side, "ob_low": ob_low, "ob_high": ob_high,
            "tf": tf, "event_type": event["type"], "retest_close": close,
            # [診斷新增]：把事件本身的振幅與「跟上一個同時框事件的間隔/是否翻方向」帶過去，
            # 等confirm時一起存進position（用K棒數量而非固定時長，避免視窗塞不進一個pivot週期）
            "swing_amplitude": event.get("swing_amplitude", np.nan),
            "amplitude_atr_ratio": event.get("amplitude_atr_ratio", np.nan),
            "bars_since_prev_event": event.get("bars_since_prev_event", np.nan),
            "direction_flip": event.get("direction_flip"),
        }

    return trades, df5


def _make_trade(symbol, pos, exit_time, exit_price, exit_reason, net_pnl, fee_total, fund_total,
                 exit_bar_index=None):
    risk = pos["risk_usdt"] if pos["risk_usdt"] > 0 else 1e-9
    bars_held = (exit_bar_index - pos["entry_bar_index"]) if exit_bar_index is not None else np.nan
    r_multiple = net_pnl / risk

    # [新增] MAE/MFE：進場以來最不利/最有利的價格偏移，換算成R倍數，
    # 直接證明停損距離設定是否合理，不用再靠RISK分位數間接推論
    side = pos["side"]
    if side == "BUY":
        mae = pos["entry_price"] - pos["worst_price"]
        mfe = pos["best_price"] - pos["entry_price"]
    else:
        mae = pos["worst_price"] - pos["entry_price"]
        mfe = pos["entry_price"] - pos["best_price"]
    # 用「價格距離 / risk_dist」表示更直觀：risk_dist = risk_usdt / initial_qty
    risk_dist = risk / pos["initial_qty"] if pos["initial_qty"] else np.nan
    mae_r = mae / risk_dist if risk_dist else np.nan
    mfe_r = mfe / risk_dist if risk_dist else np.nan

    # [新增] 手續費佔風險比例：直接算出來放進CSV驗證MIN_RISK_PCT_OF_PRICE過濾器有沒有生效，
    # 不用再靠總表數字回推
    fee_pct_of_risk = fee_total / risk if risk else np.nan

    return {
        "Time"      : datetime.fromtimestamp(pos["entry_time"] / 1000).strftime("%Y/%m/%d %H:%M"),
        "Symbol"    : symbol,
        "Direction" : "long" if pos["side"] == "BUY" else "short",
        "Entry"     : round(pos["entry_price"], 4),
        "Stop"      : round(pos["sl"], 4),
        "Target"    : round(pos["tp1"], 4) if pos["tp1"] else "-",
        "Exit"      : round(exit_price, 4),
        "Size"      : round(pos["initial_qty"], 6),
        "Notional"  : round(pos["entry_price"] * pos["initial_qty"], 4),
        "進場原因"   : f"{pos['structure_tf']} {pos['structure_type']} 結構後OB回測進場",
        "出場原因"   : exit_reason,
        "RISK"      : round(risk, 4),
        "Result"    : round(net_pnl, 4),
        "R_Multiple": round(r_multiple, 4),
        "MAE_R"     : round(mae_r, 4) if not np.isnan(mae_r) else np.nan,
        "MFE_R"     : round(mfe_r, 4) if not np.isnan(mfe_r) else np.nan,
        "fee_pct_of_risk": round(fee_pct_of_risk, 4) if not np.isnan(fee_pct_of_risk) else np.nan,
        "Hedged?"   : "Y" if pos["hedged"] else "N",
        "tp_hit"    : pos["tp_hit"],
        "exit_time" : datetime.fromtimestamp(exit_time / 1000).strftime("%Y/%m/%d %H:%M"),
        "fee_total" : round(fee_total, 4),
        "fund_total": round(fund_total, 4),
        # [診斷新增]
        "structure_tf"        : pos["structure_tf"],
        "structure_type"      : pos["structure_type"],
        "swing_amplitude"     : pos.get("swing_amplitude", np.nan),
        "amplitude_atr_ratio" : pos.get("amplitude_atr_ratio", np.nan),
        "bars_since_prev_event": pos.get("bars_since_prev_event", np.nan),
        "direction_flip"      : pos.get("direction_flip"),
        "bars_held"           : bars_held,
        "ob_reused"           : pos.get("ob_reused", False),
        "retest_mode"         : pos.get("retest_mode", ""),
    }

# ----------------------------------------------
# 統計與輸出（與 v21 相同，未變更）
# ----------------------------------------------

def compute_stats(df_all):
    df_all = df_all.sort_values("Time").reset_index(drop=True)
    df_all["cum_pnl"] = df_all["Result"].cumsum()
    total  = len(df_all)
    wins   = df_all[df_all["Result"] > 0]
    losses = df_all[df_all["Result"] <= 0]
    win_rate = len(wins) / total * 100 if total else 0
    not_hedged = df_all[df_all["Hedged?"] == "N"]
    avg_r_no_hedge = not_hedged["R_Multiple"].mean() if len(not_hedged) else 0
    hedged = df_all[df_all["Hedged?"] == "Y"]
    hedged_count = len(hedged)
    avg_r_hedged = hedged["R_Multiple"].mean() if hedged_count else 0
    avg_win  = wins["Result"].mean()   if len(wins)   else 0
    avg_loss = losses["Result"].mean() if len(losses) else 0
    avg_rr   = abs(avg_win / avg_loss) if avg_loss != 0 else float("nan")
    returns = df_all["Result"]
    sharpe = returns.mean() / returns.std() * np.sqrt(len(returns)) if returns.std() > 0 else float("nan")
    cum = df_all["cum_pnl"]
    max_dd = (cum - cum.cummax()).min()
    return {
        "total": total, "win_rate": win_rate, "avg_r_no_hedge": avg_r_no_hedge,
        "avg_win": avg_win, "avg_loss": avg_loss, "avg_rr": avg_rr, "sharpe": sharpe,
        "max_dd": max_dd, "hedged_count": hedged_count, "avg_r_hedged": avg_r_hedged,
        "final_cum_pnl": cum.iloc[-1] if total else 0,
    }

def print_results(all_trades, dfs):
    if not all_trades:
        print("無交易信號")
        return
    df_all = pd.DataFrame(all_trades)
    stats  = compute_stats(df_all)
    sym_counts = df_all.groupby("Symbol").size().to_dict()
    total_fee  = df_all["fee_total"].sum()
    total_fund = df_all["fund_total"].sum()

    print(f"\n{'='*70}")
    print(f"{' / '.join(SYMBOLS)}  ICT/SMC結構+OB回測進場  流動性關鍵位分批止盈 [smc_v2]")
    print(f"風控：固定風險 {TARGET_RISK_USDT} USDT/筆（動態槓桿，名目上限{MAX_NOTIONAL_USDT} USDT/"
          f"下限{MIN_NOTIONAL_USDT} USDT）  手續費：VIP0 {FEE_RATE*100}%  資金費率：{FUND_RATE*100}%/8h"
          f"  滑價：{SLIPPAGE_BPS}bps")
    print(f"{'='*70}")
    for sym in SYMBOLS:
        n       = sym_counts.get(sym, 0)
        sym_net = df_all[df_all["Symbol"] == sym]["Result"].sum()
        sym_l   = (df_all[df_all["Symbol"] == sym]["Direction"] == "long").sum()
        sym_s   = (df_all[df_all["Symbol"] == sym]["Direction"] == "short").sum()
        print(f"  {sym:<10} 交易 {n:>4} 次（多 {sym_l} / 空 {sym_s}）  淨盈虧 {sym_net:>+.4f} USDT")
    print(f"{'─'*70}")
    print(f"總交易次數      : {stats['total']}")
    print(f"累積 PnL        : {stats['final_cum_pnl']:+.4f} USDT")
    print(f"勝率(%)         : {stats['win_rate']:.1f}%")
    print(f"平均R倍數(不含套保): {stats['avg_r_no_hedge']:+.4f}")
    print(f"平均盈利單      : {stats['avg_win']:+.4f} USDT")
    print(f"平均虧損單      : {stats['avg_loss']:+.4f} USDT")
    print(f"平均盈虧比      : {stats['avg_rr']:.2f}")
    print(f"夏普比率        : {stats['sharpe']:.3f}")
    print(f"最大回撤        : {stats['max_dd']:.4f} USDT")
    print(f"套保策略數      : {stats['hedged_count']}")
    print(f"套保後平均R倍數 : {stats['avg_r_hedged']:+.4f}")
    print(f"{'─'*70}")
    print(f"總手續費        : -{total_fee:.4f} USDT")
    print(f"總資金費率      : -{total_fund:.4f} USDT")
    print(f"{'='*70}")

    out = "backtest_result_smc_v1.csv"
    cols = ["Time","Symbol","Direction","Entry","Stop","Target","Exit","Size","Notional",
            "進場原因","出場原因","RISK","Result","R_Multiple","MAE_R","MFE_R","fee_pct_of_risk",
            "Hedged?","tp_hit",
            "structure_tf","structure_type","swing_amplitude","amplitude_atr_ratio",
            "bars_since_prev_event","direction_flip","bars_held","ob_reused","retest_mode"]
    try:
        df_all[cols].to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n結果已存至 {out}")
    except PermissionError:
        print(f"\n[!] {out} 被佔用，請關閉 Excel")

# ----------------------------------------------
# 主程序
# ----------------------------------------------

if __name__ == "__main__":
    all_trades = []
    dfs        = {}
    for sym in SYMBOLS:
        print(f"\n=== {sym} ===")
        df5  = fetch_all_klines(sym, "5m",  PAGES_5M)
        df15 = fetch_all_klines(sym, "15m", PAGES_15M)
        print(f"  回測中...")
        trades, df5 = backtest_symbol(sym, df5, df15)
        all_trades.extend(trades)
        dfs[sym] = df5
        print(f"  {sym} 完成，{len(trades)} 筆交易")
    print_results(all_trades, dfs)