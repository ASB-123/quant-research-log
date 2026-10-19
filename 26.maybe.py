"""
scan_bitunix_funding_intervals.py
====================
[工具] 掃描 Bitunix 全部永續合約的實際資金費結算週期，找出非標準8小時的幣種

背景：官方說明有提到「資金費通常每8小時收取一次，但Bitunix會依市場狀況對
部分交易對調整資金費結算時間」。這支工具就是去查「到底哪些幣不是8小時」，
給你評估要不要把這些幣納入 funding_reversal / funding_cash_carry 回測。

用到的兩支API（都已在官方文件核實過欄位）：
  1. GET /api/v1/futures/market/trading_pairs（不帶symbols參數＝抓全部交易對）
     用來取得目前 OPEN 狀態的所有永續合約清單。
  2. GET /api/v1/futures/market/funding_rate?symbol=XXX
     回傳裡直接有 fundingInterval（int32，單位：小時）這個欄位，不用像
     resettlement time反推，官方就是這樣給的。

[⚠ 注意]
  1. Rate limit：兩支API文件都寫 10 req/sec/ip，這支工具用逐一symbol查詢
     funding_rate，數量可能有上百個，程式裡有加sleep節流，不要調快。
  2. fundingInterval 是「目前」的結算週期設定，不是歷史上固定不變的——
     Bitunix可能因為市場波動臨時調整某幣的週期，所以這只能告訴你「現在」
     是多久一次，不保證回測歷史區間內全程都是這個週期。如果你要拿某個
     1小時結算的幣去跑回測，抓到的歷史資金費率資料本身（get_funding_rate_
     history）的實際筆數間隔，才是回測時真正該信賴的依據，這支工具只是
     「篩選候選名單」用的第一步。
  3. 只篩 symbolStatus == "OPEN" 的幣（可以正常交易的），CANCEL_ONLY /
     STOP 狀態的不列入，避免你選到快下架或已停牌的幣。
"""

import requests
import pandas as pd
import time

FUTURES_BASE_URL = "https://fapi.bitunix.com"
REQ_INTERVAL_SEC = 0.15   # 10 req/sec上限，抓保守一點的間隔避免被限流


def get_all_trading_pairs():
    """不帶symbols參數，抓全部永續合約清單"""
    res = requests.get(f"{FUTURES_BASE_URL}/api/v1/futures/market/trading_pairs", timeout=15)
    data = res.json()
    if data.get("code") not in (0, "0"):
        raise RuntimeError(f"trading_pairs 查詢失敗：{data}")
    pairs = data.get("data", [])
    open_pairs = [p["symbol"] for p in pairs if p.get("symbolStatus") == "OPEN"]
    print(f"  [trading_pairs] 共 {len(pairs)} 個交易對，其中 OPEN 狀態 {len(open_pairs)} 個")
    return open_pairs


def get_funding_interval(symbol):
    """查單一symbol目前的資金費結算週期(小時)"""
    for attempt in range(3):
        try:
            res = requests.get(
                f"{FUTURES_BASE_URL}/api/v1/futures/market/funding_rate",
                params={"symbol": symbol}, timeout=10)
            data = res.json()
            if data.get("code") in (0, "0") and data.get("data"):
                row = data["data"][0] if isinstance(data["data"], list) else data["data"]
                return {
                    "symbol": symbol,
                    "fundingInterval_hr": int(row.get("fundingInterval", 8)),
                    "fundingRate": float(row.get("fundingRate", 0)),
                    "nextFundingTime": row.get("nextFundingTime"),
                }
            return None
        except Exception as e:
            print(f"\n  [{symbol}] 查詢失敗（第{attempt+1}次）：{e}")
            time.sleep(1)
    return None

def calc_indicators(df):

    ma20 = df['close'].rolling(20).mean()
    ma60 = df['close'].rolling(60).mean()
    ma120 = df['close'].rolling(120).mean()

    ema20 = df['close'].ewm(span= 20, adjust= False).mean()
    ema60 = df['close'].ewm(span= 60, adjust= False).mean()
    ema120 = df['close'].ewm(span= 120, adjust= False).mean()

    ema12 = df['close'].ewm(span= 12, adjust= False).mean()
    ema26 = df['close'].ewm(span= 26, adjust= False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span= 9, adjust= False).mean()
    hist = dif - dea

    return {
        "ema20_now"  : ema20.iloc[-1],
        "ema60_now"  : ema60.iloc[-1],
        "ema120_now" : ema120.iloc[-1],
        "hist_now"   : hist.iloc[-1],
        "hist_prev"  : hist.iloc[-2],
    }

def checkcondition(ind):
    maalligened = ind["ema20now"] > ind["ema60now"] > ind["ema120now"]
    macdcross = ind["histprev"] < 0 and ind["histnow"] > 0
    return maalligened and macdcross

def main():
    print("=== 掃描 Bitunix 全部永續合約的資金費結算週期 ===")
    symbols = get_all_trading_pairs()

    results = []
    for i, sym in enumerate(symbols):
        info = get_funding_interval(sym)
        if info:
            results.append(info)
        print(f"  進度 {i+1}/{len(symbols)}  當前: {sym}", end="\r")
        time.sleep(REQ_INTERVAL_SEC)

    print(f"\n\n完成，成功取得 {len(results)}/{len(symbols)} 個幣的結算週期資料")

    df = pd.DataFrame(results)
    df = df.sort_values(["fundingInterval_hr", "symbol"]).reset_index(drop=True)
    df.to_csv("bitunix_funding_intervals_scan.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print("結算週期分佈")
    print(f"{'='*60}")
    dist = df["fundingInterval_hr"].value_counts().sort_index()
    for hr, cnt in dist.items():
        print(f"  每 {hr} 小時結算一次：{cnt} 個幣")

    non_8h = df[df["fundingInterval_hr"] != 8]
    print(f"\n{'='*60}")
    print(f"非8小時結算的幣種（共 {len(non_8h)} 個）")
    print(f"{'='*60}")
    if len(non_8h):
        print(non_8h[["symbol", "fundingInterval_hr", "fundingRate"]].to_string(index=False))
    else:
        print("  沒有找到——目前所有OPEN幣種都是標準8小時結算")

    print(f"\n完整清單已存至 bitunix_funding_intervals_scan.csv")
    print("\n[下一步建議] 從非8小時清單中，優先看 fundingInterval_hr 較小（如1或4小時）")
    print("且 fundingRate 絕對值較大的幣，這種通常是波動大、資金費率訊號較活躍的標的，")
    print("但務必同時查一下該幣現貨市場的流動性/深度，避免抓到雖然資金費率吸睛、")
    print("但現貨盤口太薄、實際下單滑價會吃光套利空間的幣。")

if __name__ == "__main__":
    main()