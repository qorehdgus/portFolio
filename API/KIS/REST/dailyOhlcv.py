from KIS.REST.env import KIS_APP_KEY
from KIS.REST.env import KIS_APP_SECRET
from KIS.REST.env import KIS_ACCESS_TOKEN
from KIS.REST.env import BASE

import pandas as pd
import httpx


async def fetch_daily_ohlcv(stock_code: str, days: int = 200) -> pd.DataFrame:
    """
    국내 주식 일봉 데이터 조회

    Args:
        stock_code: 6자리 종목코드 (예: "005930" - 삼성전자)
        days: 조회할 일수 (최대 200일)

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    #await ensure_token()

    url = f"{BASE}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    headers = {
        "authorization": f"Bearer {KIS_ACCESS_TOKEN}",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET,
        "tr_id": "FHKST03010100",
        "custtype": "P",
    }
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",  # J: 전체(코스피+코스닥)
        "FID_INPUT_ISCD": stock_code,
        "FID_INPUT_DATE_1": "",  # 시작일 (빈 값이면 오늘부터)
        "FID_INPUT_DATE_2": "",  # 종료일
        "FID_PERIOD_DIV_CODE": "D",  # D: 일봉
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)

    data = response.json()
    if data["rt_cd"] != "0":
        raise RuntimeError(f"API 오류: {data['msg1']}")

    # 응답 데이터를 DataFrame으로 변환
    output = data["output2"]
    df = pd.DataFrame(output)
    df.columns = ["date", "open", "high", "low", "close", "volume", "value"]

    # 데이터 타입 변환
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df.sort_values("date").reset_index(drop=True).tail(days)