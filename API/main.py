import asyncio
from KIS.REST.env import KIS_APP_KEY
from KIS.REST.env import KIS_APP_SECRET
from KIS.REST.getAccessToken import fetch_token
from KIS.REST.dailyOhlcv import fetch_daily_ohlcv

async def main():
    #access토큰 발행 테스트
    #token = await fetch_token(APIKEY, SECRETKEY)
    print('token')

if __name__ == "__main__":
    asyncio.run(main())