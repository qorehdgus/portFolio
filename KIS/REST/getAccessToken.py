import httpx

BASE = "https://openapi.koreainvestment.com:9443"

async def fetch_token(app_key: str, app_secret: str) -> str:
    """KIS API 토큰 발급"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE}/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "appkey": app_key,
                "appsecret": app_secret,
            },
            timeout=5,
        )
    result = response.json()
    return result["access_token"]
