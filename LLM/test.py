import asyncio
import websockets

async def chat_handler(websocket, path):
    prompt = await websocket.recv()
    print(f"Client prompt: {prompt}")

    # 예시: 토큰을 하나씩 생성했다고 가정
    tokens = ["네", "이", "버", "란", " ", "뭐", "야"]

    for token in tokens:
        await websocket.send(token)
        await asyncio.sleep(0.2)  # 토큰 생성 지연을 흉내냄

    await websocket.send("[END]")  # 응답 종료 표시

async def main():
    async with websockets.serve(chat_handler, "localhost", 8765):
        print("WebSocket server started at ws://localhost:8765")
        await asyncio.Future()  # 서버 계속 실행

asyncio.run(main())
