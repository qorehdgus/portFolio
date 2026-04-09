from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import redis, json

app = FastAPI()

# 허용할 origin 목록 (개발 중이라면 * 로 전체 허용 가능)
origins = [
    "http://localhost:9090",  # JS 클라이언트가 실행되는 주소
    "http://127.0.0.1:9090",
    "*"  # 개발 중에는 전체 허용, 운영에서는 특정 도메인만 허용
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    #r = redis.Redis(host="localhost", port=6379, db=0)
    pool = redis.ConnectionPool(host="localhost", port=6379, db=0, max_connections=20)
    r = redis.Redis(connection_pool=pool)
except redis.ConnectionError:
    raise RuntimeError("Redis 서버에 연결할 수 없습니다.")

from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_id: str
    prompt: str

@app.post("/chat")
def chat_request(req: ChatRequest):
    task = {"session_id": req.session_id, "prompt": req.prompt}
    r.lpush("chat_queue", json.dumps(task))
    return {"status": "queued", "session_id": req.session_id}

@app.get("/stream")
def stream(session_id: str):
    def event_generator():
        pubsub = r.pubsub()
        pubsub.subscribe(f"stream:{session_id}")
        for message in pubsub.listen():
            if message["type"] == "message":
                yield f"data: {message['data'].decode()}\n\n"
                if message["data"].decode() == "[DONE]":
                    break
    return StreamingResponse(event_generator(), media_type="text/event-stream")
