import redis, json, requests, time

MODEL_SERVER_URL = "http://localhost:8001/answer"  # modelServer.py가 띄운 API 주소
import loggerModule

logger = loggerModule.getLogger('worker');

RETRY_MAX = 3
RETRY_DELAY_SEC = 2
REQUEST_TIMEOUT = 10

try:
    #r = redis.Redis(host="localhost", port=6379, db=0)
    pool = redis.ConnectionPool(host="localhost", port=6379, db=0, max_connections=20)
    r = redis.Redis(connection_pool=pool)
except redis.ConnectionError:
    logger.error('Redis 서버에 연결할 수 없습니다.')
    raise RuntimeError("Redis 서버에 연결할 수 없습니다.")

def run():
    while True:
        _, task_json = r.brpop("chat_queue")
        task = json.loads(task_json)
        session_id = task["session_id"]
        prompt = task["prompt"]

        for attempt in range(1, RETRY_MAX + 1):
            try:
                response = requests.post(
                    MODEL_SERVER_URL,
                    json={"session_id": session_id, "prompt": prompt},
                    timeout=REQUEST_TIMEOUT
                )
                response.raise_for_status()
                break
            except requests.RequestException as e:
                logger.error(f"모델 서버 호출 실패 (시도 {attempt}/{RETRY_MAX}): {e}")
                if attempt >= RETRY_MAX:
                    logger.error(f"작업 처리 실패, session_id={session_id}")
                else:
                    time.sleep(RETRY_DELAY_SEC * attempt)
                    continue
