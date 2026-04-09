import redis, json, requests

MODEL_SERVER_URL = "http://localhost:8001/answer"  # modelServer.py가 띄운 API 주소
import loggerModule

logger = loggerModule.getLogger('worker');

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

        try:
            # 모델 서버에 요청 보내기
            response = requests.post(
                MODEL_SERVER_URL,
                params={"session_id": session_id, "prompt": prompt}
            )
            if response.status_code != 200:
                print(f"모델 서버 호출 실패: {response.text}")
        except Exception as e:
            logger.error(f"워커 오류: {e}")

            print(f"워커 오류: {e}")
