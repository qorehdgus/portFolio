import multiprocessing
import uvicorn
import signal
import sys
import workers
from controller import app as controller_app
from modelServer import create_app   # app 대신 create_app 가져오기
import loggerModule

processes = []

def start_worker():
    workers.run()

def start_api():
    uvicorn.run(controller_app, host="0.0.0.0", port=8000)

def modelServer_api():
    # create_app() 호출 시에만 모델 로드됨
    uvicorn.run(create_app(), host="0.0.0.0", port=8001)

def shutdown_handler(signum, frame):
    print(f"Received signal {signum}, shutting down...")
    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join()
    sys.exit(0)

logger = loggerModule.getLogger('main')

if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # 워커 실행 (필요 개수만)
    for i in range(1):
        logger.info(f"{i}번째 worker 동작")
        p = multiprocessing.Process(target=start_worker)
        p.start()
        processes.append(p)

    # 컨트롤러 API 실행
    api_proc = multiprocessing.Process(target=start_api)
    api_proc.start()
    processes.append(api_proc)
    logger.info("controller api 서버 동작")

    # 모델 서버 실행
    modelApi_proc = multiprocessing.Process(target=modelServer_api)
    modelApi_proc.start()
    processes.append(modelApi_proc)
    logger.info("model api 서버 동작")

    # 모든 프로세스 대기
    for p in processes:
        p.join()
