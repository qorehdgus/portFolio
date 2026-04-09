import logging

def getLogger(worker_id):
    logger = logging.getLogger(f"worker-{worker_id}")
    handler = logging.FileHandler(f"worker_{worker_id}.log")
    formatter = logging.Formatter(
        "%(asctime)s [%(processName)s] %(levelname)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger