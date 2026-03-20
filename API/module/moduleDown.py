from huggingface_hub import snapshot_download

from huggingface_hub import login
login("")  # 권한 있는 토큰 입력

model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"

snapshot_download(
    repo_id=model_name,  # 원하는 모델 이름
    local_dir="C:\Users\dhbaek\Desktop\project\LLM\model\naver"  # 원하는 저장 경로
)

