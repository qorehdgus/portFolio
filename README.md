# portFolio
포트폴리오
깃허브 연동 테스트

26.04.07 수행 기록<br/>
데스크탑 교체 >> 12vram의 GPU를 활용 가능하여 모델 변경<br/>
SmolLM3-3B 모델로 선정<br/>
1) Vram 용량으로 인해 3B모델 선정 (양자화를 4int로 수행하였으나 안전성을 고려)<br/>
2) 최대 128K 토큰을 지원한다고 함. (추후 뉴스 분석 등 토큰 다수 활용 대비)<br/>
3) 한국어 지원<br/>
https://huggingface.co/HuggingFaceTB/SmolLM3-3B

* llm_int8_enable_fp32_cpu_offload=True 옵션 제거<br/>
CPU 오프로딩을 제거하고 GPU로만 구동하니 답변의 품질 향상을 확인.<br/>
데이터 이동이 없어 토큰 생성이 끊기지 않아서로 추측.<br/>

26.04.09 수행 기록<br/>
llama-3-Korean-Bllossom-8B 모델로 변경<br/>
1) 기존 모델인 SmolLM3-3B의 답변 능력이 떨어져 8B 모델로 변경<br/>
2) 4int 양자화 유지하였으며 모델을 로드할 경우 GPU전용 메모리의 약 7.8GB를 차지하는 것을 확인<br/>
