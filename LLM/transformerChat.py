from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import re

# 모델 이름 (Hugging Face Hub에 올라온 모델 경로로 교체)
model_name = "C:/Users/user/Desktop/github/portFolio/LLM/model/SmolLM3-3B"

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="cuda",
    quantization_config=bnb_config,  # 여기서 load_in_4bit 대신 quantization_config 사용
    dtype="auto"
)

chat = [
    {"role": "system", "content": "/no_think"},
    {"role": "system", "content": "너는 한글로 답변하고 짧고 간결하게 답변하는 챗봇이야."}
]

# 대화 루프
print("Chatbot ready! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    chat.append({"role": "user", "content": user_input})

    # 입력 토큰화
    #inputs = tokenizer(chat, return_tensors="pt").to(model.device)



    inputs = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([inputs], return_tensors="pt").to(model.device)


    # 모델 응답 생성
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=32768,
        do_sample=True,
        temperature=0.7,   # 다양성 조절
        top_p=0.9,         # nucleus sampling
        repetition_penalty=1.2  # 반복 억제
    )
    #response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # <think> 태그 제거
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    # assistant 이후 텍스트만 추출
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()

    chat.append({"role": "assistant", "content": response})
    print("Bot:", response)