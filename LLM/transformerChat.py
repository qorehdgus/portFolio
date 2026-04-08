
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
import threading
import torch
import re
import redis
import json

app = FastAPI()

try:
    r = redis.Redis(host="localhost", port=6379, db=0)
except redis.ConnectionError:
    raise RuntimeError("Redis 서버에 연결할 수 없습니다.")

def save_message(session_id, role, content):
    try:
        message = {"role": role, "content": content}
        r.rpush(f"chat:{session_id}", json.dumps(message))
    except Exception as e:
        # Redis 오류 처리
        print(f"Redis 저장 오류: {e}")

def load_chat(session_id):
    try:
        messages = r.lrange(f"chat:{session_id}", 0, -1)
        return [json.loads(m) for m in messages]
    except Exception as e:
        print(f"Redis 불러오기 오류: {e}")
        return []

try:
    # 모델 이름 (Hugging Face Hub에 올라온 모델 경로로 교체)
    #model_name = "C:/Users/user/Desktop/github/portFolio/LLM/model/llama-3.2-Korean-Bllossom-3B"
    model_name = "C:/Users/user/Desktop/github/portFolio/LLM/model/llama-3-Korean-Bllossom-8B"

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
except Exception as e:
    raise RuntimeError(f"모델 로딩 실패: {e}")

@app.get("/chat")
def chat_endpoint(session_id: str, prompt: str):
    try:
        # Redis에서 기존 대화 불러오기
        chat_history = load_chat(session_id)

        # 저장된 내용이 없으면 기본 system 메시지 추가
        #print(chat_history)
        #print(not chat_history)
        if not chat_history:
            chat_history = [
                {"role": "system", "content": "너는 한글로 답변하고 짧고 간결하게 답변하는 챗봇이야."},
                {"role": "system", "content": "너는 👉이런 이모지를 적극 활용하여 답변하는 챗봇이야."}
            ]
        
        # 새 유저 입력 추가
        chat_history.append({"role": "user", "content": prompt})
        # 새 메시지 저장
        save_message(session_id, "user", prompt)

        # 모델 입력 생성
        inputs = tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        model_inputs = tokenizer([inputs], return_tensors="pt").to(model.device)
        #inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 스트리머 생성
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

        terminators = [
            tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # 모델 실행을 별도 스레드에서 수행
        generation_kwargs = dict(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            eos_token_id=terminators,
            repetition_penalty=1.2,
            streamer=streamer
        )
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        def event_stream():
            response_text = ""
            assistant_check = False
            for token in streamer:
                if "assistant" in token:
                    assistant_check = True
                    continue  
                if(assistant_check):
                    response_text += token
                    yield f"{token}"
                    
            # 최종 응답 Redis에 저장
            save_message(session_id, "assistant", response_text)
            yield " [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")