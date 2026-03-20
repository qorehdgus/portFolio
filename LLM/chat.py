
#from transformers import AutoModelForCausalLM, AutoTokenizer
#from ctransformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer
import onnxruntime as ort
import websockets
import numpy as np
import asyncio
import json

#local_model_path="C:/Users/dhbaek/Desktop/project/LLM/model/hyperClova0.5B"
#local_model_path="C:/Users/dhbaek/Desktop/project/LLM/model/koGPT2"
#local_model_path="C:/Users/dhbaek/Desktop/project/LLM/model/koLlama"
local_model_path="C:/Users/dhbaek/Desktop/project/LLM/model/hyperClova0.5B/onnx_fp32_hyperclova/model.onnx"
local_tokenizer_path="C:/Users/dhbaek/Desktop/project/LLM/model/hyperClova0.5B/onnx_fp32_hyperclova"
#local_model_path="C:/Users/dhbaek/Desktop/project/LLM/model/hyperClova1.5B/onnx_fp32_hyperClova1.5B/model.onnx"
#local_tokenizer_path="C:/Users/dhbaek/Desktop/project/LLM/model/hyperClova1.5B/onnx_fp32_hyperClova1.5B"

session = ort.InferenceSession(local_model_path)
print("ONNX 모델 로딩 성공:", session)

#내장 그래픽에서는 pytorch의 CUDA 지원 불가하여 cpu로만 구동
#model = AutoModelForCausalLM.from_pretrained(local_model_path, device_map="auto")
#model = AutoModelForCausalLM.from_pretrained(local_model_path).to("cpu")
#양자화시킨 모델을 사용하려 했으나 변환 스크립트가 LLAMA모델만 지원하여 우선 중단

#model = AutoModelForCausalLM.from_pretrained(
#    local_model_path,
#    model_file="koLlama.gguf",   # 변환된 파일
#    local_files_only=True
#)

tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)

#def chat(prompt):
#    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
#    outputs = model.generate(**inputs, max_length=200)
#    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def sample_next_token(logits, temperature=1.0, top_k=50, top_p=0.9):
    # 마지막 위치의 로짓
    logits = logits[0, -1]

    # temperature 적용
    logits = logits / max(temperature, 1e-6)

    # 안정화된 softmax
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)

    # top-k 필터링
    if top_k > 0:
        top_indices = np.argsort(probs)[-top_k:]
        mask = np.ones_like(probs, dtype=bool)
        mask[top_indices] = False
        probs[mask] = 0
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            return int(np.argmax(logits))

    # top-p 필터링
    if top_p < 1.0:
        sorted_indices = np.argsort(-probs)
        cumulative_probs = np.cumsum(probs[sorted_indices])
        cutoff = cumulative_probs > top_p
        probs[sorted_indices[cutoff]] = 0
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            return int(np.argmax(logits))

    # 샘플링
    return int(np.random.choice(len(probs), p=probs))

#prompt: 사용자 입력문장
#max_new_tokens: 생성 최대 토큰수
#temperature: 샘플링 온도(낮으면 정확, 높으면 창의)
#top_k, top_p: 샘플링 전략
async def chat_handler(websocket):

    #print(path)
    prompt = await websocket.recv()

    max_new_tokens=500
    temperature=1.0
    top_k=50
    top_p=0.9
    
    chat = [
        {"role": "tool_list", "content": "[{'name':'get_weather','description':'특정 도시의 현재 날씨를 조회하는 도구','input_schema':{'type':'object','arguments':{'city':{'type':'string'}}}}]"}, 
        {"role": "system", "content": "너는 assistant 역할로만 답변하며, 새로운 user 질문을 생성하지 않는다.\n assistant 답변은 반드시 하나만 생성한다."},
        {"role": "user", "content": prompt},
    ]
    
    inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_dict=True, return_tensors="np")
    #full_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in chat])
    #inputs = tokenizer(full_prompt, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)

    for _ in range(max_new_tokens):
        position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[None, :]
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }

        outputs = session.run(None, ort_inputs)
        logits = outputs[0]

        next_token_id = sample_next_token(logits, temperature, top_k, top_p)

        if next_token_id == tokenizer.eos_token_id:
            break

        # 토큰을 바로 클라이언트로 전송
        token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
        
        await websocket.send(token_text)
        await asyncio.sleep(0.01)  # 아주 짧은 지연

        input_ids = np.concatenate([input_ids, [[next_token_id]]], axis=1)
        attention_mask = np.concatenate([attention_mask, [[1]]], axis=1)

    #return tokenizer.decode(input_ids[0], skip_special_tokens=True)


async def main():
    async with websockets.serve(chat_handler, "localhost", 8765):
        print("WebSocket server started at ws://localhost:8765")
        await asyncio.Future()

asyncio.run(main())