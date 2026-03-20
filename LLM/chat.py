
from transformers import AutoTokenizer
import onnxruntime as ort
import websockets
import numpy as np
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import re
import redis


local_model_path="C:/Users/dhbaek/Desktop/project/LLM/model/hyperClova0.5B/onnx_fp32_hyperclova/model.onnx"
local_tokenizer_path="C:/Users/dhbaek/Desktop/project/LLM/model/hyperClova0.5B/onnx_fp32_hyperclova"

try:
    session_ort = ort.InferenceSession(
        local_model_path,
        providers=["DmlExecutionProvider"]
    )
    print("DirectML GPU 사용 중:", session_ort.get_providers())
except Exception as e:
    print("GPU 실패, CPU로 fallback:", e)
    session_ort = ort.InferenceSession(local_model_path, providers=["CPUExecutionProvider"])

#session_ort = ort.InferenceSession(local_model_path)
print("ONNX 모델 로딩 성공:", session_ort)

# ① MCP 서버 연결 설정
server_params = StdioServerParameters(
    command="C:/Users/dhbaek/Desktop/project/venv/MCP/Scripts/python.exe",
    args=["C:/Users/dhbaek/Desktop/project/MCP/test.py"]  # FastMCP 서버 파일
)

tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

MAX_TURNS = 3  # 최근 3턴만 유지
TTL = 3600     # 1시간 후 자동 삭제

def get_history(user_id: str) -> list:
    data = r.get(f"chat:{user_id}")
    return json.loads(data) if data else []

def save_history(user_id: str, history: list):
    # 최근 MAX_TURNS*2개만 유지 (user/assistant 쌍이라 *2)
    trimmed = history[-(MAX_TURNS * 2):]
    r.setex(
        f"chat:{user_id}",
        TTL,
        json.dumps(trimmed, ensure_ascii=False)
    )

def delete_history(user_id: str):
    r.delete(f"chat:{user_id}")

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
async def generate(input_ids, attention_mask, websocket, max_new_tokens, temperature, top_k, top_p):
    """LLM 토큰 생성 + 스트리밍. 생성된 전체 텍스트 반환"""
    generated_ids = []
    for i, _ in enumerate(range(max_new_tokens)):
        position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[None, :]
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }
 
        outputs = session_ort.run(None, ort_inputs)
        logits = outputs[0]
        next_token_id = sample_next_token(logits, temperature, top_k, top_p)
 
        # assistant 턴 종료 토큰 확인
        stop_tokens = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")]

        if next_token_id in stop_tokens:
            break


        #await websocket.send(token_text)
        #await asyncio.sleep(0.01)
 
        input_ids = np.concatenate([input_ids, [[next_token_id]]], axis=1)
        attention_mask = np.concatenate([attention_mask, [[1]]], axis=1)
        generated_ids.append(next_token_id)
        print(i)

    token_text = tokenizer.decode(generated_ids, skip_special_tokens=True)       
    print(token_text)
    return token_text or ""
 
async def execute_tool(tool_name, tool_input, mcp_session, websocket):
    """MCP Tool 실행 + 스트리밍으로 과정 전달"""
    tool_result = await mcp_session.call_tool(tool_name, tool_input)
    result_text = tool_result.content[0].text
    return result_text

def parse_tool_call(text):
    """
    LLM 응답에서 <tool_call>...</tool_call> 형식 파싱
    예: <tool_call>{"tool": "add", "input": {"a": 3, "b": 5}}</tool_call>
    """
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1).strip())
            if "tool" in parsed and "input" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    return None

def rule_based_tool_check(prompt, tools):
    """
    사용자 질문을 보고 Rule-based로 Tool 필요 여부 판단.
    Tool name/description 키워드와 질문을 매칭.
    """
    prompt_lower = prompt.lower()
    for tool in tools:
        keywords = tool["name"].lower().split("_")
        if any(kw in prompt_lower for kw in keywords):
            return tool
    return None
 

async def chat_handler(websocket, mcp_session, tools, tool_list_str, user_id):
    prompt = await websocket.recv()
 
    max_new_tokens = 100
    temperature = 1.0
    top_k = 50
    top_p = 0.9

    history = get_history(user_id)
    history.append({"role": "user", "content": prompt})

    # ① Rule-based로 먼저 Tool 필요 여부 판단
    #rule_tool = rule_based_tool_check(prompt, tools)
 
    chat = [
        {"role": "tool_list", "content": tool_list_str},
        {"role": "system", "content": (
            "- AI 언어모델의 이름은 'CLOVA X'이며 네이버에서 만들었다.\n"
            "- 절대로 user 질문을 스스로 만들지 마라. assistant 답변 하나만 생성하고 멈춰라.\n"
            "- 너는 assistant 역할로만 답변하며, 새로운 user 질문을 생성하지 않는다.\n"
            "- assistant 답변은 반드시 하나만 생성한다."
        )},
        *history,
    ]

    print(chat)
    inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_dict=True, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)
 
    # ② 1차 LLM 생성 (스트리밍)
    llm_response = await generate(input_ids, attention_mask, websocket, max_new_tokens, temperature, top_k, top_p)
    print("llm_response: ")
    print(llm_response)
    # ③ LLM 응답에서 Tool 호출 파싱 시도
    tool_call = parse_tool_call(llm_response)

    result_text = None
 
    if tool_call:
        # LLM이 형식을 잘 따른 경우
        result_text = await execute_tool(tool_call["tool"], tool_call["input"], mcp_session, websocket)
        print("result_text: "+result_text)

    # ④ Tool 결과가 있으면 LLM 재호출해서 최종 답변 생성
    if result_text:
        chat.append({"role": "assistant", "content": llm_response})
        chat.append({"role": "tool_result", "content": result_text})
 
        inputs2 = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_dict=True, return_tensors="np")
        input_ids2 = inputs2["input_ids"].astype(np.int64)
        attention_mask2 = inputs2["attention_mask"].astype(np.int64)
        
        llm_response = await generate(input_ids2, attention_mask2, websocket, max_new_tokens, temperature, top_k, top_p)
        
        print("llm_response2: ")
        print(llm_response)

    history.append({"role": "assistant", "content": llm_response})

    # Redis에 저장
    save_history(user_id, history)
    
    print(user_id)

    print('소켓에서 내용 전송')
    await websocket.send(llm_response)

async def main():
    async with stdio_client(server_params) as (read, write):
        
        async with ClientSession(read, write) as mcp_session:
            await mcp_session.initialize()
 
            # MCP에서 Tool 목록 자동으로 가져오기
            tools_result = await mcp_session.list_tools()
            tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.inputSchema
                }
                for t in tools_result.tools
            ]
            tool_list_str = json.dumps(tools, ensure_ascii=False)
            #print("MCP Tool 목록:", tool_list_str)
 
            async def handler(websocket):
                user_id = str(websocket.id)  # 고유 ID (UUID 형식 - 임시)
                await chat_handler(websocket, mcp_session, tools, tool_list_str,user_id)
 
            async with websockets.serve(handler, "localhost", 8765):
                print("WebSocket server started at ws://localhost:8765")
                await asyncio.Future()

asyncio.run(main())