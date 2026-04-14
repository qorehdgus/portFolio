import os
import traceback
from xml.parsers.expat import model

import redis, json, threading
import uvicorn
import re
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from pydantic import BaseModel

from LLM.version2.test import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
MODEL_NAME = os.getenv("MODEL_NAME")

def create_app():
    app = FastAPI()

    # Redis 연결
    pool = redis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, db=0, max_connections=20)
    r = redis.Redis(connection_pool=pool)

    # 허용할 origin 목록
    origins = ["http://localhost:9090", "http://127.0.0.1:9090", "*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 모델 로딩 (여기서만 실행)
    model_name = MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
        quantization_config=bnb_config,
        dtype="auto"
    )

    # Redis 저장/불러오기 함수
    def save_message(session_id, role, content):
        message = {"role": role, "content": content}
        key = f"chat:{session_id}"
        r.rpush(key, json.dumps(message))
        r.expire(key, 360)

    def load_chat(session_id):
        messages = r.lrange(f"chat:{session_id}", 0, -1)
        return [json.loads(m) for m in messages]

    class ChatRequest(BaseModel):
        session_id: str
        prompt: str

    async def run_mcp_tool(tool_name: str, tool_input: dict):
        """MCP 서버에서 특정 툴 실행"""
        server_params = StdioServerParameters(
            command="python",
            args=["/app/test.py"]
        ) 
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, tool_input)
                return result.content[0].text if result.content else "MCP 결과 없음"

    async def get_tools_from_mcp():
        """MCP 서버에서 툴 목록 가져오기"""

        server_params = StdioServerParameters(
            command="python",
            args=["/app/test.py"]
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                tools = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    }
                    for tool in tools_result.tools
                ]
                return tools

    def safe_json_parse(model_output: str):
        """
        모델 출력이 파이썬 dict 스타일일 경우 자동으로 JSON으로 변환.
        """
        try:
            # 먼저 그대로 JSON 파싱 시도
            return json.loads(model_output)
        except json.JSONDecodeError:
            # 작은따옴표를 큰따옴표로 바꿔서 재시도
            fixed_output = re.sub(r"'", '"', model_output)
            try:
                return json.loads(fixed_output)
            except json.JSONDecodeError:
                # 그래도 실패하면 None 반환
                return None


    # MCP 툴 목록 가져오기 엔드포인트
    @app.get("/tools")
    async def list_tools():
        tools = await get_tools_from_mcp()
        return {"tools": tools}

    @app.post("/answer")
    async def chat_request(chat_request: ChatRequest):
        print('수신')
        session_id = chat_request.session_id
        prompt = chat_request.prompt
        chat_history = load_chat(session_id)
        print(not chat_history)
        if not chat_history:
            tools = await get_tools_from_mcp()
            chat_history.append({"role": "system", "content": f"tool_List: {json.dumps(tools, ensure_ascii=False)}"})
            save_message(session_id, "system", f"tool_List: {json.dumps(tools, ensure_ascii=False)}")

            chat_history.append({"role": "system", "content": "Always answer ONLY by calling one tool from tool_List."})
            save_message(session_id, "system", "Always answer ONLY by calling one tool from tool_List.")

            chat_history.append({"role": "system", "content": "Output MUST be valid JSON: {\"tool\":\"tool_name\",\"input\":{...}}."})
            save_message(session_id, "system", "Output MUST be valid JSON: {\"tool\":\"tool_name\",\"input\":{...}}.")

            chat_history.append({"role": "system", "content": "If the user's request cannot be fulfilled with any tool in tool_List, reply only with 'Tools cannot be used.'"})
            save_message(session_id, "system", "If the user's request cannot be fulfilled with any tool in tool_List, reply only with 'Tools cannot be used.'")

            chat_history.append({"role": "system", "content": "If the last message is an MCP result, do not output JSON. Instead, only use the MCP result to create a clear and natural conversational answer in Korean for the user."})
            save_message(session_id, "system", "If the last message is an MCP result, do not output JSON. Instead, only use the MCP result to create a clear and natural conversational answer in Korean for the user.")


        chat_history.append({"role": "user", "content": prompt})
        save_message(session_id, "user", prompt)

        inputs = tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        model_inputs = tokenizer([inputs], return_tensors="pt").to(model.device)

        terminators = [
            tokenizer.eos_token_id
        ]

        # 1차 출력 (툴 호출 여부 확인)
        output_ids = model.generate(
            **model_inputs, 
            max_new_tokens=512,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            eos_token_id=terminators,
            repetition_penalty=1.2
        
        )
        # 입력 토큰 길이 확인
        input_length = model_inputs["input_ids"].shape[-1]

        # 모델 출력에서 입력 부분을 제외하고 답변만 추출
        response_ids = output_ids[0][input_length:]
        first_output = tokenizer.decode(response_ids, skip_special_tokens=True)
        #first_output = tokenizer.decode(output_ids[0][model_inputs.shape[-1]:], skip_special_tokens=True)
        print(f'1차 출력: {first_output}')
        mcp_result = None
        if first_output.strip().startswith("{") and "tool" in first_output:
            print('MCP 활용 답변 수행')

            try:
                tool_call = safe_json_parse(first_output)
                tool_name = tool_call["tool"]
                tool_input = tool_call.get("input", {})
                print(f'호출할 툴: {tool_name}, 입력: {tool_input}')
                mcp_result = await run_mcp_tool(tool_name, tool_input)
                chat_history.append({"role": "system", "content": f"MCP result: {mcp_result}"})

                # MCP 결과 반영 후 최종 답변 생성
                inputs = tokenizer.apply_chat_template(
                    chat_history,
                    tokenize=False,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                model_inputs = tokenizer([inputs], return_tensors="pt").to(model.device)
                print(f'업데이트된 입력: {inputs}')
            except Exception as e:
                mcp_result = f"MCP 실행 오류: {e}"
                traceback.print_exc()
        print(f'MCP 결과: {mcp_result}')
        # 최종 답변 생성 (스트리밍)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

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

        response_text = ""
        print('최종 답변 송부')
        for token in streamer:
            response_text += token
            r.publish(f"stream:{session_id}", token)
        save_message(session_id, "assistant", response_text)
        r.publish(f"stream:{session_id}", "[DONE]")

        return {"response": response_text}

    return app


# 독립 실행 시
if __name__ == "__main__":
    uvicorn.run(create_app(), host="0.0.0.0", port=8001)
