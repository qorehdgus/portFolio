import redis, json, threading
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from pydantic import BaseModel

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def create_app():
    app = FastAPI()

    # Redis 연결
    pool = redis.ConnectionPool(host="localhost", port=6379, db=0, max_connections=20)
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
    model_name = "C:/Users/user/Desktop/github/portFolio/LLM/model/llama-3-Korean-Bllossom-8B"
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

    async def get_tools_from_mcp():
        """MCP 서버에서 툴 목록 가져오기"""
        server_params = StdioServerParameters(
            command="C:/Users/dhbaek/Desktop/project/venv/MCP/Scripts/python.exe",
            args=["C:/Users/dhbaek/Desktop/project/MCP/test.py"]  # FastMCP 서버 파일
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

    @app.post("/answer")
    def chat_request(chat_request: ChatRequest):
        session_id = chat_request.session_id
        prompt = chat_request.prompt
        chat_history = load_chat(session_id)
        if not chat_history:
            chat_history = [
                {"role": "system", "content": "너는 한글로 답변하고 짧고 간결하게 답변하는 챗봇이야."},
                {"role": "system", "content": "너는 👉이런 이모지를 적극 활용하여 답변하는 챗봇이야."}
            ]

        chat_history.append({"role": "user", "content": prompt})
        save_message(session_id, "user", prompt)

        inputs = tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        model_inputs = tokenizer([inputs], return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

        terminators = [
            tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

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
