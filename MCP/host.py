import asyncio
import json
import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ① MCP 서버 연결 설정
server_params = StdioServerParameters(
    command="C:/Users/dhbaek/Desktop/project/venv/MCP/Scripts/python.exe",
    args=["C:/Users/dhbaek/Desktop/project/MCP/test.py"]  # FastMCP 서버 파일
)

async def run_host():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
 
            # ② MCP 서버에서 Tool 목록 가져오기
            tools_result = await session.list_tools()
            print(tools_result)
            # ③ Claude API가 이해하는 Tool 형식으로 변환
            tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
                for tool in tools_result.tools
            ]
            print(tools)

            # ④ Resource 미리 읽어두기 (대화 시작 시 한 번만)
            greeting_resource = await session.read_resource("greeting://donghyun")
            system_prompt = f"사용자 정보: {greeting_resource.contents[0].text}"
 
            print("Host 시작! 질문을 입력하세요. (종료: 'quit')")
 
            client = anthropic.Anthropic()
            messages = []
 
            while True:
                user_input = input("\n사용자: ")
                if user_input == "quit":
                    break
 
                messages.append({"role": "user", "content": user_input})
 
                # ⑤ Claude 호출
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    system=system_prompt,
                    tools=tools,
                    messages=messages
                )
 
                # ⑥ Claude 응답 무조건 체크
                assistant_message = {"role": "assistant", "content": response.content}
                messages.append(assistant_message)
 
                for block in response.content:
                    if block.type == "tool_use":
                        # ⑦ Tool 호출 감지 → MCP 서버에 전달
                        print(f"\n[Host] Tool 호출 감지: {block.name}({block.input})")
                        tool_result = await session.call_tool(block.name, block.input)
                        result_text = tool_result.content[0].text
                        print(f"[Host] Tool 결과: {result_text}")
 
                        # ⑧ Tool 결과를 다시 Claude에게 전달
                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": result_text
                                }
                            ]
                        })
 
                        # ⑨ Tool 결과 받은 후 Claude 재호출
                        final_response = client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=1000,
                            system=system_prompt,
                            tools=tools,
                            messages=messages
                        )
 
                        for final_block in final_response.content:
                            if final_block.type == "text":
                                print(f"\nClaude: {final_block.text}")
 
                    elif block.type == "text":
                        # 일반 텍스트 응답
                        print(f"\nClaude: {block.text}")
 
 
if __name__ == "__main__":
    asyncio.run(run_host())
 