from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk, ToolMessage
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report, memory_recall)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch, memory_inject, token_guard
import json


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report, memory_recall],
            middleware=[monitor_tool, log_before_model, report_prompt_switch, memory_inject, token_guard],
        )

    def execute_stream(self, query: str):
        """同步流式执行（Streamlit 兼容）"""
        input_dict = {"messages": [{"role": "user", "content": query}]}

        for chunk in self.agent.stream(input_dict, stream_mode="messages", context={"report": False}):
            msg, _ = chunk
            if isinstance(msg, AIMessageChunk):
                if msg.content:
                    yield msg.content
                elif msg.tool_calls:
                    for tc in msg.tool_calls:
                        yield json.dumps({
                            "type": "tool_call",
                            "tool": tc.get("name", ""),
                            "args": tc.get("args", {})
                        }, ensure_ascii=False)
            elif isinstance(msg, ToolMessage):
                yield json.dumps({
                    "type": "tool_result",
                    "tool": getattr(msg, "name", ""),
                    "content": str(msg.content)[:200]
                }, ensure_ascii=False)

    async def execute_stream_async(self, query: str, session_id: str = None, user_id: str = None):
        """
        异步流式执行（FastAPI SSE 兼容）

        使用 stream_mode="messages" 实现真正的逐 token 流式输出。
        每个 chunk 是 (message_chunk, metadata) 元组：
          - AIMessageChunk(content="...") → 直接产出 token 文本
          - AIMessageChunk(tool_calls=[...]) → 产出 JSON 工具调用事件
          - ToolMessage → 产出 JSON 工具结果事件

        user_id 通过 runtime.context 传入 middleware，供 memory_inject 使用。
        """
        input_dict = {"messages": [{"role": "user", "content": query}]}
        context = {"report": False, "session_id": session_id, "user_id": user_id}

        async for msg, metadata in self.agent.astream(
            input_dict, stream_mode="messages", context=context
        ):
            if isinstance(msg, AIMessageChunk):
                if msg.content:
                    # 逐 token 产出（可能是一个字、几个字或一小段）
                    yield msg.content
                elif msg.tool_calls:
                    # 工具调用事件
                    for tc in msg.tool_calls:
                        yield json.dumps({
                            "type": "tool_call",
                            "tool": tc.get("name", ""),
                            "args": tc.get("args", {})
                        }, ensure_ascii=False)
            elif isinstance(msg, ToolMessage):
                # 工具返回结果
                yield json.dumps({
                    "type": "tool_result",
                    "tool": getattr(msg, "name", ""),
                    "content": str(msg.content)[:200]
                }, ensure_ascii=False)


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
