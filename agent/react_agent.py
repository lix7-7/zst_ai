from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report, memory_recall)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch, memory_inject, token_guard


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
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"

    async def execute_stream_async(self, query: str, session_id: str = None):
        """
        异步流式执行（FastAPI SSE 兼容）

        Args:
            query: 用户输入问题
            session_id: 会话ID，用于记忆关联
        """
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        context = {
            "report": False,
            "session_id": session_id,
        }

        async for chunk in self.agent.astream(input_dict, stream_mode="values", context=context):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
