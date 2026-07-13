from typing import Callable, Awaitable
from utils.prompt_loader import load_system_prompts, load_report_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger
import inspect


@wrap_tool_call
async def monitor_tool(
        # 请求的数据封装
        request: ToolCallRequest,
        # 执行的函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command | Awaitable[ToolMessage | Command]],
) -> ToolMessage | Command:             # 工具执行的监控
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    try:
        # 兼容同步和异步 handler
        if inspect.iscoroutinefunction(handler) or inspect.isawaitable(handler):
            result = await handler(request)
        else:
            result = handler(request)

        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e


@before_model
async def log_before_model(
        state: AgentState,          # 整个Agent智能体中的状态记录
        runtime: Runtime,           # 记录了整个执行过程中的上下文信息
):         # 在模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")

    last_msg = state['messages'][-1] if state.get('messages') else None
    if last_msg and hasattr(last_msg, 'content') and last_msg.content:
        logger.debug(f"[log_before_model]{type(last_msg).__name__} | {str(last_msg.content)[:200]}")

    return None


@dynamic_prompt                 # 每一次在生成提示词之前，调用此函数
def report_prompt_switch(request: ModelRequest):     # 动态切换提示词
    is_report = request.runtime.context.get("report", False)
    if is_report:               # 是报告生成场景，返回报告生成提示词内容
        return load_report_prompts()

    return load_system_prompts()


@before_model
async def memory_inject(
        state: AgentState,
        runtime: Runtime,
):
    """
    记忆注入中间件：在每次模型调用前，将多轮对话记忆注入到消息列表

    从 runtime.context 中获取 session_id，拉取短期记忆（最近对话）和
    长期记忆（语义检索相关历史摘要），注入到 system 消息中。
    """
    session_id = runtime.context.get("session_id")
    if not session_id:
        return None

    try:
        from memory.manager import MemoryManager
        mgr = MemoryManager.get_instance()

        # 确保 MemoryManager 已初始化
        if not mgr._initialized:
            try:
                await mgr.initialize()
            except Exception:
                pass

        # 获取当前用户输入作为检索 query
        current_query = ""
        for msg in reversed(state["messages"]):
            if hasattr(msg, "type") and msg.type == "human":
                current_query = msg.content
                break

        if current_query:
            # 异步获取记忆上下文
            context = await mgr.get_context(session_id, current_query)

            if context and state["messages"]:
                # 将记忆注入到第一条 system 消息
                first_msg = state["messages"][0]
                if hasattr(first_msg, "content") and first_msg.type == "system":
                    first_msg.content = context + "\n\n" + first_msg.content
    except Exception:
        pass  # 记忆注入失败不影响核心对话

    return None
