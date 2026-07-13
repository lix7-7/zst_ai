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

    # 防止 ReAct 循环中重复注入（每次 model call 都触发 @before_model）
    first_msg = state["messages"][0]
    if hasattr(first_msg, "content") and "<!--memory_injected-->" in first_msg.content:
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
                # 将记忆注入到第一条 system 消息（带防重复标记）
                if hasattr(first_msg, "content") and first_msg.type == "system":
                    first_msg.content = context + "\n\n" + first_msg.content + "\n<!--memory_injected-->"
    except Exception:
        pass  # 记忆注入失败不影响核心对话

    return None


# ============================================================
# Token 窗口守卫
# ============================================================

_SUMMARY_PROMPT = """请将以下对话历史压缩为一段简短摘要（不超过 150 字），保留关键信息：
- 用户的核心需求和偏好
- Agent 给出的重要结论或建议
- 未解决的问题

对话：
{conversation}

摘要："""


@before_model
async def token_guard(
    state: AgentState,
    runtime: Runtime,
):
    """
    Token 窗口守卫：在每次模型调用前检查上下文 token 数，
    超出阈值时自动裁剪最早的历史消息，必要时生成摘要。

    必须在 memory_inject 之后执行，此时 system prompt 已包含注入的记忆。
    """
    session_id = runtime.context.get("session_id")
    if not session_id:
        return None

    # 检查是否已经处理过（单次 ReAct 循环只裁一次）
    first_msg = state["messages"][0]
    if not (hasattr(first_msg, "content") and first_msg.type == "system"):
        return None
    if "<!--token_guarded-->" in first_msg.content:
        return None

    try:
        from utils.token_counter import get_token_counter
        counter = get_token_counter()

        system_text = first_msg.content
        other_messages = state["messages"][1:]  # system 之后的消息

        # 1. 检查是否超标
        is_over, budget = counter.check(system_text, other_messages)
        if not is_over:
            logger.debug(
                f"[token_guard] OK total={budget.total}/{budget.limit} "
                f"session={session_id}"
            )
            return None

        logger.warning(
            f"[token_guard] OVER total={budget.total}/{budget.limit} "
            f"session={session_id}, 开始截断..."
        )

        # 2. 解析 system prompt：找出历史记录段
        new_system = _trim_history(counter, system_text, budget.limit)
        first_msg.content = new_system + "\n<!--token_guarded-->"

    except Exception:
        pass  # 守卫失败不影响核心对话

    return None


def _parse_system_parts(system_text: str) -> tuple[str, str, str]:
    """
    将 system prompt 拆分为三部分：
        prefix:  历史记录之前的内容（通常是空或摘要）
        history: 历史对话记录文本
        suffix:  历史记录之后的内容（系统提示词正文 + 标记）
    """
    history_marker = "## 历史对话记录"
    long_term_marker = "## 相关历史记忆"
    injected_marker = "<!--memory_injected-->"

    if history_marker not in system_text:
        return ("", "", system_text)

    # 找到历史记录段起始
    hist_start = system_text.find(history_marker)
    prefix = system_text[:hist_start]

    # 找到历史记录段结束（下一个 ## 标题 或 <!--memory_injected-->）
    after_hist = system_text[hist_start:]
    hist_end = len(after_hist)

    # 找下一个段落标记
    for end_marker in [long_term_marker, injected_marker]:
        pos = after_hist.find(end_marker)
        if pos > 0 and pos < hist_end:
            hist_end = pos

    history_block = after_hist[:hist_end].rstrip()
    suffix = after_hist[hist_end:]

    # 如果 suffix 中有 long_term_memory + injected_marker，把它们从 history 段中分离
    return (prefix, history_block, suffix)


def _parse_history_lines(history_block: str) -> list[str]:
    """将历史记录块解析为独立行列表，每个条目是 '用户: ...' 或 '客服: ...'"""
    lines = []
    for line in history_block.split("\n"):
        stripped = line.strip()
        if stripped.startswith("用户:") or stripped.startswith("客服:"):
            lines.append(stripped)
    return lines


def _trim_history(counter, system_text: str, limit: int) -> str:
    """
    逐步裁剪历史消息，直到 system prompt 总 token 数回到 limit 以下。
    如果裁掉的消息 ≥ 6 条，则生成 LLM 摘要替代它们。
    """
    prefix, history_block, suffix = _parse_system_parts(system_text)
    history_lines = _parse_history_lines(history_block)

    if not history_lines:
        # 没有历史可裁
        return system_text

    trimmed = []
    remaining = list(history_lines)

    # 逐条裁掉最早的消息
    while remaining and counter.count(
        prefix + "\n".join(["## 历史对话记录"] + remaining) + suffix
    ) > limit:
        trimmed.append(remaining.pop(0))  # 移除最早的

    if not trimmed:
        return system_text

    logger.warning(
        f"[token_guard] 裁掉 {len(trimmed)} 条旧消息 "
        f"({counter.count(system_text)} → {counter.count(prefix + '\n'.join(['## 历史对话记录'] + remaining) + suffix)})"
    )

    # 裁得多了 → 生成摘要
    token_config = _load_token_config_safe()
    trigger = token_config.get("summarize_trigger_messages", 6)

    if len(trimmed) >= trigger:
        summary = _generate_summary_sync(trimmed)
        if summary:
            parts = []
            if prefix.strip():
                parts.append(prefix.rstrip())
            parts.append("## 历史对话摘要")
            parts.append(summary)
            parts.append("")
            if remaining:
                parts.append("## 历史对话记录")
                parts.extend(remaining)
            parts.append(suffix.lstrip())
            return "\n".join(parts)

    # 不需要摘要，直接拼接
    parts = []
    if prefix.strip():
        parts.append(prefix.rstrip())
    if remaining:
        parts.append("## 历史对话记录")
        parts.extend(remaining)
    parts.append(suffix.lstrip())
    return "\n".join(parts)


def _generate_summary_sync(messages: list[str]) -> str:
    """同步调用 LLM 生成摘要（在异步上下文中用 run_in_executor 避免阻塞）"""
    try:
        from model.factory import chat_model
        conversation = "\n".join(messages[-12:])  # 最多取最近 12 条做摘要
        prompt = _SUMMARY_PROMPT.format(conversation=conversation)
        response = chat_model.invoke(prompt)
        summary = response.content.strip()
        logger.info(f"[token_guard] 生成摘要 ({len(messages)} 条 → {len(summary)} 字): {summary[:80]}...")
        return summary
    except Exception as e:
        logger.warning(f"[token_guard] 摘要生成失败: {e}")
        return ""


def _load_token_config_safe() -> dict:
    """安全加载 token 配置"""
    try:
        from utils.config_handler import load_yaml_config
        conf = load_yaml_config("config/memory.yml")
        return conf.get("token_window", {})
    except Exception:
        return {}
