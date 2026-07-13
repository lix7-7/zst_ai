"""
对话 SSE 流式接口
POST /api/v1/chat/stream
"""
import json
import traceback
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from api.schemas.chat import ChatRequest
from utils.logger_handler import logger

router = APIRouter(prefix="/chat", tags=["对话"])


def format_sse(data: dict) -> str:
    """将 dict 格式化为 SSE 事件字符串"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/stream", summary="SSE 流式对话")
async def chat_stream(request: ChatRequest, req: Request):
    """
    发送用户问题，返回 SSE 流式响应。

    事件格式:
        data: {"type": "content", "content": "..."}   # 逐 token 内容
        data: {"type": "tool_call", "content": "...", "metadata": {...}}  # 工具调用
        data: {"type": "done"}                          # 对话结束
        data: {"type": "error", "content": "..."}       # 错误

    示例:
        curl -N -X POST http://localhost:8000/api/v1/chat/stream \
          -H "Content-Type: application/json" \
          -d '{"query": "小户型适合哪些扫地机器人？", "session_id": "test123"}'
    """
    agent = req.app.state.agent

    async def event_generator():
        full_response = []
        try:
            async for chunk in agent.execute_stream_async(request.query, request.session_id):
                full_response.append(chunk)
                yield format_sse({"type": "content", "content": chunk.rstrip("\n")})

            logger.info(f"[SSE] 对话完成 session={request.session_id}, 响应长度={len(''.join(full_response))}")
            yield format_sse({"type": "done"})

            # 异步保存对话到记忆系统
            try:
                from memory.manager import MemoryManager
                mgr = MemoryManager.get_instance()
                await mgr.add_interaction(
                    session_id=request.session_id,
                    user_msg=request.query,
                    assistant_msg="".join(full_response),
                )
            except Exception:
                pass  # 记忆保存失败不影响响应

        except Exception as e:
            logger.error(f"[SSE] 对话异常 session={request.session_id}: {str(e)}")
            logger.debug(traceback.format_exc())
            yield format_sse({"type": "error", "content": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        },
    )
