"""
对话 SSE 流式接口
POST /api/v1/chat/stream
"""
import json
import traceback
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from api.schemas.chat import (
    ChatRequest, SessionInfo, SessionListResponse,
    ChatMessage, SessionMessagesResponse, DeleteSessionResponse,
)
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
            async for chunk in agent.execute_stream_async(
                request.query, request.session_id, request.user_id
            ):
                if chunk.startswith('{"type":'):
                    # 工具调用/结果事件（已是 JSON 格式）
                    yield format_sse(json.loads(chunk))
                else:
                    # 普通文本 token（逐字流式输出）
                    full_response.append(chunk)
                    yield format_sse({"type": "content", "content": chunk})

            logger.info(f"[SSE] 对话完成 session={request.session_id}, 响应长度={len(''.join(full_response))}")
            yield format_sse({"type": "done"})

            # 异步保存对话到记忆系统
            try:
                from memory.manager import MemoryManager
                mgr = MemoryManager.get_instance()
                assistant_text = "".join(full_response)
                await mgr.add_interaction(
                    session_id=request.session_id,
                    user_msg=request.query,
                    assistant_msg=assistant_text,
                    user_id=request.user_id,
                )
                # 有 user_id 时注册/刷新会话归属
                if request.user_id:
                    await mgr.register_user_session(
                        request.user_id, request.session_id,
                        preview=request.query[:50],
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


# ---- 会话管理 API ----

@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(req: Request, user_id: str = None):
    """
    列出会话。

    - 无 user_id: 扫描所有 Redis 会话（旧行为，向后兼容）
    - 有 user_id: 从用户索引快速查询（新行为）
    """
    try:
        mgr = req.app.state.memory_manager
        if mgr is None:
            return SessionListResponse(user_id=user_id or "", sessions=[])

        # ---- 新路径：用户过滤 ----
        if user_id:
            sessions_meta = await mgr.get_user_sessions(user_id)
            from datetime import datetime, timezone
            sessions = [
                SessionInfo(
                    session_id=m["session_id"],
                    user_id=m.get("user_id", user_id),
                    message_count=m.get("message_count", 0),
                    last_active=datetime.fromtimestamp(
                        m["last_active"], tz=timezone.utc
                    ).isoformat() if m.get("last_active") else None,
                    preview=m.get("preview", ""),
                )
                for m in sessions_meta
            ]
            return SessionListResponse(user_id=user_id, sessions=sessions)

        # ---- 旧路径：全量扫描（向后兼容） ----
        if mgr.short_term is None or mgr.short_term.redis is None:
            return SessionListResponse(sessions=[])

        redis_client = mgr.short_term.redis
        sessions: list[SessionInfo] = []
        from datetime import datetime, timezone

        # SCAN 遍历 session:*:messages 键（避免 KEYS 阻塞）
        cursor = 0
        while True:
            cursor, keys = await redis_client.scan(
                cursor, match="session:*:messages", count=50
            )
            for key in keys:
                parts = key.split(":")
                if len(parts) >= 3:
                    session_id = parts[1]
                    msg_count = await redis_client.llen(key)
                    last_msgs = await redis_client.lrange(key, 0, 0)
                    last_active = None
                    preview = ""
                    if last_msgs:
                        import json
                        try:
                            msg = json.loads(last_msgs[0])
                            ts = msg.get("timestamp", 0)
                            last_active = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                        except Exception:
                            pass
                    all_msgs = await redis_client.lrange(key, 0, -1)
                    for m_raw in reversed(all_msgs):
                        try:
                            import json as _json
                            m = _json.loads(m_raw)
                            if m.get("role") == "user":
                                preview = m.get("content", "")[:50]
                                break
                        except Exception:
                            pass
                    sessions.append(SessionInfo(
                        session_id=session_id,
                        message_count=msg_count,
                        last_active=last_active,
                        preview=preview,
                    ))
            if cursor == 0:
                break

        sessions.sort(key=lambda s: s.last_active or "", reverse=True)
        return SessionListResponse(sessions=sessions)

    except Exception as e:
        logger.warning(f"[Session] 列会话失败: {e}")
        return SessionListResponse(user_id=user_id or "", sessions=[])


@router.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
async def get_session_messages(session_id: str, req: Request):
    """获取指定会话的历史消息"""
    try:
        mgr = req.app.state.memory_manager
        if mgr is None or mgr.short_term is None:
            return SessionMessagesResponse(session_id=session_id, messages=[])

        history = await mgr.short_term.get_history(session_id)
        messages = [
            ChatMessage(
                role=m.get("role", "unknown"),
                content=m.get("content", ""),
                timestamp=m.get("timestamp"),
            )
            for m in history
        ]
        return SessionMessagesResponse(session_id=session_id, messages=messages)

    except Exception as e:
        logger.warning(f"[Session] 获取消息失败 session={session_id}: {e}")
        return SessionMessagesResponse(session_id=session_id, messages=[])


@router.delete("/sessions/{session_id}", response_model=DeleteSessionResponse)
async def delete_session(session_id: str, req: Request, user_id: str = None):
    """删除指定会话（Redis + 用户索引）"""
    try:
        mgr = req.app.state.memory_manager
        if mgr is not None:
            await mgr.clear(session_id, user_id=user_id)
        return DeleteSessionResponse(success=True, session_id=session_id)
    except Exception as e:
        logger.warning(f"[Session] 删除会话失败 session={session_id}: {e}")
        return DeleteSessionResponse(success=False, session_id=session_id)
