"""
Pydantic 请求/响应模型
"""
import uuid
from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求"""
    query: str = Field(..., description="用户输入的问题", min_length=1, max_length=5000)
    session_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:12],
        description="会话ID，用于关联多轮对话记忆"
    )


class SSEChunk(BaseModel):
    """SSE 流式输出的单条数据"""
    type: str = Field(..., description="消息类型: content | tool_call | done | error")
    content: str = Field(default="", description="文本内容")
    metadata: dict = Field(default_factory=dict, description="附加元数据")


class KnowledgeStats(BaseModel):
    """知识库统计信息"""
    document_count: int = Field(..., description="已索引的文档chunk数量")
    collection_name: str = Field(..., description="Chroma collection 名称")


class ReindexResponse(BaseModel):
    """重建索引响应"""
    success: bool
    message: str
    new_document_count: Optional[int] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
