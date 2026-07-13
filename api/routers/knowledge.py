"""
知识库管理接口
POST /api/v1/knowledge/reindex
GET  /api/v1/knowledge/stats
"""
from fastapi import APIRouter, Request
from api.schemas.chat import KnowledgeStats, ReindexResponse
from utils.logger_handler import logger

router = APIRouter(prefix="/knowledge", tags=["知识库"])


@router.post("/reindex", response_model=ReindexResponse, summary="重建知识库索引")
async def reindex(request: Request):
    """
    重新扫描 data/ 目录，将新增/变更的文档索引到向量库。
    """
    try:
        from rag.vector_store import VectorStoreService
        vs = VectorStoreService()
        old_count = vs.get_document_count()
        vs.load_document()
        new_count = vs.get_document_count()

        logger.info(f"[知识库] 重建索引完成: {old_count} → {new_count} chunks")
        return ReindexResponse(
            success=True,
            message=f"索引重建完成",
            new_document_count=new_count,
        )
    except Exception as e:
        logger.error(f"[知识库] 重建索引失败: {str(e)}")
        return ReindexResponse(success=False, message=str(e))


@router.get("/stats", response_model=KnowledgeStats, summary="知识库统计")
async def stats(request: Request):
    """
    获取知识库统计信息：已索引文档数、collection 名称等。
    """
    try:
        from rag.vector_store import VectorStoreService
        vs = VectorStoreService()
        return KnowledgeStats(
            document_count=vs.get_document_count(),
            collection_name=vs.vector_store._collection.name,
        )
    except Exception as e:
        from utils.config_handler import chroma_conf
        return KnowledgeStats(
            document_count=0,
            collection_name=chroma_conf["collection_name"],
        )
