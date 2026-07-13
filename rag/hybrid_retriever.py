"""
混合检索器：BM25 (关键词精准匹配) + Dense Vector (语义匹配) → RRF 融合 → Reranker 精排

检索流程:
    1. Dense Vector 语义检索 top dense_k
    2. BM25 关键词检索 top sparse_k
    3. RRF 公式融合，取 top rerank_top_n
    4. gte-rerank 交叉编码器精排，返回 top final_k

使用方式:
    vs = VectorStoreService()
    hybrid_retriever = HybridRetriever(vs)
    docs = hybrid_retriever.retrieve("小户型适合什么扫地机器人", top_k=5)
"""
from typing import Optional
import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_chroma import Chroma
from dashscope import TextReRank
from utils.config_handler import chroma_conf
from utils.logger_handler import logger


class HybridRetriever:
    """
    BM25 + Dense Vector → RRF 融合 → Reranker 精排

    检索流程:
        1. Dense Vector 语义检索 top dense_k
        2. BM25 关键词检索 top sparse_k
        3. RRF 公式融合，取 top rerank_top_n 作为候选
        4. gte-rerank 交叉编码器精排，返回 top final_k
    """

    def __init__(
        self,
        vector_store: Chroma,
        dense_k: Optional[int] = None,
        sparse_k: Optional[int] = None,
        final_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        use_rerank: bool = True,
        rerank_top_n: Optional[int] = None,
    ):
        hybrid_conf = chroma_conf.get("hybrid_search", {})

        self.vector_store = vector_store
        self.dense_k = dense_k or hybrid_conf.get("dense_k", 20)
        self.sparse_k = sparse_k or hybrid_conf.get("sparse_k", 20)
        self.final_k = final_k or hybrid_conf.get("final_k", 5)
        self.rrf_k = rrf_k or hybrid_conf.get("rrf_k", 60)
        self.use_rerank = use_rerank
        self.rerank_top_n = rerank_top_n or hybrid_conf.get("rerank_top_n", 20)
        self.rerank_model = hybrid_conf.get("rerank_model", "gte-rerank")

        self.dense_retriever = vector_store.as_retriever(
            search_kwargs={"k": self.dense_k}
        )

        # BM25 索引（懒加载）
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: list[Document] = []
        self._bm25_corpus: list[list[str]] = []

    def _tokenize(self, text: str) -> list[str]:
        """中文分词"""
        return list(jieba.cut(text))

    def _build_bm25(self, documents: list[Document]):
        """构建 BM25 索引"""
        if not documents:
            logger.warning("[HybridRetriever] 无文档可用于构建 BM25")
            return

        self._bm25_docs = documents
        self._bm25_corpus = [self._tokenize(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(self._bm25_corpus)
        logger.info(f"[HybridRetriever] BM25 索引构建完成，文档数={len(documents)}")

    def _ensure_bm25(self):
        """确保 BM25 已初始化"""
        if self._bm25 is not None:
            return

        result = self.vector_store.get(include=["documents", "metadatas"])
        if not result["documents"]:
            logger.warning("[HybridRetriever] Chroma 向量库为空，BM25 无法构建")
            self._bm25 = None
            self._bm25_docs = []
            self._bm25_corpus = []
            return

        documents = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(result["documents"], result["metadatas"])
        ]
        self._build_bm25(documents)

    def _dense_search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Dense vector 语义检索，返回 (文档, 相似度分数)"""
        docs = self.dense_retriever.invoke(query)
        results = []
        for doc in docs:
            # Chroma 返回的文档可能不带分数，给默认值
            score = doc.metadata.get("score", 1.0) if doc.metadata else 1.0
            results.append((doc, score))
        return results[:k]

    def _sparse_search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """BM25 关键词检索，返回 (文档, BM25分数)"""
        self._ensure_bm25()

        if self._bm25 is None or not self._bm25_corpus:
            return []

        tokenized_query = self._tokenize(query)
        bm25_scores = self._bm25.get_scores(tokenized_query)

        # 取 top k
        indexed_scores = list(enumerate(bm25_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = indexed_scores[:k]

        return [
            (self._bm25_docs[idx], score)
            for idx, score in top_indices
            if score > 0
        ]

    def _rrf_fusion(
        self,
        dense_results: list[tuple[Document, float]],
        sparse_results: list[tuple[Document, float]],
        top_k: int,
    ) -> list[Document]:
        """
        RRF (Reciprocal Rank Fusion) 融合

        公式: score(doc) = Σ 1 / (k + rank_i)
        - k = 60 (RRF 平滑参数)
        - rank 从 1 开始
        - 同一文档在两路结果中都出现，分数累加
        """
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}  # content → Document 映射

        def doc_key(doc: Document) -> str:
            return doc.page_content

        # Dense 路 RRF 分数
        for rank, (doc, _) in enumerate(dense_results, start=1):
            key = doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (self.rrf_k + rank)
            if key not in doc_map:
                doc_map[key] = doc

        # Sparse 路 RRF 分数
        for rank, (doc, _) in enumerate(sparse_results, start=1):
            key = doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (self.rrf_k + rank)
            if key not in doc_map:
                doc_map[key] = doc

        # 按 RRF score 降序排列
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        logger.info(
            f"[HybridRetriever] RRF 融合: dense={len(dense_results)}, "
            f"sparse={len(sparse_results)}, merged={len(sorted_keys)}, final={min(top_k, len(sorted_keys))}"
        )

        return [doc_map[k] for k in sorted_keys[:top_k]]

    def _rerank(self, query: str, candidates: list[Document], top_k: int) -> list[Document]:
        """
        Cross-Encoder 重排序：用 gte-rerank 对候选文档精确打分

        Cross-Encoder 直接将 (query, document) 配对输入模型，能捕捉细粒度语义交互，
        比 embedding 向量距离更精准。

        Args:
            query: 用户查询
            candidates: RRF 融合后的候选文档列表
            top_k: 返回文档数

        Returns:
            精排后的文档列表
        """
        if len(candidates) <= top_k:
            return candidates

        try:
            documents = [doc.page_content for doc in candidates]
            result = TextReRank.call(
                model=self.rerank_model,
                query=query,
                documents=documents,
                top_n=top_k,
            )

            if result.status_code != 200:
                logger.warning(
                    f"[HybridRetriever] Reranker API 异常 "
                    f"status={result.status_code} message={result.message}，回退到 RRF 排序"
                )
                return candidates[:top_k]

            # 按 rerank 分数重新排序
            reranked_indices = [item["index"] for item in result.output["results"]]
            reranked = [candidates[idx] for idx in reranked_indices]

            logger.info(
                f"[HybridRetriever] Reranker 精排完成: "
                f"candidates={len(candidates)} → top_{top_k}"
            )
            return reranked

        except Exception as e:
            logger.warning(
                f"[HybridRetriever] Reranker 调用失败 {str(e)}，回退到 RRF 排序"
            )
            return candidates[:top_k]

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[Document]:
        """
        混合检索主入口：Dense + BM25 → RRF 融合 → Reranker 精排

        Args:
            query: 用户查询
            top_k: 返回文档数，默认使用配置值

        Returns:
            排序后的文档列表
        """
        k = top_k or self.final_k

        dense_results = self._dense_search(query, self.dense_k)
        sparse_results = self._sparse_search(query, self.sparse_k)

        # 如果 BM25 无结果，退化到纯 Dense
        if not sparse_results:
            logger.info("[HybridRetriever] BM25 无结果，使用纯 Dense 检索")
            return [doc for doc, _ in dense_results[:k]]

        # 如果 Dense 无结果，退化到纯 BM25
        if not dense_results:
            logger.info("[HybridRetriever] Dense 无结果，使用纯 BM25 检索")
            return [doc for doc, _ in sparse_results[:k]]

        # RRF 融合：取 rerank_top_n 个候选
        candidates = self._rrf_fusion(dense_results, sparse_results, self.rerank_top_n)

        # Reranker 精排
        if self.use_rerank and len(candidates) > k:
            return self._rerank(query, candidates, k)

        return candidates[:k]

    def invalidate_cache(self):
        """清除 BM25 缓存，文档更新后调用"""
        self._bm25 = None
        self._bm25_docs = []
        self._bm25_corpus = []
        logger.info("[HybridRetriever] BM25 缓存已清除")


if __name__ == '__main__':
    from rag.vector_store import VectorStoreService

    vs = VectorStoreService()
    retriever = HybridRetriever(vs.vector_store)

    test_queries = [
        "小户型适合哪些扫地机器人？",
        "扫地机器人迷路了怎么办",
        "怎么更换滤网",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"查询: {q}")
        print(f"{'='*60}")
        docs = retriever.retrieve(q, top_k=3)
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知")
            print(f"\n[{i}] 来源: {source}")
            print(f"    内容: {doc.page_content[:120]}...")
