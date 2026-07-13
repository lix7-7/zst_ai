"""
RAG 评测脚本
三路检索对比（Dense / BM25 / Hybrid）+ RAGAS 生成质量评测

用法: python -m eval.evaluate
"""
import json
import time
import os
from pathlib import Path
from collections import defaultdict

from rag.vector_store import VectorStoreService
from rag.hybrid_retriever import HybridRetriever
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import jieba

from utils.logger_handler import logger


# ============================================================
# 1. 加载测试数据
# ============================================================

def load_test_queries() -> list[dict]:
    path = Path(__file__).parent / "test_queries.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["queries"]


# ============================================================
# 2. 三类检索器
# ============================================================

class DenseRetriever:
    """纯 Dense Vector 检索（Chroma 语义搜索）"""

    def __init__(self, vs: VectorStoreService):
        self.retriever = vs.get_retriever()

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return self.retriever.invoke(query)[:top_k]


class BM25Retriever:
    """纯 BM25 关键词检索"""

    def __init__(self, vs: VectorStoreService):
        result = vs.vector_store.get(include=["documents", "metadatas"])
        self._docs = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(result["documents"], result["metadatas"])
        ] if result["documents"] else []
        corpus = [list(jieba.cut(doc.page_content)) for doc in self._docs]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        if self._bm25 is None:
            return []
        tokens = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [self._docs[i] for i, s in ranked if s > 0]


# ============================================================
# 3. 检索指标计算
# ============================================================

def evaluate_retrieval(
    retriever,
    queries: list[dict],
    top_k: int = 5,
) -> dict:
    """
    多维度检索评测

    三个指标维度：
    1. Hit Rate (宽松): 前 k 个结果中至少命中 1 个 keyword
    2. Hit Rate (严格): 前 k 个结果中至少命中 2 个 keyword（排除随机命中）
    3. Keyword Coverage: 检索结果覆盖了多少比例的 keywords
    4. MRR: 第一个命中结果的排名倒数平均值
    """
    hits_loose = {k: 0 for k in [1, 3, 5]}
    hits_strict = {k: 0 for k in [1, 3, 5]}
    rr_sum = 0.0
    kw_coverage_sum = 0.0
    total = len(queries)
    per_query = []

    for q in queries:
        docs = retriever.retrieve(q["query"], top_k=top_k)
        keywords = q["keywords"]

        # 统计所有检索结果中覆盖了多少 keywords
        all_text = " ".join(d.page_content for d in docs)
        matched_kws = [kw for kw in keywords if kw in all_text]
        kw_coverage = len(matched_kws) / len(keywords)
        kw_coverage_sum += kw_coverage

        # 在排序结果中找命中位置
        loose_matched = False
        strict_matched = False

        for rank, doc in enumerate(docs, 1):
            matched_in_doc = [kw for kw in keywords if kw in doc.page_content]

            # 宽松：至少 1 个 keyword
            if not loose_matched and len(matched_in_doc) >= 1:
                loose_matched = True
                rr_sum += 1.0 / rank
                for k in hits_loose:
                    if rank <= k:
                        hits_loose[k] += 1

            # 严格：至少 2 个 keyword（排除巧合命中）
            if not strict_matched and len(matched_in_doc) >= 2:
                strict_matched = True
                for k in hits_strict:
                    if rank <= k:
                        hits_strict[k] += 1

            if loose_matched and strict_matched:
                break

        per_query.append({
            "id": q["id"],
            "query": q["query"],
            "keywords_total": len(keywords),
            "keywords_matched": len(matched_kws),
            "coverage": round(kw_coverage, 2),
            "loose_hit": loose_matched,
            "strict_hit": strict_matched,
        })

    return {
        "hit_rate@1": hits_loose[1] / total,
        "hit_rate@3": hits_loose[3] / total,
        "hit_rate@5": hits_loose[5] / total,
        "strict_hit_rate@3": hits_strict[3] / total,
        "strict_hit_rate@5": hits_strict[5] / total,
        "mrr": rr_sum / total,
        "avg_keyword_coverage": round(kw_coverage_sum / total, 3),
        "total": total,
        "per_query": per_query,
    }


# ============================================================
# 4. RAGAS 生成质量评测
# ============================================================

def evaluate_ragas(
    retriever,
    queries: list[dict],
    vs: VectorStoreService,
    sample_size: int = 10,
):
    """
    使用 RAGAS 评测生成质量（Faithfulness / Context Relevance / Answer Relevance）

    从 30 条中采样 sample_size 条，用 LLM 生成回答，RAGAS 自动评分。
    """
    from rag.rag_service import RagSummarizeService
    from model.factory import chat_model

    rag = RagSummarizeService(use_hybrid=True)

    # 采样
    samples = queries[:sample_size]

    questions = []
    answers = []
    contexts_list = []

    print(f"\n  RAGAS 评测: 处理 {len(samples)} 条样本...")
    for q in samples:
        docs = retriever.retrieve(q["query"], top_k=3)
        answer = rag.rag_summarize(q["query"])
        questions.append(q["query"])
        answers.append(answer)
        contexts_list.append([doc.page_content for doc in docs])
        print(f"    [{q['id']}] {q['query'][:30]}... OK")

    # 构建 RAGAS Dataset
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, context_relevancy, answer_relevancy
        from ragas.llms import LangchainLLMWrapper
        from datasets import Dataset

        judge_llm = LangchainLLMWrapper(chat_model)

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
        })

        print(f"\n  RAGAS 自动评分中（LLM Judge: qwen3.7-max）...")
        result = evaluate(
            dataset,
            metrics=[faithfulness, context_relevancy, answer_relevancy],
            llm=judge_llm,
        )

        return result

    except ImportError as e:
        print(f"  [WARNING] RAGAS import failed: {e}")
        return None
    except Exception as e:
        print(f"  [WARNING] RAGAS evaluation failed: {e}")
        return None


# ============================================================
# 5. 主流程
# ============================================================

def main():
    print("=" * 62)
    print("  智扫通 RAG 评测系统")
    print("=" * 62)

    # 初始化
    print("\n[1/4] 初始化向量库和检索器...")
    vs = VectorStoreService()
    doc_count = vs.get_document_count()
    print(f"  向量库文档数: {doc_count}")

    queries = load_test_queries()
    print(f"  测试 query 数: {len(queries)}")

    # 场景分布
    cats = defaultdict(int)
    for q in queries:
        cats[q["category"]] += 1
    for c, n in cats.items():
        print(f"    {c}: {n} 条")

    # 四路检索器
    retrievers = {
        "Dense Only":         DenseRetriever(vs),
        "BM25 Only":          BM25Retriever(vs),
        "Hybrid (RRF)":       HybridRetriever(vs.vector_store, use_rerank=False),
        "Hybrid + Rerank":    HybridRetriever(vs.vector_store, use_rerank=True),
    }

    # ---- 检索指标 ----
    print("\n[2/4] 检索指标评测 (Hit Rate + MRR)...")

    results = {}
    for name, retriever in retrievers.items():
        start = time.time()
        r = evaluate_retrieval(retriever, queries, top_k=5)
        elapsed = time.time() - start
        results[name] = r
        print(f"  {name:16s}  宽松HR@5={r['hit_rate@5']:.1%}  "
              f"严格HR@5={r['strict_hit_rate@5']:.1%}  "
              f"KW覆盖={r['avg_keyword_coverage']:.1%}  MRR={r['mrr']:.4f}  ({elapsed:.0f}s)")

    # ---- 对比表格 ----
    print("\n[3/4] 三路对比汇总")
    print("-" * 72)
    print(f"  {'Retriever':<18s} {'宽松HR@5':>9s} {'严格HR@5':>9s} {'KW覆盖':>8s} {'MRR':>7s}")
    print("-" * 72)
    for name, r in results.items():
        print(f"  {name:<18s} {r['hit_rate@5']:>8.1%} {r['strict_hit_rate@5']:>8.1%} "
              f"{r['avg_keyword_coverage']:>7.1%} {r['mrr']:>7.4f}")
    print("-" * 72)

    # Rerank 相对于 Dense/RRF 的提升
    dense = results["Dense Only"]
    bm25 = results["BM25 Only"]
    rrf = results["Hybrid (RRF)"]
    rerank = results["Hybrid + Rerank"]
    strict_lift = (rerank["strict_hit_rate@5"] - dense["strict_hit_rate@5"]) / max(dense["strict_hit_rate@5"], 0.01) * 100
    kw_lift = (rerank["avg_keyword_coverage"] - dense["avg_keyword_coverage"]) / max(dense["avg_keyword_coverage"], 0.01) * 100
    rrf_to_rerank = (rerank["strict_hit_rate@5"] - rrf["strict_hit_rate@5"]) / max(rrf["strict_hit_rate@5"], 0.01) * 100
    print(f"  Rerank 相对 Dense:  严格命中 +{strict_lift:.0f}%  关键词覆盖 +{kw_lift:.0f}%")
    print(f"  Rerank 相对 RRF:    严格命中 +{rrf_to_rerank:.0f}%")

    # 汇总结果
    eval_summary = {
        "system": "智扫通 RAG 检索评测",
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_queries_count": len(queries),
        "vector_store_documents": doc_count,
        "categories": dict(cats),
        "results": {
            name: {
                "hit_rate@1": round(r["hit_rate@1"], 4),
                "hit_rate@3": round(r["hit_rate@3"], 4),
                "hit_rate@5": round(r["hit_rate@5"], 4),
                "strict_hit_rate@3": round(r["strict_hit_rate@3"], 4),
                "strict_hit_rate@5": round(r["strict_hit_rate@5"], 4),
                "mrr": round(r["mrr"], 4),
                "avg_keyword_coverage": round(r["avg_keyword_coverage"], 4),
            }
            for name, r in results.items()
        },
        "rerank_vs_dense": {
            "strict_hit_rate@5_lift_pct": round(strict_lift, 1),
            "kw_coverage_lift_pct": round(kw_lift, 1),
        },
        "rerank_vs_rrf": {
            "strict_hit_rate@5_lift_pct": round(rrf_to_rerank, 1),
        },
    }

    # 保存结果到文件
    output_path = Path(__file__).parent / "eval_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, ensure_ascii=False, indent=2)
    print(f"\n  详细结果已保存: {output_path}")

    # ---- RAGAS ----
    print("\n[4/4] RAGAS 生成质量评测...")
    try:
        ragas_result = evaluate_ragas(
            HybridRetriever(vs.vector_store),
            queries,
            vs,
            sample_size=10,
        )
        if ragas_result is not None:
            print(f"\n  {'指标':<22s} {'分数':>6s}")
            print("  " + "-" * 28)
            # RAGAS result is a dict-like object
            for key, value in ragas_result.items():
                print(f"  {key:<22s} {value:>6.4f}")
    except Exception as e:
        print(f"  RAGAS evaluation skipped: {e}")

    print("\n" + "=" * 62)
    print("  评测完成")
    print("=" * 62)


if __name__ == "__main__":
    main()
