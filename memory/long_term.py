"""
长期记忆：对话摘要 → 向量库存储 → 语义检索

当短期记忆窗口溢出时，自动将旧对话压缩为摘要存入向量库，
后续用户提问时可以语义检索相关历史记忆。
"""
import json
import time
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class LongTermMemory:
    """
    长期记忆系统

    工作流程:
        1. 短期记忆溢出 → summarize_and_store() 生成摘要存入 Chroma
        2. 用户新提问 → retrieve_relevant() 语义检索相关历史摘要
    """

    def __init__(
        self,
        embedding_model,
        collection_name: str = "long_term_memory",
        persist_directory: str = "chroma_db",
        similarity_k: int = 3,
    ):
        """
        Args:
            embedding_model: Embeddings 实例（复用现有 DashScopeEmbeddings）
            collection_name: Chroma collection 名称
            persist_directory: 持久化目录
            similarity_k: 检索返回的相关记忆数
        """
        self.embedding_model = embedding_model
        self.similarity_k = similarity_k

        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=get_abs_path(persist_directory),
        )

    async def summarize_and_store(
        self,
        session_id: str,
        messages: list[dict],
    ) -> str:
        """
        将一段对话压缩为摘要并存入向量库

        Args:
            session_id: 会话ID
            messages: 要归档的消息列表 (json 格式)

        Returns:
            生成的摘要文本
        """
        if not messages:
            return ""

        try:
            # 拼接对话文本
            conversation = ""
            for msg in messages:
                role_label = "用户" if msg["role"] == "user" else "助手"
                conversation += f"{role_label}: {msg['content']}\n"

            # 用简单规则生成摘要（生产环境可改为调用 LLM 摘要）
            summary = self._generate_summary(conversation, messages)

            # 存入向量库
            metadata = {
                "session_id": session_id,
                "timestamp": time.time(),
                "message_count": len(messages),
                "type": "conversation_summary",
            }

            doc = Document(page_content=summary, metadata=metadata)
            self.vector_store.add_documents([doc])

            logger.info(
                f"[LongTerm] 已存储摘要 session={session_id}, "
                f"消息数={len(messages)}, 摘要长度={len(summary)}"
            )
            return summary
        except Exception as e:
            logger.error(f"[LongTerm] 摘要存储失败: {str(e)}")
            return ""

    def _generate_summary(self, conversation: str, messages: list[dict]) -> str:
        """
        生成对话摘要（规则化实现）

        注：生产环境可接入 LLM 生成更精准的摘要，如:
            chain = ChatPromptTemplate.from_template(
                "请用中文将以下对话压缩为一句话摘要:\n{conversation}"
            ) | model | StrOutputParser()
            return chain.invoke({"conversation": conversation})
        """
        # 取首尾消息提取关键信息
        if not messages:
            return "空对话"

        first_user_msg = ""
        last_msg = ""
        for msg in messages:
            if msg["role"] == "user" and not first_user_msg:
                first_user_msg = msg["content"]
            last_msg = msg["content"]

        # 简单的规则化摘要
        topic = first_user_msg[:80] if first_user_msg else "无"
        summary = (
            f"[对话摘要] 用户提问: {topic}... | "
            f"共 {len(messages)} 条消息 | "
            f"最后回复: {last_msg[:60]}..."
        )
        return summary

    async def retrieve_relevant(
        self,
        query: str,
        k: int = None,
    ) -> list[dict]:
        """
        检索与当前查询相关的历史记忆

        Args:
            query: 当前用户查询
            k: 返回的记忆数量

        Returns:
            相关记忆列表 [{summary, metadata, similarity_score}]
        """
        if k is None:
            k = self.similarity_k

        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)

            memories = []
            for doc, score in results:
                memories.append({
                    "summary": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score,
                })

            if memories:
                logger.info(
                    f"[LongTerm] 检索到 {len(memories)} 条相关记忆, "
                    f"最高分={memories[0]['similarity_score']:.4f}"
                )

            return memories
        except Exception as e:
            logger.warning(f"[LongTerm] 检索失败: {str(e)}")
            return []

    def format_for_prompt(self, memories: list[dict]) -> str:
        """将长期记忆格式化为提示词可用的文本"""
        if not memories:
            return ""

        lines = ["## 相关历史记忆", ""]
        for i, m in enumerate(memories, 1):
            lines.append(f"{i}. {m['summary']}")
        lines.append("---")
        return "\n".join(lines)

    def get_memory_count(self) -> int:
        """获取已存储的长期记忆数量"""
        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0
