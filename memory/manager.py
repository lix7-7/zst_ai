"""
MemoryManager — 记忆系统统一入口

协调 ShortTermMemory (Redis) 和 LongTermMemory (Chroma 向量库)
对外提供核心接口:
    add_interaction()   — 记录一轮对话
    get_context()       — 获取记忆上下文（用于注入 agent 提示词）
    clear()             — 清除会话记忆
    get_user_sessions() — 获取用户的所有会话列表
"""
import asyncio
import time
from typing import Optional
import redis.asyncio as aioredis
from model.factory import embed_model
from utils.config_handler import load_yaml_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory


class MemoryManager:
    """
    记忆系统统一管理器

    使用方式:
        mgr = MemoryManager()
        await mgr.add_interaction(session_id, user_msg, assistant_msg)
        context = await mgr.get_context(session_id, current_query)
    """

    _instance: Optional["MemoryManager"] = None

    def __init__(self):
        self._redis_client = None
        self.short_term: Optional[ShortTermMemory] = None
        self.long_term: Optional[LongTermMemory] = None
        self._initialized = False

    async def initialize(self):
        """初始化 Redis 连接和记忆模块"""
        if self._initialized:
            return

        try:
            memory_conf = load_yaml_config("config/memory.yml")

            redis_conf = memory_conf.get("redis", {})
            short_term_conf = memory_conf.get("short_term", {})
            long_term_conf = memory_conf.get("long_term", {})

            # 连接 Redis
            self._redis_client = aioredis.Redis(
                host=redis_conf.get("host", "localhost"),
                port=redis_conf.get("port", 6379),
                db=redis_conf.get("db", 0),
                password=redis_conf.get("password") or None,
                decode_responses=True,
            )

            # ping 测试连接
            await self._redis_client.ping()
            logger.info("[Memory] Redis 连接成功")

            # 初始化短期记忆
            self.short_term = ShortTermMemory(
                redis_client=self._redis_client,
                max_rounds=short_term_conf.get("max_rounds", 10),
                ttl_hours=short_term_conf.get("ttl_hours", 24),
                archive_threshold=short_term_conf.get("archive_threshold", 10),
            )

            # 初始化长期记忆
            self.long_term = LongTermMemory(
                embedding_model=embed_model,
                collection_name=long_term_conf.get("collection_name", "long_term_memory"),
                persist_directory=long_term_conf.get("persist_directory", "chroma_db"),
                similarity_k=long_term_conf.get("similarity_k", 3),
            )

            self._initialized = True
            logger.info("[Memory] MemoryManager 初始化完成")

        except Exception as e:
            logger.warning(f"[Memory] Redis 连接失败，记忆功能不可用: {str(e)}")
            self.short_term = ShortTermMemory(None, max_rounds=10)
            self.long_term = LongTermMemory(embed_model)
            self._initialized = True  # 降级模式仍然可用（短期记忆返回空）

    async def add_interaction(
        self,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
        user_id: str = None,
    ) -> None:
        """
        记录一轮完整对话

        Args:
            session_id: 会话ID
            user_msg: 用户消息
            assistant_msg: 助手回复
            user_id: 可选，用户标识。传入后自动更新会话元数据
        """
        await self._ensure_initialized()

        # 写入短期记忆
        await self.short_term.add_message(session_id, "user", user_msg)
        await self.short_term.add_message(session_id, "assistant", assistant_msg)

        # 检查是否需要归档到长期记忆
        if await self.short_term.should_archive(session_id):
            old_msgs = await self.short_term.pop_oldest_rounds(session_id, rounds=3)
            if old_msgs:
                # 异步归档，不阻塞当前请求
                asyncio.create_task(self.long_term.summarize_and_store(session_id, old_msgs))

        # 更新会话元数据 + 刷新用户索引 TTL
        if user_id and self.short_term.redis is not None:
            try:
                msg_count = await self.short_term.get_message_count(session_id)
                await self.short_term.update_session_meta(
                    session_id,
                    last_active=time.time(),
                    message_count=msg_count,
                )
                await self.short_term.redis.expire(
                    self.short_term._user_key(user_id),
                    self.short_term.ttl_seconds,
                )
            except Exception:
                pass

    async def get_context(self, session_id: str, current_query: str) -> str:
        """
        获取记忆上下文，用于注入 Agent 系统提示词

        Args:
            session_id: 会话ID
            current_query: 当前用户问题（用于检索相关长期记忆）

        Returns:
            格式化的记忆上下文字符串
        """
        await self._ensure_initialized()

        parts = []

        # 短期记忆：最近对话
        short_history = await self.short_term.get_history(session_id)
        if short_history:
            parts.append(self.short_term.format_for_prompt(short_history))

        # 长期记忆：检索相关历史摘要
        long_memories = await self.long_term.retrieve_relevant(current_query)
        relevant = [m for m in long_memories if m["similarity_score"] < 1.0]
        if relevant:
            parts.append(self.long_term.format_for_prompt(relevant))

        context = "\n".join(parts) if parts else ""
        if context:
            logger.debug(f"[Memory] 上下文注入 session={session_id}, 长度={len(context)}")

        return context

    async def clear(self, session_id: str, user_id: str = None) -> None:
        """
        清除某个会话的所有记忆

        Args:
            session_id: 会话ID
            user_id: 可选，传入后同时从用户索引中移除
        """
        await self._ensure_initialized()
        await self.short_term.clear(session_id)
        if user_id:
            await self.short_term.unregister_session(session_id, user_id)
        logger.info(f"[Memory] 已清除会话 {session_id} 的全部记忆")

    # ============================================================
    # 用户-会话关联 API
    # ============================================================

    async def register_user_session(
        self, user_id: str, session_id: str, preview: str = ""
    ) -> None:
        """注册会话归属（SADD + HSET 元数据）"""
        await self._ensure_initialized()
        await self.short_term.register_session(user_id, session_id, preview)

    async def update_session_meta(self, session_id: str, **updates) -> None:
        """增量更新会话元数据"""
        await self._ensure_initialized()
        await self.short_term.update_session_meta(session_id, **updates)

    async def get_user_sessions(self, user_id: str) -> list[dict]:
        """
        获取某用户的所有会话（按 last_active 降序）

        Returns:
            [{session_id, user_id, preview, last_active, message_count, created_at}]
        """
        await self._ensure_initialized()
        if self.short_term.redis is None:
            return []
        session_ids = await self.short_term.get_user_session_ids(user_id)
        result = []
        for sid in session_ids:
            meta = await self.short_term.get_session_meta(sid)
            if meta:
                meta["session_id"] = sid
                result.append(meta)
        result.sort(key=lambda s: s.get("last_active", 0), reverse=True)
        return result

    async def _ensure_initialized(self):
        if not self._initialized:
            await self.initialize()

    async def close(self):
        """关闭 Redis 连接"""
        if self._redis_client:
            await self._redis_client.close()
            logger.info("[Memory] Redis 连接已关闭")

    @classmethod
    def get_instance(cls) -> "MemoryManager":
        """获取单例"""
        if cls._instance is None:
            cls._instance = MemoryManager()
        return cls._instance
