"""
短期记忆：Redis 滑动窗口

每个 session 一个 Redis List，存储最近 N 轮对话消息。
Key 格式: session:{session_id}:messages
"""
import json
import time
from typing import Optional
from utils.logger_handler import logger


class ShortTermMemory:
    """
    Redis 短期记忆

    存储结构:
        Redis List: session:{session_id}:messages
        每条消息 JSON: {"role": "user/assistant", "content": "...", "timestamp": ...}

    使用 LPUSH + LTRIM 保持滑动窗口。
    """

    def __init__(
        self,
        redis_client,
        max_rounds: int = 10,
        ttl_hours: int = 24,
        archive_threshold: int = 10,
    ):
        """
        Args:
            redis_client: redis.asyncio.Redis 实例
            max_rounds: 最大保留轮数（每轮 = user + assistant 各一条）
            ttl_hours: Redis key 存活时间
            archive_threshold: 超过此轮数触发长期记忆归档
        """
        self.redis = redis_client
        self.max_rounds = max_rounds
        self.max_messages = max_rounds * 2  # user + assistant 各算一条
        self.ttl_seconds = ttl_hours * 3600
        self.archive_threshold = archive_threshold

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}:messages"

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        添加一条消息到会话窗口

        Args:
            session_id: 会话ID
            role: "user" 或 "assistant"
            content: 消息内容
        """
        try:
            key = self._key(session_id)
            msg = json.dumps({
                "role": role,
                "content": content,
                "timestamp": time.time(),
            }, ensure_ascii=False)

            await self.redis.lpush(key, msg)
            await self.redis.ltrim(key, 0, self.max_messages - 1)  # 保持窗口
            await self.redis.expire(key, self.ttl_seconds)

            logger.debug(f"[ShortTerm] 会话 {session_id} 新增 {role} 消息")
        except Exception as e:
            logger.warning(f"[ShortTerm] 添加消息失败 (非致命): {str(e)}")

    async def get_history(self, session_id: str) -> list[dict]:
        """
        获取最近 N 轮对话历史

        Args:
            session_id: 会话ID

        Returns:
            消息列表，按时间顺序排列 (最早在前)
        """
        try:
            key = self._key(session_id)
            messages = await self.redis.lrange(key, 0, -1)
            result = [json.loads(m) for m in messages]
            result.reverse()  # LPUSH 是最新在前，翻转为时间顺序
            return result
        except Exception as e:
            logger.warning(f"[ShortTerm] 获取历史失败 (非致命): {str(e)}")
            return []

    async def get_message_count(self, session_id: str) -> int:
        """获取当前会话的消息数"""
        try:
            key = self._key(session_id)
            return await self.redis.llen(key)
        except Exception:
            return 0

    async def should_archive(self, session_id: str) -> bool:
        """检查是否应该触发长期记忆归档"""
        count = await self.get_message_count(session_id)
        return count >= self.archive_threshold * 2

    async def pop_oldest_rounds(self, session_id: str, rounds: int = 3) -> list[dict]:
        """
        弹出最旧的 N 轮对话（用于归档到长期记忆）

        Args:
            session_id: 会话ID
            rounds: 要弹出的轮数

        Returns:
            被弹出的消息列表（时间顺序）
        """
        try:
            key = self._key(session_id)
            pop_count = rounds * 2

            # Redis List 尾部是最旧的消息（LPUSH + LTRIM 保持新消息在头部）
            # 用 RPOP 弹出最旧的消息
            old_messages = []
            for _ in range(pop_count):
                msg_raw = await self.redis.rpop(key)
                if msg_raw:
                    old_messages.append(json.loads(msg_raw))

            logger.info(f"[ShortTerm] 会话 {session_id} 归档 {len(old_messages)} 条旧消息到长期记忆")
            return old_messages
        except Exception as e:
            logger.warning(f"[ShortTerm] 弹出旧消息失败: {str(e)}")
            return []

    async def clear(self, session_id: str) -> None:
        """清除某个会话的所有记忆"""
        try:
            key = self._key(session_id)
            await self.redis.delete(key)
            logger.info(f"[ShortTerm] 已清除会话 {session_id}")
        except Exception as e:
            logger.warning(f"[ShortTerm] 清除会话失败: {str(e)}")

    def format_for_prompt(self, messages: list[dict]) -> str:
        """将历史消息列表格式化为提示词可用的文本"""
        if not messages:
            return ""

        lines = ["## 历史对话记录"]
        for msg in messages:
            role_label = "用户" if msg["role"] == "user" else "客服"
            lines.append(f"{role_label}: {msg['content']}")
        lines.append("---")
        return "\n".join(lines)
