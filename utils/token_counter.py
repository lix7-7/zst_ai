"""
Token 估算工具

主方案: tiktoken + cl100k_base（中文 ≈ 1~2 token/char，估算偏差 ±15%）
Fallback: 字符启发式（中文/1.5 + 英文/0.75，偏差 ±30%）

用于 TokenGuard 中间件的窗口守卫，只需估算，不需要精确到字节。
"""
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from utils.logger_handler import logger
from utils.config_handler import load_yaml_config


# ============================================================
# TokenBudget
# ============================================================

@dataclass
class TokenBudget:
    """一次 token 预算快照"""
    system_tokens: int = 0          # system prompt（含注入的历史）
    messages_tokens: int = 0        # 消息列表（不含 system）
    total: int = 0                  # 合计
    limit: int = 8000               # 阈值
    available: int = 0              # 剩余可用

    @property
    def is_over_budget(self) -> bool:
        return self.total > self.limit


# ============================================================
# TokenCounter
# ============================================================

class TokenCounter:
    """
    Token 估算器

    使用方式:
        counter = TokenCounter()
        count = counter.count("你好世界")
        is_over, budget = counter.check(system_prompt, messages)
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or _load_token_config()
        self.max_tokens = cfg.get("max_context_tokens", 8000)
        self.reserved = cfg.get("reserved_response_tokens", 2000)
        self.safety_margin = cfg.get("safety_margin", 0.15)
        self.limit = int(self.max_tokens * (1 - self.safety_margin))

        # 尝试加载 tiktoken
        self._encoder = self._init_encoder()

    def _init_encoder(self):
        """尝试加载 tiktoken，失败则返回 None（使用 heuristic fallback）"""
        try:
            import tiktoken
            encoder = tiktoken.get_encoding("cl100k_base")
            logger.info("[TokenCounter] 使用 tiktoken (cl100k_base) 进行 token 估算")
            return encoder
        except ImportError:
            logger.warning("[TokenCounter] tiktoken 未安装，使用字符启发式估算（偏差 ±30%）")
            return None
        except Exception as e:
            logger.warning(f"[TokenCounter] tiktoken 加载失败: {e}，使用字符启发式估算")
            return None

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def count(self, text: str) -> int:
        """估算单段文本的 token 数"""
        if not text:
            return 0
        if self._encoder:
            return len(self._encoder.encode(text))
        return _heuristic_count(text)

    def count_messages(self, messages: list) -> int:
        """
        估算 LangChain 消息列表的 token 数

        仅统计消息的 content 字段，不含 metadata/type 等开销；
        实际 token 消耗会略高于估算值，安全 margin 在 limit 中已考虑。
        """
        total = 0
        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                content = msg.content
                if isinstance(content, str):
                    total += self.count(content)
                elif isinstance(content, list):
                    # 多模态 content（text + image_url 等）
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            total += self.count(block["text"])
        return total

    def check(self, system_prompt: str, messages: list) -> tuple[bool, TokenBudget]:
        """
        检查是否超出 token 预算

        Args:
            system_prompt: system 消息的完整内容（含注入的历史）
            messages:    system 消息之后的消息列表

        Returns:
            (是否超标, TokenBudget 详情)
        """
        system_tokens = self.count(system_prompt)
        messages_tokens = self.count_messages(messages)
        total = system_tokens + messages_tokens

        budget = TokenBudget(
            system_tokens=system_tokens,
            messages_tokens=messages_tokens,
            total=total,
            limit=self.limit,
            available=max(0, self.limit - total),
        )
        return total > self.limit, budget

    def count_history_block(self, text: str) -> int:
        """估算 '## 历史对话记录' 块的 token 数"""
        return self.count(text)


# ============================================================
# 字符启发式 fallback
# ============================================================

# 中文字符范围（含标点）
_CJK_RE = re.compile(r'[一-鿿㐀-䶿豈-﫿　-〿＀-￯]')
# 英文单词
_WORD_RE = re.compile(r'[a-zA-Z]+')


def _heuristic_count(text: str) -> int:
    """
    字符启发式 token 估算

    - 中文字符: ~1.5 token/char
    - 英文单词: ~1.3 token/word（含前后空格）
    - 其他（数字/标点/空白）: ~1 token/char
    """
    cjk_chars = len(_CJK_RE.findall(text))
    remaining = _CJK_RE.sub('', text)
    english_words = len(_WORD_RE.findall(remaining))
    other_chars = len(re.sub(r'\s', '', remaining)) - english_words

    tokens = cjk_chars / 1.5 + english_words / 0.75 + other_chars
    return max(1, int(tokens))


# ============================================================
# 单例 & 配置加载
# ============================================================

_counter_instance: Optional[TokenCounter] = None


def get_token_counter(config: Optional[dict] = None) -> TokenCounter:
    """获取 TokenCounter 单例"""
    global _counter_instance
    if _counter_instance is None:
        _counter_instance = TokenCounter(config)
    return _counter_instance


def reset_token_counter():
    """重置单例（测试用）"""
    global _counter_instance
    _counter_instance = None


@lru_cache(maxsize=1)
def _load_token_config() -> dict:
    """加载 token 窗口配置（缓存）"""
    try:
        memory_conf = load_yaml_config("config/memory.yml")
        return memory_conf.get("token_window", {})
    except Exception:
        return {}
