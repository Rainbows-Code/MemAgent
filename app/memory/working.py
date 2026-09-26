import json
import logging
import time
from typing import Dict, List, Any, Optional
import redis
from app.config import settings

logger = logging.getLogger(__name__)


class WorkingMemory:
    """
    Redis 工作记忆 (Working Memory) 管理类
    存储格式: Redis LIST (key: chat:{user_id}:working)
    单个元素: JSON 字符串 {"role": "user"/"assistant", "content": str, "timestamp": float}
    """

    def __init__(
        self,
        user_id: str = "default_user",
        window_size: int = settings.WORKING_MEMORY_WINDOW_SIZE,
        ttl_seconds: int = settings.WORKING_MEMORY_TTL_SECONDS,
        redis_client: Optional[redis.Redis] = None,
    ):
        self.user_id = user_id
        self.window_size = window_size  # 保留近 N 轮 (即 2N 条消息)
        self.ttl_seconds = ttl_seconds
        self.key = f"chat:{self.user_id}:working"

        # 尝试连接 Redis
        self._fallback_memory: List[Dict[str, Any]] = []
        if redis_client is not None:
            self.redis = redis_client
            self.is_redis_available = True
        else:
            try:
                self.redis = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    password=settings.REDIS_PASSWORD or None,
                    decode_responses=True,
                    socket_timeout=0.5,
                    socket_connect_timeout=0.5,
                )
                self.redis.ping()
                self.is_redis_available = True
            except (redis.ConnectionError, redis.TimeoutError, Exception) as e:
                logger.warning(
                    f"无法连接到 Redis ({e})，WorkingMemory 将自动降级使用内存存储 fallback。"
                )
                self.redis = None
                self.is_redis_available = False

    def add_message(self, role: str, content: str) -> None:
        """
        追加一条新消息到工作记忆中，自动续期 TTL 并保持窗口裁剪
        """
        msg_obj = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
        }

        if self.is_redis_available and self.redis:
            try:
                # 1. 序列化并右压入 LIST
                json_str = json.dumps(msg_obj, ensure_ascii=False)
                self.redis.rpush(self.key, json_str)

                # 2. 保持最大 2 * window_size 条消息 (LTRIM 保留末尾部分)
                max_messages = self.window_size * 2
                self.redis.ltrim(self.key, -max_messages, -1)

                # 3. 自动续期 TTL
                self.redis.expire(self.key, self.ttl_seconds)
                return
            except redis.RedisError as e:
                logger.error(f"Redis 追加消息失败 ({e})，回退到内存模式")
                self.is_redis_available = False
                self._fallback_memory = self.get_history()

        # Fallback 内存处理
        self._fallback_memory.append(msg_obj)
        max_messages = self.window_size * 2
        if len(self._fallback_memory) > max_messages:
            self._fallback_memory = self._fallback_memory[-max_messages:]

    def get_history(self) -> List[Dict[str, Any]]:
        """
        获取当前工作记忆中的历史消息列表 (按时间正序)
        """
        if self.is_redis_available and self.redis:
            try:
                raw_list = self.redis.lrange(self.key, 0, -1)
                messages = []
                for item in raw_list:
                    messages.append(json.loads(item))
                return messages
            except redis.RedisError as e:
                logger.error(f"读取 Redis 工作记忆失败 ({e})")
                self.is_redis_available = False

        return list(self._fallback_memory)

    def clear(self) -> None:
        """
        清空当前用户的 Redis 工作记忆
        """
        if self.is_redis_available and self.redis:
            try:
                self.redis.delete(self.key)
            except redis.RedisError as e:
                logger.error(f"清空 Redis 工作记忆失败 ({e})")

        if hasattr(self, "_fallback_memory"):
            self._fallback_memory.clear()

    def get_context_messages(self, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """
        构建用于直接提交给 LLM 的 messages 数组（包含可选的 System Prompt）
        """
        history = self.get_history()
        result = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})
        for msg in history:
            result.append({"role": msg["role"], "content": msg["content"]})
        return result
