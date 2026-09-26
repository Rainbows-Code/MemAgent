import math
import logging
import time
from typing import Dict, List, Any, Optional
from app.config import settings
from app.llm import LLMClient, llm_client
from app.memory.episodic import EpisodicMemory
from app.memory.semantic import SemanticMemory

logger = logging.getLogger(__name__)

IMPORTANCE_WEIGHTS = {
    "user_explicit": 1.0,  # 用户明示偏好
    "tool_result": 0.7,    # 工具调用结果
    "system_derived": 0.5, # 系统根据经验推导
    "chitchat": 0.3,       # 闲聊对话
}


class MemoryForgettingEngine:
    """
    记忆写入与遗忘策略引擎 (Memory Writing & Forgetting Policy Engine)
    包含: 重要度打分 + 时间衰减 + 冲突覆盖归档 + 摘要压缩 + 记忆淘汰/瘦身
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
        semantic_memory: Optional[SemanticMemory] = None,
        forgetting_threshold: float = settings.FORGETTING_THRESHOLD,
        decay_lambda: float = settings.TIME_DECAY_LAMBDA,
    ):
        self.llm = llm or llm_client
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.forgetting_threshold = forgetting_threshold
        self.decay_lambda = decay_lambda

    def calculate_importance_score(self, source: str, custom_score: Optional[float] = None) -> float:
        """
        根据信息来源计算初始重要度打分
        """
        if custom_score is not None:
            return max(0.0, min(1.0, custom_score))
        return IMPORTANCE_WEIGHTS.get(source, 0.5)

    def calculate_time_decay_score(self, initial_score: float, created_at_timestamp: float) -> float:
        """
        根据指数衰减公式计算记忆的动态真实得分: Score(t) = Score0 * exp(-lambda * delta_days)
        """
        now = time.time()
        delta_seconds = max(0.0, now - created_at_timestamp)
        delta_days = delta_seconds / 86400.0
        decayed_score = initial_score * math.exp(-self.decay_lambda * delta_days)
        return round(decayed_score, 4)

    async def compress_conversation_summary(
        self,
        user_id: str,
        recent_messages: List[Dict[str, str]],
    ) -> Optional[str]:
        """
        摘要压缩: 将最近 N 轮对话通过 LLM 做精炼压缩，并存入 pgvector 情景记忆
        """
        if not recent_messages or len(recent_messages) < 2:
            return None

        prompt_messages = [
            {
                "role": "system",
                "content": "你是一个对话摘要提取专家。请简明扼要地将以下用户与助手的对话总结为1-2句核心偏好与关键事件摘要。",
            },
            {
                "role": "user",
                "content": f"对话内容如下:\n" + "\n".join([f"{m['role']}: {m['content']}" for m in recent_messages]),
            }
        ]

        try:
            res = await self.llm.achat_completion(prompt_messages, temperature=0.3)
            summary = res["content"].strip()

            if summary and self.episodic_memory:
                # 计算重要度 (通常由摘要产生的经验重要度取 0.7)
                await self.episodic_memory.add_memory(
                    user_id=user_id,
                    summary=summary,
                    importance=0.7,
                )
                logger.info(f"生成并存储压缩摘要 [{user_id}]: {summary}")
            return summary
        except Exception as e:
            logger.error(f"摘要压缩失败: {e}")
            return None

    async def prune_forgotten_memories(
        self,
        user_id: str,
        facts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        根据衰减打分淘汰低重要度记忆 (低于 forgetting_threshold 且非用户明示偏好的冗余数据)
        :return: {"retained": list, "pruned": list, "prune_rate": float}
        """
        retained = []
        pruned = []

        for fact in facts:
            source = fact.get("source", "system_derived")
            init_score = self.calculate_importance_score(source, fact.get("confidence"))

            # 从 updated_at 提取时间戳
            updated_at_ts = time.time()
            decayed_score = self.calculate_time_decay_score(init_score, updated_at_ts)

            # 用户明示偏好 (user_explicit) 免疫自动遗忘；低于阈值的辅助记忆将被淘汰
            if source == "user_explicit" or decayed_score >= self.forgetting_threshold:
                retained.append(fact)
            else:
                pruned.append(fact)

        total = len(facts)
        prune_rate = (len(pruned) / total) if total > 0 else 0.0

        return {
            "retained": retained,
            "pruned": pruned,
            "prune_rate": round(prune_rate, 4),
        }
