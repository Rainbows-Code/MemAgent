import logging
from typing import Dict, List, Any, Optional
from app.memory.working import WorkingMemory
from app.memory.episodic import EpisodicMemory
from app.memory.semantic import SemanticMemory

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """
    记忆检索与按需注入模块 (Memory Retrieval & Dynamic Prompt Injection)
    按需调度三层记忆 (Working, Episodic, Semantic)，防止每轮全量历史注入
    """

    def __init__(
        self,
        working_memory: Optional[WorkingMemory] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
        semantic_memory: Optional[SemanticMemory] = None,
    ):
        self.working_memory = working_memory
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory

    async def retrieve_context(
        self,
        user_id: str,
        current_query: str,
        top_k_episodic: int = 3,
        min_episodic_similarity: float = 0.2,
    ) -> Dict[str, Any]:
        """
        按需检索三层记忆内容

        :return: {
            "recent_messages": list,          # 工作记忆: 最近 N 轮
            "retrieved_summaries": list,       # 情景记忆: 向量相似 Top-K 摘要
            "user_facts": list,               # 语义记忆: 用户偏好与约束事实
            "injected_memory_meta": dict,     # 审计元数据 (记录本次注入了哪些记忆 ID)
        }
        """
        # 1. 工作记忆检索 (最近 N 轮)
        recent_messages = []
        if self.working_memory:
            recent_messages = self.working_memory.get_history()

        # 2. 情景记忆检索 (pgvector 向量 Cosine Top-K 摘要)
        retrieved_summaries = []
        if self.episodic_memory:
            retrieved_summaries = await self.episodic_memory.search_memory(
                user_id=user_id,
                query=current_query,
                top_k=top_k_episodic,
                min_similarity=min_episodic_similarity,
            )

        # 3. 语义记忆检索 (用户所有结构化偏好与约束)
        user_facts = []
        if self.semantic_memory:
            user_facts = await self.semantic_memory.get_all_facts(user_id)

        # 4. 构建审计元信息 (便于离线评估冗余率与命中率)
        meta = {
            "working_message_count": len(recent_messages),
            "episodic_summary_ids": [s["id"] for s in retrieved_summaries],
            "semantic_fact_keys": [f["key"] for f in user_facts],
        }

        return {
            "recent_messages": recent_messages,
            "retrieved_summaries": retrieved_summaries,
            "user_facts": user_facts,
            "injected_memory_meta": meta,
        }

    def build_injected_prompt(
        self,
        base_system_prompt: str,
        retrieved_context: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        根据按需检索出的记忆构建精简高效的 LLM Messages 列表
        只注入强相关的“语义记忆偏好”与“情景记忆摘要”，而非全量对话历史
        """
        user_facts = retrieved_context.get("user_facts", [])
        retrieved_summaries = retrieved_context.get("retrieved_summaries", [])
        recent_messages = retrieved_context.get("recent_messages", [])

        # 1. 组装动能 Prompt 区块
        system_parts = [base_system_prompt]

        if user_facts:
            system_parts.append("\n【用户长效偏好与约束 (语义记忆)】:")
            for fact in user_facts:
                system_parts.append(f"- {fact['key']}: {fact['value']} (置信度: {fact.get('confidence', 1.0)})")

        if retrieved_summaries:
            system_parts.append("\n【相关历史会话经验 (情景记忆)】:")
            for s in retrieved_summaries:
                system_parts.append(f"- [相似度 {s.get('similarity', 0.0)}]: {s['summary']}")

        combined_system_prompt = "\n".join(system_parts)

        # 2. 组装 Messages 结构
        messages = [{"role": "system", "content": combined_system_prompt}]

        # 3. 拼接最近工作记忆对话
        for msg in recent_messages:
            messages.append({"role": msg["role"], "content": msg["content"]})

        return messages
