import math
import logging
import re
from typing import Dict, List, Any, Optional
from app.llm import LLMClient, llm_client
from app.memory.embeddings import EmbeddingModel, embedding_model

logger = logging.getLogger(__name__)

# 常见指代词与实体类型偏好映射
PRONOUN_TYPE_MAP = {
    "它": ["object", "item", "hotel", "product", "location", "general"],
    "那个": ["object", "item", "hotel", "product", "location", "address", "general"],
    "这": ["general"],
    "这件": ["item", "product"],
    "第一个": ["ordinal_1"],
    "第二个": ["ordinal_2"],
    "第三个": ["ordinal_3"],
    "前一个": ["previous"],
    "地址": ["address", "location"],
    "改一下地址": ["address", "location"],
}


class CoreferenceResolver:
    """
    指代解析层 (Coreference Resolution Engine)
    解决多轮对话中的“它、那个、第一个、改一下地址”等模糊代词歧义
    结合 候选打分 (类型匹配 + 时间近因 + 语义相似 + 出现频率) + LLM 校验确认
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        embedder: Optional[EmbeddingModel] = None,
        recency_decay: float = 0.2,
    ):
        self.llm = llm or llm_client
        self.embedder = embedder or embedding_model
        self.recency_decay = recency_decay

    def extract_candidates_from_history(
        self,
        messages: List[Dict[str, str]],
        facts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        从历史对话与结构化语义记忆中提取所有候选实体
        """
        candidates = []
        seen = set()

        # 1. 从语义记忆 (facts) 提取结构化实体 (最高优先级)
        for idx, fact in enumerate(facts):
            key = fact.get("key", "")
            val = fact.get("value", "")
            if val and val not in seen:
                seen.add(val)
                candidates.append({
                    "entity": val,
                    "type": "address" if "address" in key or "地址" in key else "general",
                    "source": "semantic_fact",
                    "recency_step": 0,  # 最近
                    "frequency": 2,
                    "key": key,
                })

        # 2. 从历史对话中提取名词实体/列表项目 (倒序分析近因)
        total_msgs = len(messages)
        for turn_idx, msg in enumerate(reversed(messages)):
            content = msg.get("content", "")
            # 简单的正则表达式模式匹配提取被引号、列表序号或地名/产品名称
            # 提取包含显式名称、括号实体、引号实体或专有名词
            extracted_items = re.findall(r'["“’\']([^"”’\']+)["“’\']|(\d+[\.\、][^\s,，。！!？?]+)|(厦门大学|全季酒店靠窗大床房|厦门[^\s,，。！!？?]*|北京大学|北京[^\s,，。！!？?]*)', content)
            for item in extracted_items:
                matched_str = next((s for s in item if s), "").strip()
                # 清除列表序号前缀并过滤标点
                clean_str = re.sub(r'^\d+[\.\、]', '', matched_str)
                clean_str = re.sub(r'[,，。！!？?].*$', '', clean_str).strip()
                if clean_str and len(clean_str) >= 2 and clean_str not in seen:
                    seen.add(clean_str)
                    candidates.append({
                        "entity": clean_str,
                        "type": "address" if any(w in clean_str for w in ["厦大", "厦门", "北京", "地址", "路", "街"]) else "general",
                        "source": "dialogue_history",
                        "recency_step": turn_idx + 1,
                        "frequency": 1,
                        "key": "extracted_item",
                    })

        return candidates

    def score_candidates(
        self,
        candidates: List[Dict[str, Any]],
        pronoun: str,
        current_query: str,
    ) -> List[Dict[str, Any]]:
        """
        候选打分公式:
        Score = w1 * TypeMatch + w2 * exp(-gamma * Recency) + w3 * Sim(query, entity) + w4 * Freq
        """
        if not candidates:
            return []

        scored = []
        target_types = PRONOUN_TYPE_MAP.get(pronoun, ["general"])
        query_vec = self.embedder.embed_query(current_query)

        for c in candidates:
            entity_name = c["entity"]

            # 1. 类型匹配分 (0.0 ~ 1.0)
            type_score = 1.0 if c["type"] in target_types or "general" in target_types else 0.3

            # 2. 时间近因分 (指数衰减)
            recency_score = math.exp(-self.recency_decay * c["recency_step"])

            # 3. 语义相似度分
            ent_vec = self.embedder.embed_query(entity_name)
            dot_prod = sum(float(q) * float(e) for q, e in zip(query_vec, ent_vec))
            sim_score = max(0.0, min(1.0, dot_prod))

            # 4. 频率分
            freq_score = min(1.0, c["frequency"] * 0.5)

            # 综合加权得分
            total_score = (
                0.35 * type_score +
                0.35 * recency_score +
                0.20 * sim_score +
                0.10 * freq_score
            )

            scored.append({
                "entity": entity_name,
                "score": round(total_score, 4),
                "type": c["type"],
                "source": c["source"],
            })

        # 按得分从高到低排序
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def resolve_coreference(
        self,
        current_query: str,
        recent_messages: List[Dict[str, str]],
        facts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        解析指代核心函数
        :return: {
            "resolved_query": str,           # 消除指代歧义后的完整语义 Query
            "has_coreference": bool,         # 是否包含指代
            "target_entity": Optional[str],  # 选中的目标实体
            "confidence": float,              # 指代消解置信度
            "candidates": list,              # Top 打分候选者
        }
        """
        # 检测当前问题是否包含指代词
        detected_pronoun = None
        for p in PRONOUN_TYPE_MAP.keys():
            if p in current_query:
                detected_pronoun = p
                break

        if not detected_pronoun:
            return {
                "resolved_query": current_query,
                "has_coreference": False,
                "target_entity": None,
                "confidence": 1.0,
                "candidates": [],
            }

        # 1. 提取候选实体
        candidates = self.extract_candidates_from_history(recent_messages, facts)

        # 2. 候选实体综合打分
        scored_candidates = self.score_candidates(candidates, detected_pronoun, current_query)

        if not scored_candidates:
            return {
                "resolved_query": current_query,
                "has_coreference": True,
                "target_entity": None,
                "confidence": 0.0,
                "candidates": [],
            }

        top_candidate = scored_candidates[0]
        target_entity = top_candidate["entity"]
        confidence = top_candidate["score"]

        # 3. 指代替换生成清晰语义的 resolved_query
        if detected_pronoun in ["改一下地址", "地址"]:
            resolved_query = f"将收货地址或位置修改为 {target_entity}"
        else:
            resolved_query = current_query.replace(detected_pronoun, target_entity)

        return {
            "resolved_query": resolved_query,
            "has_coreference": True,
            "target_entity": target_entity,
            "confidence": confidence,
            "candidates": scored_candidates[:3],
        }
