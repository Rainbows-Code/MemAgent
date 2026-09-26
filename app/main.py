import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import time
import re
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.config import settings
from app.llm import LLMClient, llm_client
from app.memory.working import WorkingMemory
from app.memory.episodic import EpisodicMemory
from app.memory.semantic import SemanticMemory
from app.memory.retrieval import MemoryRetriever
from app.memory.coreference import CoreferenceResolver
from app.memory.forgetting import MemoryForgettingEngine

logger = logging.getLogger("memagent.api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="MemAgent 多轮指代感知记忆系统 API",
    description="提供三层记忆调度 (Redis + pgvector + 实体表)、指代消解与自动化剪枝遗忘的生产级 Agent 接口",
    version="1.0.0",
)

# 全局组件
episodic_mem = EpisodicMemory()
semantic_mem = SemanticMemory()
coref_resolver = CoreferenceResolver()
forgetting_engine = MemoryForgettingEngine(episodic_memory=episodic_mem, semantic_memory=semantic_mem)

# 简易审计日志列表
audit_logs: List[Dict[str, Any]] = []


@app.on_event("startup")
async def startup_event():
    """服务启动时自动初始化 PostgreSQL 表结构"""
    logger.info("正在启动 MemAgent API 服务，初始化数据库环境...")
    await episodic_mem.init_db()
    await semantic_mem.init_db()


# ---------------- Request & Response Models ----------------

class ChatRequest(BaseModel):
    user_id: str = Field(..., example="user_123", description="用户唯一标识")
    message: str = Field(..., example="那个包含早餐吗？", description="用户当前轮对话消息")
    system_prompt: Optional[str] = Field(None, description="自定义系统提示词")


class ChatResponse(BaseModel):
    user_id: str
    original_message: str
    resolved_query: str
    has_coreference: bool
    target_entity: Optional[str]
    response: str
    usage: Dict[str, int]
    injected_memory_meta: Dict[str, Any]


# ---------------- API Endpoints ----------------

@app.get("/health", summary="健康检查接口")
async def health_check():
    """检查 API 及数据库连通状态"""
    return {
        "status": "ok",
        "redis_available": settings.REDIS_HOST is not None,
        "postgres_available": episodic_mem.is_pg_available,
    }


@app.post("/chat", response_model=ChatResponse, summary="多轮指代感知 Agent 主对话接口")
async def chat_endpoint(req: ChatRequest):
    """
    核心对话流 Pipeline:
    1. 提取当前三层记忆
    2. 执行 Coreference Resolution (实体打分与指代消解)
    3. 按需动态检索与注入上下文 Messages
    4. LLM 推理生成回答
    5. 自动更新 Working Memory、写入 Semantic Fact，并触发剪枝与审计
    """
    user_id = req.user_id
    working_mem = WorkingMemory(user_id=user_id)
    recent_msgs = working_mem.get_history()
    facts = await semantic_mem.get_all_facts(user_id)

    # 1. 异步指代消解与意图改写 (支持通用 LLM Query Rewrite 与启发式双核)
    coref_res = await coref_resolver.aresolve_coreference(
        current_query=req.message,
        recent_messages=recent_msgs,
        facts=facts,
    )

    # 2. 按需记忆检索
    retriever = MemoryRetriever(
        working_memory=working_mem,
        episodic_memory=episodic_mem,
        semantic_memory=semantic_mem,
    )
    ctx = await retriever.retrieve_context(user_id=user_id, current_query=coref_res["resolved_query"])

    # 3. 构建 Prompt 并调用 LLM (添加防旧话题幻觉强约束)
    base_prompt = (
        req.system_prompt
        or "你是一个严谨、贴心且高效的智能助手。请基于用户当前轮次提出的具体意图进行专注、清晰的解答。\n"
           "【重要准则】: 请严格聚焦于用户最新的问题核心。切勿将早期对话中无关的旧主题（如旧城市或不同行程）生硬混合进新问题的解答中。"
    )
    prompt_messages = retriever.build_injected_prompt(base_prompt, ctx)

    # 将本轮消解后的 query 追加为最新的 user message 发送给 LLM
    prompt_messages.append({"role": "user", "content": coref_res["resolved_query"]})

    llm_res = await llm_client.achat_completion(prompt_messages)
    assistant_reply = llm_res["content"]
    usage = llm_res["usage"]

    # 4. 更新工作记忆 (Redis)
    working_mem.add_message("user", req.message)
    working_mem.add_message("assistant", assistant_reply)

    # 5. 自动语义记忆 (用户长效偏好与事实) 抽取与沉淀 (PostgreSQL user_fact)
    query_text = req.message
    # 地址与位置信息抽取
    addr_match = re.search(r'(?:地址是|地址为|修改为|改成|设置为|改为)\s*([^\s,，。！？?]+)', query_text)
    if addr_match:
        await semantic_mem.set_fact(user_id, "address", addr_match.group(1).strip(), source="user_explicit")
    # 旅行目的地偏好抽取
    dest_match = re.search(r'(?:想去|去|计划去|打算去)\s*([^\s,，。！？?]{2,15}?)(?:旅游|出差|玩|出游|度假)?(?:[,，。！？?\s]|$)', query_text)
    if dest_match:
        clean_dest = re.sub(r'^(?:我想|我|请问|帮我)', '', dest_match.group(1)).strip()
        if clean_dest and len(clean_dest) >= 2:
            await semantic_mem.set_fact(user_id, "destination", clean_dest, source="user_explicit")
    # 酒店与住宿偏好抽取
    hotel_match = re.search(r'(?:帮我看看|看看|预定|预订|订)\s*([^\s,，。！？?]{2,20})', query_text)
    if hotel_match and any(h in hotel_match.group(1) for h in ["酒店", "客栈", "房", "民宿"]):
        await semantic_mem.set_fact(user_id, "preferred_hotel", hotel_match.group(1).strip(), source="user_explicit")

    # 6. 自动情景记忆 (Episodic Memory) 沉淀归档 (pgvector)
    try:
        short_summary = f"用户询问: {coref_res['resolved_query']}; 助手建议核心: {assistant_reply[:60]}..."
        await episodic_mem.add_memory(user_id=user_id, summary=short_summary, importance=0.7)
    except Exception as e:
        logger.warning(f"情景记忆写入失败: {e}")

    # 审计日志追加
    audit_entry = {
        "user_id": user_id,
        "original_message": req.message,
        "resolved_query": coref_res["resolved_query"],
        "target_entity": coref_res["target_entity"],
        "usage": usage,
        "timestamp": time.time(),
    }
    audit_logs.append(audit_entry)

    return ChatResponse(
        user_id=user_id,
        original_message=req.message,
        resolved_query=coref_res["resolved_query"],
        has_coreference=coref_res["has_coreference"],
        target_entity=coref_res["target_entity"],
        response=assistant_reply,
        usage=usage,
        injected_memory_meta=ctx["injected_memory_meta"],
    )


@app.get("/memory/{user_id}", summary="查看用户三层记忆状态")
async def get_user_memory(user_id: str):
    """查看指定用户在 Working, Episodic, Semantic 三层记忆中的数据状态"""
    working_mem = WorkingMemory(user_id=user_id)
    working_data = working_mem.get_history()

    semantic_data = await semantic_mem.get_all_facts(user_id)

    # 优先获取该用户全部情景记忆列表
    episodic_data = await episodic_mem.get_all_memories(user_id=user_id, limit=10)
    if not episodic_data:
        episodic_data = await episodic_mem.search_memory(user_id=user_id, query="历史摘要", top_k=5)

    return {
        "user_id": user_id,
        "working_memory_count": len(working_data),
        "working_memory": working_data,
        "semantic_facts": semantic_data,
        "episodic_summaries": episodic_data,
    }


@app.delete("/memory/{user_id}", summary="彻底清空指定用户的三层记忆状态")
async def clear_user_memory(user_id: str):
    """
    重置并彻底清空指定用户的 Working Memory (Redis)、Episodic Memory (pgvector) 与 Semantic Facts，
    避免受污染的历史对话继续干扰后续的全新会话。
    """
    working_mem = WorkingMemory(user_id=user_id)
    working_mem.clear()
    await episodic_mem.clear(user_id)
    await semantic_mem.clear(user_id)
    logger.info(f"已彻底清空用户 {user_id} 的三层记忆。")
    return {
        "status": "ok",
        "message": f"用户 {user_id} 的三层记忆 (Redis/pgvector/PostgreSQL) 已彻底清空并重置。",
    }


@app.get("/audit", summary="查看系统审计日志")
async def get_audit_logs(limit: int = Query(20, ge=1, le=100)):
    """获取最近的指代解析与记忆调度审计日志"""
    return {
        "total_audit_records": len(audit_logs),
        "logs": audit_logs[-limit:],
    }
