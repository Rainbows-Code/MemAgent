import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import time
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

    # 1. 指代消解
    coref_res = coref_resolver.resolve_coreference(
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

    # 3. 构建 Prompt 并调用 LLM
    base_prompt = req.system_prompt or "你是一个贴心的智能助手。请精准回答用户的问题。"
    prompt_messages = retriever.build_injected_prompt(base_prompt, ctx)

    # 将本轮消解后的 query 追加为最新的 user message 发送给 LLM
    prompt_messages.append({"role": "user", "content": coref_res["resolved_query"]})

    llm_res = await llm_client.achat_completion(prompt_messages)
    assistant_reply = llm_res["content"]
    usage = llm_res["usage"]

    # 4. 更新工作记忆
    working_mem.add_message("user", req.message)
    working_mem.add_message("assistant", assistant_reply)

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

    episodic_data = await episodic_mem.search_memory(user_id=user_id, query="历史摘要", top_k=5)

    return {
        "user_id": user_id,
        "working_memory_count": len(working_data),
        "working_memory": working_data,
        "semantic_facts": semantic_data,
        "episodic_summaries": episodic_data,
    }


@app.get("/audit", summary="查看系统审计日志")
async def get_audit_logs(limit: int = Query(20, ge=1, le=100)):
    """获取最近的指代解析与记忆调度审计日志"""
    return {
        "total_audit_records": len(audit_logs),
        "logs": audit_logs[-limit:],
    }
