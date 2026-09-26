import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import MagicMock, AsyncMock
from app.memory.working import WorkingMemory
from app.memory.episodic import EpisodicMemory
from app.memory.semantic import SemanticMemory
from app.memory.retrieval import MemoryRetriever


@pytest.mark.asyncio
async def test_memory_retriever_on_demand_injection():
    """测试三层记忆按需检索与动态 Prompt 构建"""
    # Mock 三层记忆
    mock_working = MagicMock(spec=WorkingMemory)
    mock_working.get_history.return_value = [
        {"role": "user", "content": "帮我看看机票"},
        {"role": "assistant", "content": "好的，去哪里？"}
    ]

    mock_episodic = MagicMock(spec=EpisodicMemory)
    mock_episodic.search_memory = AsyncMock(return_value=[
        {"id": 101, "summary": "用户经常在厦门大学和北京大学出差", "similarity": 0.85}
    ])

    mock_semantic = MagicMock(spec=SemanticMemory)
    mock_semantic.get_all_facts = AsyncMock(return_value=[
        {"key": "seat_preference", "value": "靠窗", "confidence": 1.0},
        {"key": "address", "value": "北京大学", "confidence": 1.0}
    ])

    retriever = MemoryRetriever(
        working_memory=mock_working,
        episodic_memory=mock_episodic,
        semantic_memory=mock_semantic,
    )

    user_id = "test_retrieval_user"
    query = "我要买一张去北京的机票，选个好位置"

    # 1. 按需检索
    context = await retriever.retrieve_context(user_id=user_id, current_query=query)
    assert len(context["recent_messages"]) == 2
    assert len(context["retrieved_summaries"]) == 1
    assert len(context["user_facts"]) == 2

    meta = context["injected_memory_meta"]
    assert meta["working_message_count"] == 2
    assert meta["episodic_summary_ids"] == [101]
    assert "seat_preference" in meta["semantic_fact_keys"]

    # 2. 动态 Prompt 构建
    messages = retriever.build_injected_prompt(
        base_system_prompt="你是一个出行智能助手。",
        retrieved_context=context,
    )

    assert len(messages) == 3  # system + 2 recent messages
    system_content = messages[0]["content"]
    assert "seat_preference: 靠窗" in system_content
    assert "用户经常在厦门大学和北京大学出差" in system_content


if __name__ == "__main__":
    pytest.main(["-vs", __file__])

