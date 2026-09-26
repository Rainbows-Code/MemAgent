import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import time
from unittest.mock import MagicMock, AsyncMock
from app.memory.forgetting import MemoryForgettingEngine
from app.memory.episodic import EpisodicMemory


def test_importance_scoring_weights():
    """测试不同数据来源的初始重要度打分"""
    engine = MemoryForgettingEngine()
    assert engine.calculate_importance_score("user_explicit") == 1.0
    assert engine.calculate_importance_score("tool_result") == 0.7
    assert engine.calculate_importance_score("chitchat") == 0.3


def test_time_decay_math():
    """测试时间衰减公式计算 (0天、10天衰减计算)"""
    engine = MemoryForgettingEngine(decay_lambda=0.01)
    now = time.time()

    # 当天刚创建 (无衰减)
    score_today = engine.calculate_time_decay_score(1.0, now)
    assert score_today == 1.0

    # 10天前创建 (exp(-0.01 * 10) = exp(-0.1) ≈ 0.9048)
    ten_days_ago = now - (10 * 86400)
    score_10_days = engine.calculate_time_decay_score(1.0, ten_days_ago)
    assert 0.89 <= score_10_days <= 0.91


@pytest.mark.asyncio
async def test_summary_compression():
    """测试对话摘要生成与压缩存储"""
    mock_llm = MagicMock()
    mock_llm.achat_completion = AsyncMock(return_value={"content": "用户预订了靠窗房间并喜欢素食。"})

    mock_episodic = MagicMock(spec=EpisodicMemory)
    mock_episodic.add_memory = AsyncMock(return_value={"id": 1})

    engine = MemoryForgettingEngine(llm=mock_llm, episodic_memory=mock_episodic)

    messages = [
        {"role": "user", "content": "我要一个靠窗订房"},
        {"role": "assistant", "content": "好的，另外您有餐饮特殊需求吗？"},
        {"role": "user", "content": "我只吃素食"}
    ]

    summary = await engine.compress_conversation_summary("test_user", messages)
    assert summary == "用户预订了靠窗房间并喜欢素食。"
    assert mock_episodic.add_memory.called


@pytest.mark.asyncio
async def test_prune_forgotten_memories():
    """测试根据重要度与衰减淘汰冗余记忆 (淘汰率计算)"""
    engine = MemoryForgettingEngine(forgetting_threshold=0.4)

    facts = [
        {"key": "address", "value": "厦门大学", "source": "user_explicit", "confidence": 1.0},
        {"key": "weather_chat", "value": "今天天气晴朗", "source": "chitchat", "confidence": 0.2},
        {"key": "temp_search", "value": "搜过全季", "source": "chitchat", "confidence": 0.1},
    ]

    res = await engine.prune_forgotten_memories("user123", facts)
    assert len(res["retained"]) == 1  # 只有 user_explicit 保持保留
    assert len(res["pruned"]) == 2    # 闲聊低分数被成功剪枝
    assert res["retained"][0]["key"] == "address"
    assert res["prune_rate"] > 0.6    # 冗余记忆淘汰率 > 60%


if __name__ == "__main__":
    pytest.main(["-vs", __file__])
