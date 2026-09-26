import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import MagicMock, AsyncMock
from app.chat import BaselineAgent
from app.llm import LLMClient



@pytest.fixture
def mock_llm():
    """创建一个 Mock LLM Client，避免测试消耗真实 Token 和耗时"""
    client = MagicMock(spec=LLMClient)
    client.chat_completion.return_value = {
        "content": "这是 Mock 的 LLM 回复",
        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
    }
    client.achat_completion = AsyncMock(return_value={
        "content": "这是 Mock 的异步 LLM 回复",
        "usage": {"prompt_tokens": 55, "completion_tokens": 12, "total_tokens": 67},
    })
    return client


def test_baseline_agent_basic_flow(mock_llm):
    """测试 BaselineAgent 的基本对话与历史记录功能"""
    agent = BaselineAgent(user_id="test_user", window_size=2, llm=mock_llm)

    # 第 1 轮对话
    res1 = agent.chat("你好")
    assert res1["response"] == "这是 Mock 的 LLM 回复"
    assert res1["history_count"] == 2
    assert res1["usage"]["total_tokens"] == 60
    assert len(res1["context_messages"]) == 2  # system + user 1

    # 第 2 轮对话
    res2 = agent.chat("我是小明")
    assert res2["history_count"] == 4
    assert len(res2["context_messages"]) == 4  # system + u1 + a1 + u2


def test_baseline_agent_sliding_window(mock_llm):
    """测试滑动窗口裁剪功能 (window_size=2，截断超出的旧对话)"""
    agent = BaselineAgent(user_id="test_user", window_size=2, llm=mock_llm)

    # 模拟 3 轮对话 (共 6 条消息: u1, a1, u2, a2, u3, a3)
    agent.chat("消息 1")
    agent.chat("消息 2")
    res3 = agent.chat("消息 3")

    # history 包含了 6 条
    assert agent.history_count if hasattr(agent, 'history_count') else len(agent.history) == 6

    # 但 context_messages 只能出现 system + 最近 4 条 (u2, a2, u3)
    # 因为 window_size=2，消息限制为 max 4 条历史
    ctx = res3["context_messages"]
    assert ctx[0]["role"] == "system"
    assert len(ctx) == 5  # 1 system + 4 recent messages (a1, u2, a2, u3)
    # 最早的 u1 ("消息 1") 已被截断滑出
    user_contents = [msg["content"] for msg in ctx if msg["role"] == "user"]
    assert "消息 1" not in user_contents
    assert user_contents == ["消息 2", "消息 3"]


@pytest.mark.asyncio
async def test_baseline_agent_async(mock_llm):
    """测试 BaselineAgent 的异步对话"""
    agent = BaselineAgent(user_id="test_user", window_size=2, llm=mock_llm)
    res = await agent.achat("异步测试消息")
    assert res["response"] == "这是 Mock 的异步 LLM 回复"
    assert res["usage"]["total_tokens"] == 67


def test_baseline_agent_clear_history(mock_llm):
    """测试清空历史功能"""
    agent = BaselineAgent(user_id="test_user", llm=mock_llm)
    agent.chat("测试消息")
    assert len(agent.history) == 2
    agent.clear_history()
    assert len(agent.history) == 0


if __name__ == "__main__":
    pytest.main(["-vs", __file__])

