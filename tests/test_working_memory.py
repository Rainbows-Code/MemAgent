import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import MagicMock
import redis
from app.memory.working import WorkingMemory


def test_working_memory_fallback_mode():
    """测试当 Redis 不可用时的内存 Fallback 模式"""
    mock_bad_redis = MagicMock(spec=redis.Redis)
    mock_bad_redis.ping.side_effect = redis.ConnectionError("Redis down")

    mem = WorkingMemory(user_id="fallback_user", window_size=2, redis_client=mock_bad_redis)
    # 不管是 redis 报异常还是强行断开，验证 fallback 逻辑
    mem.is_redis_available = False

    mem.add_message("user", "你好")
    mem.add_message("assistant", "你好！我是 AI 助手")

    history = mem.get_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "你好"
    assert history[1]["content"] == "你好！我是 AI 助手"


def test_working_memory_sliding_window_fallback():
    """测试 Fallback 模式下的滑动窗口裁剪 (window_size=2，最多留 4 条消息)"""
    mock_bad_redis = MagicMock(spec=redis.Redis)
    mem = WorkingMemory(user_id="fallback_user", window_size=2, redis_client=mock_bad_redis)
    mem.is_redis_available = False

    # 追加 3 轮 (6 条) 消息
    for i in range(1, 4):
        mem.add_message("user", f"问题 {i}")
        mem.add_message("assistant", f"回答 {i}")

    history = mem.get_history()
    assert len(history) == 4  # 保留最近 4 条
    assert history[0]["content"] == "问题 2"
    assert history[-1]["content"] == "回答 3"


def test_working_memory_redis_mock():
    """测试 Redis 正常的逻辑 (Mock Redis Client)"""
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    mock_redis.lrange.return_value = [
        '{"role": "user", "content": "hello", "timestamp": 123456.78}',
        '{"role": "assistant", "content": "hi", "timestamp": 123457.00}'
    ]

    mem = WorkingMemory(user_id="test_user", window_size=5, redis_client=mock_redis)

    # 测试读取
    history = mem.get_history()
    assert len(history) == 2
    assert history[0]["content"] == "hello"
    mock_redis.lrange.assert_called_with("chat:test_user:working", 0, -1)

    # 测试写入
    mem.add_message("user", "新消息")
    assert mock_redis.rpush.called
    assert mock_redis.ltrim.called
    assert mock_redis.expire.called

    # 测试清空
    mem.clear()
    mock_redis.delete.assert_called_with("chat:test_user:working")


def test_get_context_messages():
    """测试 get_context_messages 构建 Prompt 格式"""
    mock_bad_redis = MagicMock(spec=redis.Redis)
    mem = WorkingMemory(user_id="prompt_user", window_size=2, redis_client=mock_bad_redis)
    mem.is_redis_available = False
    mem._fallback_memory = [
        {"role": "user", "content": "想吃火锅", "timestamp": 1.0},
        {"role": "assistant", "content": "好的，推荐海底捞", "timestamp": 2.0}
    ]

    messages = mem.get_context_messages(system_prompt="你是一个助理")
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "你是一个助理"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


if __name__ == "__main__":
    pytest.main(["-vs", __file__])
