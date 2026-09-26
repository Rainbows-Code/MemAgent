import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from app.memory.coreference import CoreferenceResolver


def test_coreference_resolution_pronoun_ta():
    """测试代词『它』指代消解"""
    resolver = CoreferenceResolver()

    messages = [
        {"role": "user", "content": "帮我看看全季酒店靠窗大床房"},
        {"role": "assistant", "content": "全季酒店靠窗大床房目前有房，价格为 380 元。"}
    ]
    facts = []

    res = resolver.resolve_coreference(
        current_query="它包含早餐吗？",
        recent_messages=messages,
        facts=facts,
    )

    assert res["has_coreference"] is True
    assert res["target_entity"] == "全季酒店靠窗大床房"
    assert "全季酒店靠窗大床房" in res["resolved_query"]
    assert res["confidence"] > 0.4


def test_coreference_resolution_nane():
    """测试代词『那个』指代消解"""
    resolver = CoreferenceResolver()

    messages = [
        {"role": "user", "content": "我想去厦门大学逛逛"},
        {"role": "assistant", "content": "厦门大学非常美，需要提前预约入校。"}
    ]
    facts = []

    res = resolver.resolve_coreference(
        current_query="那个地方需要门票吗？",
        recent_messages=messages,
        facts=facts,
    )

    assert res["has_coreference"] is True
    assert res["target_entity"] == "厦门大学"
    assert "厦门大学" in res["resolved_query"]


def test_coreference_resolution_change_address():
    """测试『改一下地址』及语义记忆结合消解"""
    resolver = CoreferenceResolver()

    messages = []
    facts = [
        {"key": "address", "value": "北京大学", "confidence": 1.0}
    ]

    res = resolver.resolve_coreference(
        current_query="改一下地址",
        recent_messages=messages,
        facts=facts,
    )

    assert res["has_coreference"] is True
    assert res["target_entity"] == "北京大学"
    assert "北京大学" in res["resolved_query"]


def test_no_coreference():
    """测试常规无指代问题"""
    resolver = CoreferenceResolver()

    res = resolver.resolve_coreference(
        current_query="今天北京的天气怎么样？",
        recent_messages=[],
        facts=[],
    )

    assert res["has_coreference"] is False
    assert res["resolved_query"] == "今天北京的天气怎么样？"
    assert res["target_entity"] is None


if __name__ == "__main__":
    pytest.main(["-vs", __file__])
