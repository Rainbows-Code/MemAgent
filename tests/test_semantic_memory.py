import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pytest_asyncio
from app.memory.semantic import SemanticMemory


@pytest.mark.asyncio
async def test_semantic_memory_crud_and_conflict_resolution():
    """测试结构化语义记忆的写入、修改冲突覆盖与审计归档"""
    semantic = SemanticMemory()
    await semantic.init_db()

    user_id = "test_semantic_user"
    await semantic.clear(user_id)

    # 1. 写入用户明示偏好
    fact1 = await semantic.set_fact(
        user_id=user_id,
        key="address",
        value="厦门大学",
        confidence=1.0,
        source="user_explicit",
    )
    assert fact1["value"] == "厦门大学"
    assert fact1["old_value"] is None

    fact2 = await semantic.set_fact(
        user_id=user_id,
        key="seat_preference",
        value="靠窗",
        confidence=0.9,
        source="user_explicit",
    )
    assert fact2["value"] == "靠窗"

    # 2. 查询全部偏好
    all_facts = await semantic.get_all_facts(user_id)
    assert len(all_facts) == 2
    keys = [f["key"] for f in all_facts]
    assert "address" in keys
    assert "seat_preference" in keys

    # 3. 冲突覆盖测试：更新 address 为 "北京大学"
    fact_updated = await semantic.set_fact(
        user_id=user_id,
        key="address",
        value="北京大学",
        confidence=1.0,
        source="user_explicit",
    )
    assert fact_updated["value"] == "北京大学"
    assert fact_updated["old_value"] == "厦门大学"

    # 4. 确认最新值为 "北京大学"
    current_addr = await semantic.get_fact(user_id, "address")
    assert current_addr["value"] == "北京大学"

    # 5. 校验归档历史表 user_fact_history
    history = await semantic.get_fact_history(user_id, key="address")
    assert len(history) >= 1
    latest_hist = history[0]
    assert latest_hist["old_value"] == "厦门大学"
    assert latest_hist["new_value"] == "北京大学"

    # 6. 清理数据
    await semantic.clear(user_id)
    empty_facts = await semantic.get_all_facts(user_id)
    assert len(empty_facts) == 0


if __name__ == "__main__":
    pytest.main(["-vs", __file__])
