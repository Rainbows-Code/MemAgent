import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_api_health():
    """测试 /health 接口"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_api_chat_and_memory():
    """测试 /chat 端到端对话与 /memory 查看接口"""
    user_id = "test_api_user"

    # 第 1 轮对话
    payload1 = {
        "user_id": user_id,
        "message": "帮我看看全季酒店靠窗大床房",
    }
    res1 = client.post("/chat", json=payload1)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["user_id"] == user_id
    assert "response" in data1

    # 第 2 轮带指代对话 ("它包含早餐吗？")
    payload2 = {
        "user_id": user_id,
        "message": "它包含早餐吗？",
    }
    res2 = client.post("/chat", json=payload2)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["has_coreference"] is True
    assert "全季酒店" in data2["resolved_query"]

    # 查看 /memory/{user_id} 接口
    mem_res = client.get(f"/memory/{user_id}")
    assert mem_res.status_code == 200
    mem_data = mem_res.json()
    assert mem_data["user_id"] == user_id
    assert mem_data["working_memory_count"] >= 2


def test_api_audit():
    """测试 /audit 审计日志接口"""
    audit_res = client.get("/audit?limit=5")
    assert audit_res.status_code == 200
    audit_data = audit_res.json()
    assert "total_audit_records" in audit_data


if __name__ == "__main__":
    pytest.main(["-vs", __file__])
