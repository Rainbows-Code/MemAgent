import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pytest_asyncio
from app.memory.embeddings import EmbeddingModel
from app.memory.episodic import EpisodicMemory


def test_embedding_model_basic():
    """测试 Embedding 向量输出格式与维度"""
    embedder = EmbeddingModel(dimension=768)
    vec = embedder.embed_query("用户偏好靠窗座位")
    assert len(vec) == 768
    assert isinstance(vec[0], float)

    batch_vecs = embedder.embed_documents(["句子一", "句子二"])
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 768


@pytest.mark.asyncio
async def test_episodic_memory_real_pgvector():
    """测试真实 PostgreSQL pgvector 连通、建表、写入与 Cosine 向量检索"""
    episodic = EpisodicMemory()
    is_ready = await episodic.init_db()

    user_id = "test_episodic_user"
    await episodic.clear(user_id)

    # 1. 插入两条摘要
    rec1 = await episodic.add_memory(
        user_id=user_id,
        summary="用户上次在厦门大学附近预订了全季酒店靠窗大床房",
        importance=0.8,
    )
    rec2 = await episodic.add_memory(
        user_id=user_id,
        summary="用户表示自己对花粉过敏，喜欢素食餐饮",
        importance=0.6,
    )
    assert rec1["id"] is not None
    assert rec2["id"] is not None

    # 2. 向量检索：搜索与“酒店/住宿”最相关的摘要
    results = await episodic.search_memory(
        user_id=user_id,
        query="用户之前住过什么酒店？有什么房间偏好？",
        top_k=2,
    )

    assert len(results) == 2
    for item in results:
        assert "summary" in item
        assert "similarity" in item
        assert -1.0 <= item["similarity"] <= 1.0

    # 清理测试数据
    await episodic.clear(user_id)


if __name__ == "__main__":
    pytest.main(["-vs", __file__])
