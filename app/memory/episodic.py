import logging
import asyncio
from typing import Dict, List, Any, Optional
import asyncpg
import numpy as np
from app.config import settings
from app.memory.embeddings import EmbeddingModel, embedding_model

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """
    pgvector 情景记忆 (Episodic Memory) 管理类
    用于存储会话摘要 (Summary) 及其 Embedding 向量，支持 Cosine 相似度检索
    """

    def __init__(
        self,
        embedder: Optional[EmbeddingModel] = None,
        dsn: Optional[str] = None,
    ):
        self.embedder = embedder or embedding_model
        self.dsn = dsn or settings.postgres_dsn
        self.is_pg_available = False

        # 本地 Fallback 内存列表
        self._fallback_records: List[Dict[str, Any]] = []

    async def init_db(self) -> bool:
        """
        初始化 PostgreSQL 连通性并创建 pgvector 扩展及情景记忆表
        """
        try:
            conn = await asyncpg.connect(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                database=settings.POSTGRES_DB,
                timeout=3.0,
            )
            # 开启 vector 扩展
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # 创建 episodic_memory 表 (指定向量维度，例如 768)
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS episodic_memory (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(64) NOT NULL,
                summary TEXT NOT NULL,
                embedding vector({settings.EMBEDDING_DIMENSION}),
                importance FLOAT DEFAULT 0.5,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_episodic_user_id ON episodic_memory(user_id);
            """
            await conn.execute(create_table_sql)
            await conn.close()
            self.is_pg_available = True
            logger.info("PostgreSQL pgvector 情景记忆表初始化成功！")
            return True
        except Exception as e:
            logger.warning(
                f"无法连接到 PostgreSQL ({e})，EpisodicMemory 将降级为内存向量检索模式。"
            )
            self.is_pg_available = False
            return False

    async def add_memory(
        self,
        user_id: str,
        summary: str,
        importance: float = 0.5,
    ) -> Dict[str, Any]:
        """
        添加一条会话摘要并存入 Embedding 向量
        """
        vector = self.embedder.embed_query(summary)

        if self.is_pg_available:
            try:
                conn = await asyncpg.connect(
                    host=settings.POSTGRES_HOST,
                    port=settings.POSTGRES_PORT,
                    user=settings.POSTGRES_USER,
                    password=settings.POSTGRES_PASSWORD,
                    database=settings.POSTGRES_DB,
                    timeout=3.0,
                )
                vec_str = f"[{','.join(str(v) for v in vector)}]"
                insert_sql = """
                INSERT INTO episodic_memory (user_id, summary, embedding, importance)
                VALUES ($1, $2, $3::vector, $4)
                RETURNING id, created_at;
                """
                row = await conn.fetchrow(insert_sql, user_id, summary, vec_str, importance)
                await conn.close()

                return {
                    "id": row["id"],
                    "user_id": user_id,
                    "summary": summary,
                    "importance": importance,
                    "created_at": str(row["created_at"]),
                }
            except Exception as e:
                logger.error(f"向 PostgreSQL 写入情景记忆失败 ({e})，回退到内存模式")
                self.is_pg_available = False

        # Fallback 内存向量处理
        record = {
            "id": len(self._fallback_records) + 1,
            "user_id": user_id,
            "summary": summary,
            "vector": vector,
            "importance": importance,
        }
        self._fallback_records.append(record)
        return record

    async def search_memory(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
        min_similarity: float = -1.0,
    ) -> List[Dict[str, Any]]:
        """
        根据用户查询问题，检索基于 Cosine 相似度的 Top-K 相关历史摘要
        """
        query_vector = self.embedder.embed_query(query)

        if self.is_pg_available:
            try:
                conn = await asyncpg.connect(
                    host=settings.POSTGRES_HOST,
                    port=settings.POSTGRES_PORT,
                    user=settings.POSTGRES_USER,
                    password=settings.POSTGRES_PASSWORD,
                    database=settings.POSTGRES_DB,
                    timeout=3.0,
                )
                vec_str = f"[{','.join(str(v) for v in query_vector)}]"
                # PostgreSQL pgvector 余弦相似度 SQL Query: 1 - (embedding <=> query_vec)
                search_sql = """
                SELECT id, summary, importance, created_at,
                       (1 - (embedding <=> $1::vector)) AS similarity
                FROM episodic_memory
                WHERE user_id = $2
                ORDER BY embedding <=> $1::vector ASC
                LIMIT $3;
                """
                rows = await conn.fetch(search_sql, vec_str, user_id, top_k)
                await conn.close()

                results = []
                for r in rows:
                    sim = float(r["similarity"])
                    if sim >= min_similarity:
                        results.append({
                            "id": r["id"],
                            "summary": r["summary"],
                            "importance": r["importance"],
                            "similarity": round(sim, 4),
                            "created_at": str(r["created_at"]),
                        })
                return results
            except Exception as e:
                logger.error(f"PostgreSQL pgvector 相似度检索失败 ({e})")
                self.is_pg_available = False

        # Fallback 内存 Cosine 相似度检索
        user_records = [r for r in self._fallback_records if r["user_id"] == user_id]
        if not user_records:
            return []

        q_vec = np.array(query_vector)
        scored_records = []
        for r in user_records:
            doc_vec = np.array(r["vector"])
            norm = (np.linalg.norm(q_vec) * np.linalg.norm(doc_vec))
            cos_sim = float(np.dot(q_vec, doc_vec) / norm) if norm > 0 else 0.0

            if cos_sim >= min_similarity:
                scored_records.append({
                    "id": r["id"],
                    "summary": r["summary"],
                    "importance": r["importance"],
                    "similarity": round(cos_sim, 4),
                })

        # 按相似度降序排序
        scored_records.sort(key=lambda x: x["similarity"], reverse=True)
        return scored_records[:top_k]

    async def clear(self, user_id: str) -> None:
        """清空指定用户的 Episodic 记忆"""
        if self.is_pg_available:
            try:
                conn = await asyncpg.connect(
                    host=settings.POSTGRES_HOST,
                    port=settings.POSTGRES_PORT,
                    user=settings.POSTGRES_USER,
                    password=settings.POSTGRES_PASSWORD,
                    database=settings.POSTGRES_DB,
                )
                await conn.execute("DELETE FROM episodic_memory WHERE user_id = $1;", user_id)
                await conn.close()
            except Exception as e:
                logger.error(f"清空 PostgreSQL 情景记忆失败: {e}")

        self._fallback_records = [r for r in self._fallback_records if r["user_id"] != user_id]
