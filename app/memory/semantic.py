import logging
import asyncio
from typing import Dict, List, Any, Optional
import asyncpg
from app.config import settings

logger = logging.getLogger(__name__)


class SemanticMemory:
    """
    结构化语义记忆 (Semantic Memory) 管理类
    用于存储与更新用户的长效偏好、事实与约束条件 (例如: address=厦门大学, seat_preference=靠窗)
    支持基于 SQL ON CONFLICT 的冲突覆盖与历史归档审计
    """

    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or settings.postgres_dsn
        self.is_pg_available = False
        self._fallback_facts: Dict[str, Dict[str, Any]] = {}  # {user_id: {key: fact_dict}}
        self._fallback_history: List[Dict[str, Any]] = []

    async def init_db(self) -> bool:
        """
        初始化 PostgreSQL 语义记忆表 user_fact 与归档表 user_fact_history
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

            # 主事实表
            create_fact_table = """
            CREATE TABLE IF NOT EXISTS user_fact (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(64) NOT NULL,
                key VARCHAR(128) NOT NULL,
                value TEXT NOT NULL,
                confidence FLOAT DEFAULT 1.0,
                source VARCHAR(64) DEFAULT 'user_explicit',
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT uq_user_key UNIQUE (user_id, key)
            );
            CREATE INDEX IF NOT EXISTS idx_user_fact_user ON user_fact(user_id);
            """

            # 冲突覆盖历史归档表
            create_history_table = """
            CREATE TABLE IF NOT EXISTS user_fact_history (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(64) NOT NULL,
                key VARCHAR(128) NOT NULL,
                old_value TEXT,
                new_value TEXT,
                changed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """

            await conn.execute(create_fact_table)
            await conn.execute(create_history_table)
            await conn.close()
            self.is_pg_available = True
            logger.info("PostgreSQL 语义记忆表初始化成功！")
            return True
        except Exception as e:
            logger.warning(f"无法连接到 PostgreSQL ({e})，SemanticMemory 将降级为内存字典模式。")
            self.is_pg_available = False
            return False

    async def set_fact(
        self,
        user_id: str,
        key: str,
        value: str,
        confidence: float = 1.0,
        source: str = "user_explicit",
    ) -> Dict[str, Any]:
        """
        写入或更新一条语义记忆规则/偏好。
        如果同一 key 存在旧值且值发生变化，旧值将被覆盖并归档至 user_fact_history。
        """
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

                # 1. 查旧值 (用于判定冲突覆盖与归档)
                select_sql = "SELECT value FROM user_fact WHERE user_id = $1 AND key = $2;"
                old_row = await conn.fetchrow(select_sql, user_id, key)
                old_value = old_row["value"] if old_row else None

                # 2. 如果存在旧值且发生冲突/更新，则写入归档历史表
                if old_value is not None and old_value != value:
                    archive_sql = """
                    INSERT INTO user_fact_history (user_id, key, old_value, new_value)
                    VALUES ($1, $2, $3, $4);
                    """
                    await conn.execute(archive_sql, user_id, key, old_value, value)
                    logger.info(f"语义记忆冲突覆盖归档: [{user_id}] {key}: '{old_value}' -> '{value}'")

                # 3. Upsert (插入或覆盖更新)
                upsert_sql = """
                INSERT INTO user_fact (user_id, key, value, confidence, source, updated_at)
                VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, key) DO UPDATE SET
                    value = EXCLUDED.value,
                    confidence = EXCLUDED.confidence,
                    source = EXCLUDED.source,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id, updated_at;
                """
                row = await conn.fetchrow(upsert_sql, user_id, key, value, confidence, source)
                await conn.close()

                return {
                    "id": row["id"],
                    "user_id": user_id,
                    "key": key,
                    "value": value,
                    "confidence": confidence,
                    "source": source,
                    "old_value": old_value,
                    "updated_at": str(row["updated_at"]),
                }
            except Exception as e:
                logger.error(f"写入 PostgreSQL 语义记忆失败 ({e})，降级为内存模式")
                self.is_pg_available = False

        # Fallback 内存处理
        if user_id not in self._fallback_facts:
            self._fallback_facts[user_id] = {}

        old_value = self._fallback_facts[user_id].get(key, {}).get("value")
        if old_value is not None and old_value != value:
            self._fallback_history.append({
                "user_id": user_id,
                "key": key,
                "old_value": old_value,
                "new_value": value,
            })

        fact_data = {
            "id": len(self._fallback_facts[user_id]) + 1,
            "user_id": user_id,
            "key": key,
            "value": value,
            "confidence": confidence,
            "source": source,
            "old_value": old_value,
        }
        self._fallback_facts[user_id][key] = fact_data
        return fact_data

    async def get_fact(self, user_id: str, key: str) -> Optional[Dict[str, Any]]:
        """
        获取指定 Key 的语义偏好事实
        """
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
                sql = "SELECT id, key, value, confidence, source, updated_at FROM user_fact WHERE user_id = $1 AND key = $2;"
                row = await conn.fetchrow(sql, user_id, key)
                await conn.close()

                if row:
                    return {
                        "id": row["id"],
                        "user_id": user_id,
                        "key": row["key"],
                        "value": row["value"],
                        "confidence": row["confidence"],
                        "source": row["source"],
                        "updated_at": str(row["updated_at"]),
                    }
                return None
            except Exception as e:
                logger.error(f"读取 PostgreSQL 语义记忆失败: {e}")
                self.is_pg_available = False

        return self._fallback_facts.get(user_id, {}).get(key)

    async def get_all_facts(self, user_id: str) -> List[Dict[str, Any]]:
        """
        获取用户的所有语义偏好与约束事实
        """
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
                sql = "SELECT id, key, value, confidence, source, updated_at FROM user_fact WHERE user_id = $1 ORDER BY updated_at DESC;"
                rows = await conn.fetch(sql, user_id)
                await conn.close()

                return [
                    {
                        "id": r["id"],
                        "user_id": user_id,
                        "key": r["key"],
                        "value": r["value"],
                        "confidence": r["confidence"],
                        "source": r["source"],
                        "updated_at": str(r["updated_at"]),
                    }
                    for r in rows
                ]
            except Exception as e:
                logger.error(f"读取用户全部语义记忆失败: {e}")
                self.is_pg_available = False

        return list(self._fallback_facts.get(user_id, {}).values())

    async def get_fact_history(self, user_id: str, key: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        查询语义记忆的变动与归档历史
        """
        if self.is_pg_available:
            try:
                conn = await asyncpg.connect(
                    host=settings.POSTGRES_HOST,
                    port=settings.POSTGRES_PORT,
                    user=settings.POSTGRES_USER,
                    password=settings.POSTGRES_PASSWORD,
                    database=settings.POSTGRES_DB,
                )
                if key:
                    sql = "SELECT * FROM user_fact_history WHERE user_id = $1 AND key = $2 ORDER BY changed_at DESC;"
                    rows = await conn.fetch(sql, user_id, key)
                else:
                    sql = "SELECT * FROM user_fact_history WHERE user_id = $1 ORDER BY changed_at DESC;"
                    rows = await conn.fetch(sql, user_id)
                await conn.close()

                return [dict(r) for r in rows]
            except Exception as e:
                logger.error(f"读取语义记忆历史失败: {e}")

        if key:
            return [r for r in self._fallback_history if r["user_id"] == user_id and r["key"] == key]
        return [r for r in self._fallback_history if r["user_id"] == user_id]

    async def clear(self, user_id: str) -> None:
        """清空用户的语义记忆"""
        if self.is_pg_available:
            try:
                conn = await asyncpg.connect(
                    host=settings.POSTGRES_HOST,
                    port=settings.POSTGRES_PORT,
                    user=settings.POSTGRES_USER,
                    password=settings.POSTGRES_PASSWORD,
                    database=settings.POSTGRES_DB,
                )
                await conn.execute("DELETE FROM user_fact WHERE user_id = $1;", user_id)
                await conn.execute("DELETE FROM user_fact_history WHERE user_id = $1;", user_id)
                await conn.close()
            except Exception as e:
                logger.error(f"清空 PostgreSQL 语义记忆失败: {e}")

        if user_id in self._fallback_facts:
            del self._fallback_facts[user_id]
        self._fallback_history = [r for r in self._fallback_history if r["user_id"] != user_id]
