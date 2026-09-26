import sys
import os
import socket
import asyncio
import asyncpg
import redis
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """快速检测端口 TCP 连通性"""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def check_redis():
    """检测真实 Redis 连通性"""
    print(f"[+] 正在检查 Redis ({settings.REDIS_HOST}:{settings.REDIS_PORT})...", flush=True)
    if not is_port_open(settings.REDIS_HOST, settings.REDIS_PORT, timeout=1.0):
        print(f"[FAIL] Redis 端口 {settings.REDIS_PORT} 未开放或无法连接！", flush=True)
        print("   -> 请确认 Docker / Redis 服务已启动 (命令: docker compose up -d)", flush=True)
        return False

    try:
        r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD or None,
            socket_timeout=1.0,
        )
        if r.ping():
            print(f"[SUCCESS] Redis 连接且 PING 响应正常！", flush=True)
            return True
    except Exception as e:
        print(f"[FAIL] Redis 异常: {e}", flush=True)
        return False


async def check_postgres():
    """检测真实 PostgreSQL 及 pgvector 扩展"""
    print(f"[+] 正在检查 PostgreSQL ({settings.POSTGRES_HOST}:{settings.POSTGRES_PORT})...", flush=True)
    if not is_port_open(settings.POSTGRES_HOST, settings.POSTGRES_PORT, timeout=1.0):
        print(f"[FAIL] PostgreSQL 端口 {settings.POSTGRES_PORT} 未开放或无法连接！", flush=True)
        print("   -> 请确认 Docker / PostgreSQL 服务已启动 (命令: docker compose up -d)", flush=True)
        return False

    try:
        conn = await asyncpg.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            database=settings.POSTGRES_DB,
            timeout=2.0,
        )
        print(f"[SUCCESS] PostgreSQL 连接成功！", flush=True)

        # 检查/创建 pgvector 扩展
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("[SUCCESS] pgvector 向量扩展检测与初始化成功！", flush=True)
        await conn.close()
        return True
    except Exception as e:
        print(f"[FAIL] PostgreSQL 异常: {e}", flush=True)
        return False


async def main():
    print("========================================", flush=True)
    print("MemAgent 基础设施连通性检测", flush=True)
    print("========================================", flush=True)
    r_ok = check_redis()
    p_ok = await check_postgres()
    print("========================================", flush=True)
    if r_ok and p_ok:
        print("[SUCCESS] 所有数据库与缓存环境已就绪！", flush=True)
    else:
        print("[WARNING] 部分数据库/缓存未启动，请启动后再次运行本脚本检测。", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
