import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import httpx

st.set_page_config(
    page_title="MemAgent - 多轮指代感知记忆 Agent",
    page_icon="🧠",
    layout="wide",
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.title("🧠 MemAgent 多轮指代感知记忆 Agent 控制台")
st.caption("基于 Redis (工作记忆) + PostgreSQL/pgvector (情景&语义记忆) + 指代消解 + 动态遗忘")

# Session State 初始化
if "user_id" not in st.session_state:
    st.session_state.user_id = "user_demo_01"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_meta" not in st.session_state:
    st.session_state.last_meta = {}

# 侧边栏: 用户切换与记忆状态监控
with st.sidebar:
    st.header("⚙️ 控制面板与记忆监控")
    user_id = st.text_input("当前 User ID", value=st.session_state.user_id)
    if user_id != st.session_state.user_id:
        st.session_state.user_id = user_id
        st.session_state.messages = []
        st.experimental_rerun()

    st.markdown("---")
    st.subheader("🔍 三层记忆实时镜像")

    if st.button("🔄 刷新记忆状态"):
        try:
            res = httpx.get(f"{API_BASE_URL}/memory/{st.session_state.user_id}", timeout=3.0)
            if res.status_code == 200:
                st.session_state.memory_data = res.json()
            else:
                st.error("获取记忆失败")
        except Exception as e:
            st.warning(f"无法连通后端 API ({e})")

    mem_data = st.session_state.get("memory_data", {})
    if mem_data:
        st.markdown("**【语义记忆 (用户偏好)】**")
        facts = mem_data.get("semantic_facts", [])
        if facts:
            for f in facts:
                st.info(f"🔑 **{f['key']}**: {f['value']}")
        else:
            st.caption("暂无结构化偏好")

        st.markdown("**【情景记忆 (向量摘要)】**")
        summaries = mem_data.get("episodic_summaries", [])
        if summaries:
            for s in summaries:
                st.success(f"📝 {s['summary']}")
        else:
            st.caption("暂无向量摘要")

    st.markdown("---")
    if st.button("🗑️ 清空当前对话历史"):
        st.session_state.messages = []
        st.success("对话界面已重置")

# 主界面: 聊天区
st.subheader("💬 多轮对话与指代感知解析")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "meta" in msg:
            m = msg["meta"]
            if m.get("has_coreference"):
                st.caption(f"🎯 **指代消解**: `{m['original_message']}` 👉 `{m['resolved_query']}` (目标实体: {m.get('target_entity')})")

# 用户输入
if user_input := st.chat_input("请输入您的问题 (支持使用“它、那个、第一个、改一下地址”等指代)..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🧠 检索三层记忆、执行指代消解与 LLM 推理中..."):
            try:
                payload = {
                    "user_id": st.session_state.user_id,
                    "message": user_input,
                }
                res = httpx.post(f"{API_BASE_URL}/chat", json=payload, timeout=10.0)

                if res.status_code == 200:
                    data = res.json()
                    reply = data["response"]
                    st.markdown(reply)

                    meta_info = {
                        "original_message": data["original_message"],
                        "resolved_query": data["resolved_query"],
                        "has_coreference": data["has_coreference"],
                        "target_entity": data["target_entity"],
                        "usage": data["usage"],
                    }

                    if data["has_coreference"]:
                        st.success(f"🎯 成功识别指代！消解意图: **{data['resolved_query']}** (实体: `{data['target_entity']}`)")

                    st.caption(f"⚡ Token 消耗: Prompt `{data['usage'].get('prompt_tokens', 0)}` | Completion `{data['usage'].get('completion_tokens', 0)}` | Total `{data['usage'].get('total_tokens', 0)}`")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": reply,
                        "meta": meta_info,
                    })
                else:
                    st.error(f"后端响应异常 ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"调用 API 失败: {e}")
