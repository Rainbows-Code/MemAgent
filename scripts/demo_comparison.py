import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
from app.chat import BaselineAgent
from app.memory.working import WorkingMemory
from app.memory.episodic import EpisodicMemory
from app.memory.semantic import SemanticMemory
from app.memory.coreference import CoreferenceResolver
from app.memory.retrieval import MemoryRetriever


async def run_comparison_demo():
    print("========================================================================")
    print("MemAgent vs 原版 Baseline 实际运行对比演示")
    print("========================================================================\n")

    user_id = "demo_user_contrast"

    # 初始化原版 Baseline Agent
    baseline_agent = BaselineAgent(user_id=user_id, window_size=10)

    # 初始化 MemAgent 系统
    working = WorkingMemory(user_id=user_id, window_size=2)
    episodic = EpisodicMemory()
    semantic = SemanticMemory()
    await episodic.init_db()
    await semantic.init_db()
    working.clear()
    await episodic.clear(user_id)
    await semantic.clear(user_id)

    resolver = CoreferenceResolver()
    retriever = MemoryRetriever(working_memory=working, episodic_memory=episodic, semantic_memory=semantic)

    # 模拟多轮对话动作
    dialogue_turns = [
        "我计划去厦门出差，帮我看看全季酒店靠窗大床房",
        "记一下，我的收货地址是厦门大学思明校区",
        "它包含早餐吗？",  # 👈 经典指代 1
        "帮我改一下地址为北京大学",  # 👈 经典偏好修改与指代 2
    ]

    for turn_idx, query in enumerate(dialogue_turns, 1):
        print(f"------------ [第 {turn_idx} 轮对话] 用户输入: '{query}' ------------")

        # ---------------- 1. 原版 Baseline 表现 ----------------
        base_res = await baseline_agent.achat(query)
        base_tokens = base_res["usage"].get("total_tokens", 0)

        print(f"[原版 Baseline] (无指代解析与记忆检索):")
        print(f"   - 发送给 LLM 的 Context 长度: {len(base_res['context_messages'])} 条消息 (全量历史盲目拼接)")
        print(f"   - 本轮 Token 消耗: {base_tokens} Tokens")
        print(f"   - 指代解析能力: 无 (发送原始模糊问题 '{query}')")

        # ---------------- 2. MemAgent 系统表现 ----------------
        facts = await semantic.get_all_facts(user_id)
        recent_msgs = working.get_history()

        # 执行指代消解
        coref = resolver.resolve_coreference(current_query=query, recent_messages=recent_msgs, facts=facts)

        # 模拟偏好识别与语义更新
        if "地址是" in query:
            addr_val = query.split("地址是")[-1].strip()
            await semantic.set_fact(user_id, "address", addr_val, source="user_explicit")
        elif "改一下地址" in query:
            new_addr = query.split("为")[-1].strip()
            await semantic.set_fact(user_id, "address", new_addr, source="user_explicit")

        # 按需记忆检索与 Prompt 构建
        ctx = await retriever.retrieve_context(user_id, coref["resolved_query"])
        prompt_msgs = retriever.build_injected_prompt("你是贴心助手", ctx)
        prompt_msgs.append({"role": "user", "content": coref["resolved_query"]})

        # 更新工作记忆
        working.add_message("user", query)
        working.add_message("assistant", f"关于{coref['resolved_query']}处理完成")

        mem_tokens = len(str(prompt_msgs)) // 2

        print(f"[MemAgent 系统] (三层记忆 + 指代消解):")
        print(f"   - 指代消解识别: has_coref={coref['has_coreference']}")
        if coref['has_coreference']:
            print(f"     -> 识别指代词 '{query}' 转化为标准意图: '{coref['resolved_query']}'")
            print(f"     -> 匹配置信度得分: {coref['confidence']} (目标实体: '{coref['target_entity']}')")
        print(f"   - 按需注入的语义偏好事实: {[f['key'] + '=' + f['value'] for f in ctx['user_facts']]}")
        print(f"   - 本轮 Context Token 消耗: {mem_tokens} Tokens\n")

    # 查阅冲突覆盖历史归档
    history = await semantic.get_fact_history(user_id, "address")
    if history:
        print("------------------------------------------------------------------------")
        print("[AUDIT] 语义记忆数据库审计: 观察地址变更的历史归档记录 (user_fact_history)")
        for h in history:
            print(f"   - 属性 address 变动: 旧值 '{h.get('old_value')}' -> 新值 '{h.get('new_value')}'")
    print("========================================================================\n")


if __name__ == "__main__":
    asyncio.run(run_comparison_demo())
