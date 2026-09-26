import json
import os

SCENARIOS = [
    {"domain": "酒店预订", "pronoun": "它", "target": "全季酒店靠窗大床房", "query": "它包含早餐吗？", "req_mem": "episodic"},
    {"domain": "地址修改", "pronoun": "改一下地址", "target": "厦门大学思明校区", "query": "帮我改一下地址", "req_mem": "semantic"},
    {"domain": "餐饮外卖", "pronoun": "那个", "target": "麻辣小龙虾特辣版", "query": "那个能少放点辣椒吗？", "req_mem": "working"},
    {"domain": "物流查询", "pronoun": "它", "target": "顺丰快递单号SF123456", "query": "它现在到哪里了？", "req_mem": "working"},
    {"domain": "日程安排", "pronoun": "第一个", "target": "周一上午9点的部门周会", "query": "把第一个会议改到下午3点", "req_mem": "semantic"},
]


def generate_eval_dataset(output_path: str = "data/eval_dialogues.jsonl", total_samples: int = 200):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    items = []

    for i in range(total_samples):
        tmpl = SCENARIOS[i % len(SCENARIOS)]
        idx = (i // len(SCENARIOS)) + 1

        target_entity = f"{tmpl['target']}_{idx}" if idx > 1 else tmpl["target"]

        dialogue_history = [
            {"role": "user", "content": f"我想了解一下{target_entity}的相关信息"},
            {"role": "assistant", "content": f"没问题，我已经为你查到了{target_entity}的具体细节。"},
            {"role": "user", "content": f"上次我选的{target_entity}体验非常不错。"},
            {"role": "assistant", "content": f"好的，我已经记录了你对{target_entity}的偏好。"}
        ]

        item = {
            "id": i + 1,
            "domain": tmpl["domain"],
            "dialogue_history": dialogue_history,
            "current_query": tmpl["query"],
            "coreference_pronoun": tmpl["pronoun"],
            "coreference_target": target_entity,
            "expected_answer": f"关于{target_entity}的答复",
            "required_memory": tmpl["req_mem"],
        }
        items.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"成功生成 {len(items)} 条标准多轮指代评估测试集: {output_path}")


if __name__ == "__main__":
    generate_eval_dataset()
