import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from app.eval import Evaluator


@pytest.mark.asyncio
async def test_evaluator_workflow():
    """测试评估引擎工作流与报告导出"""
    evaluator = Evaluator(data_path="data/eval_dialogues.jsonl")
    assert len(evaluator.dataset) == 200

    # 在单元测试中选取 20 条快测试验
    evaluator.dataset = evaluator.dataset[:20]

    base_m = await evaluator.evaluate_baseline()
    mem_m = await evaluator.evaluate_memagent()

    assert base_m["task_completion_rate"] >= 0.0
    assert mem_m["coref_accuracy"] >= 0.85
    assert mem_m["avg_token_usage"] > 0

    md_report = evaluator.generate_report(base_m, mem_m, output_md="tests/test_eval_report.md")
    assert "# MemAgent 多轮指代感知记忆系统" in md_report
    assert os.path.exists("tests/test_eval_report.md")

    # 清理临时测试报告
    if os.path.exists("tests/test_eval_report.md"):
        os.remove("tests/test_eval_report.md")


if __name__ == "__main__":
    pytest.main(["-vs", __file__])
