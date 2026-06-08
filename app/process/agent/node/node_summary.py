import sys
from app.process.agent.state import EvaluationState
from app.core.logger import logger


def node_summary(state: EvaluationState) -> EvaluationState:
    """
    汇总节点：统计所有评估结果，计算整体检索准确率
    """
    current_node = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行LangGraph节点：{current_node}")

    report = state.get("final_report", [])
    total = state.get("total_count", len(report))
    hit_count = sum(1 for item in report if item.get("is_hit"))
    accuracy_rate = round(hit_count / total, 4) if total > 0 else 0.0

    state["accuracy_rate"] = accuracy_rate

    logger.info(f"--- 汇总完成 | 总条数: {total} | 命中数: {hit_count} | 准确率: {accuracy_rate} ---")

    return state