

import sys
from langchain_core.messages import HumanMessage
from app.core.load_prompt import load_prompt
from app.process.agent.state import EvaluationState
from app.core.logger import logger
from lm.lm_utils import get_llm_client


def node_answer_output(state: EvaluationState) -> EvaluationState:
    """
    最终报告节点：调用 LLM 生成理由文本，追加到 final_report 列表，并递增索引
    """
    current_node = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行LangGraph节点：{current_node}")
    logger.info("--- 生成最终评估报告 ---")

    try:
        human_prompt = load_prompt(
            "answer",
            question=state.get("question", ""),
            ground_truth=state.get("ground_truth", ""),
            top_chunk=state.get("top_chunk", ""),
            similarity_score=state.get("similarity_score", 0.0),
            is_hit=state.get("is_hit", False),
        )

        llm = get_llm_client()
        response = llm.invoke([HumanMessage(content=human_prompt)])
        reasoning = response.content

    except Exception as e:
        logger.error(f"评估报告生成失败：{str(e)}", exc_info=True)
        reasoning = f"报告生成异常：{str(e)}"

    # 组装当前条的报告
    report_item = {
        "question": state.get("question", ""),
        "top_chunk": state.get("top_chunk", ""),
        "is_hit": state.get("is_hit", False),
        "reasoning": reasoning,
    }

    if "final_report" not in state or state["final_report"] is None:
        state["final_report"] = []
    state["final_report"].append(report_item)

    state["current_index"] = state.get("current_index", 0) + 1

    logger.info(f"--- 报告追加完成 | 当前索引: {state['current_index']} | 累计报告数: {len(state['final_report'])} ---")

    return state