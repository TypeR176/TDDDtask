import sys
from app.process.agent.state import EvaluationState
from app.core.logger import logger
from search.ragServer import ragServer


def node_search_server(state: EvaluationState) -> EvaluationState:
    """
    检索服务节点：调用 ragServer 获取检索结果，按 current_index 取当前条的 top_chunk
    同时将当前条的 question 和 ground_truth 写入 state 供后续节点使用
    """
    current_node = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行LangGraph节点：{current_node}")

    idx = state.get("current_index", 0)
    questions = state.get("questions", [])
    ground_truths = state.get("ground_truths", [])

    # 从批量数据中取当前条
    state["question"] = questions[idx]
    state["ground_truth"] = ground_truths[idx]

    # 调用 ragServer 获取所有检索结果，按 question 匹配取 top_chunk
    rag_results = ragServer()
    current_question = questions[idx]

    top_chunk = ""
    for item in rag_results:
        if item.get("question") == current_question:
            top_chunk = item.get("top_chunk", "")
            break

    state["top_chunk"] = top_chunk

    logger.info(f"--- 检索完成 | 索引: {idx} | question: {current_question} | top_chunk: {top_chunk[:50]}... ---")

    return state
