import json
import os
from typing import List
from app.process.agent.state import EvaluationState
import sys
from app.core.logger import logger

def node_entry(state: EvaluationState) -> EvaluationState:
    """
    拿到test_data/qa_pairs.json，将其中每一条question和answer都放到state中，顺序一定要匹配好，不能乱
    """
    # 获取当前节点名称，用于日志和任务状态记录
    current_node = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行LangGraph节点：{current_node}")

    qa_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "test_data", "qa_pairs.json")
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs: List[dict] = json.load(f)
    
    state["questions"] = [item["question"] for item in qa_pairs]
    state["ground_truths"] = [item["answer"] for item in qa_pairs]
    state["total_count"] = len(qa_pairs)
    state["current_index"] = 0
    state["final_report"] = []
    
    return state
