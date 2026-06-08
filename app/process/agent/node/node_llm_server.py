import json
import re
import sys
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.load_prompt import load_prompt
from app.process.agent.state import EvaluationState
from app.core.logger import logger
from lm.lm_utils import get_llm_client


def call_llm(question, top_chunk, ground_truth):
    # 构建提示词
    human_prompt = load_prompt("llm_eval", top_chunk=top_chunk, ground_truth=ground_truth, question=question)
    system_prompt = load_prompt("llm_eval_system")
    # 获取模型对象
    llm = get_llm_client(json_mode=True)
    # 执行调用
    messages = [
        HumanMessage(content=human_prompt),
        SystemMessage(content=system_prompt)
    ]
    response = llm.invoke(messages)
    # 返回结果
    return response.content


def node_llm_server(state: EvaluationState) -> EvaluationState:
    """
    对得分在0.4-0.8之间的回答进行AI复核
    """
    current_node = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行LangGraph节点：{current_node}")
    logger.info("--- LLM 二次复核启动 ---")

    try:
        question = state.get("question")
        ground_truth = state.get("ground_truth")
        top_chunk = state.get("top_chunk")

        if not all([question, ground_truth, top_chunk]):
            logger.warning("LLM复核输入缺失：question/ground_truth/top_chunk 不完整")
            state["is_hit"] = False
            state["reasoning"] = "输入数据不完整，无法进行LLM复核"
            return state

        raw_response = call_llm(question, top_chunk, ground_truth)
        logger.info(f"LLM 原始返回：{raw_response}")

        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            raise json.JSONDecodeError("未找到JSON内容", raw_response, 0)
        state["is_hit"] = result.get("is_hit", False)
        state["reasoning"] = result.get("reasoning", "LLM未返回有效理由")

        logger.info(f"--- LLM 复核完成 | 命中: {state['is_hit']} | 理由: {state['reasoning']} ---")

    except json.JSONDecodeError as e:
        logger.error(f"LLM返回JSON解析失败：{str(e)}，原始内容：{raw_response}")
        state["is_hit"] = False
        state["reasoning"] = "LLM返回格式异常，无法解析"
    except Exception as e:
        logger.error(f"LLM复核节点执行失败：{str(e)}", exc_info=True)
        state["is_hit"] = False
        state["reasoning"] = f"LLM复核异常：{str(e)}"

    return state