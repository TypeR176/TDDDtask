import sys
import numpy as np
from app.process.agent.state import EvaluationState
from app.core.logger import logger
from lm.embedding_utils import generate_embeddings

def node_bge_embedding(state: EvaluationState) -> EvaluationState:
    """
    BGE-M3 文本向量化处理：测算 RAG 答案是否答非所问
    核心逻辑：
        1. 拼接 top_chunk 和 ground_truth
        2. 调用 BGE-M3 生成双向量
        3. 计算余弦相似度并归一化到 0-1 之间
        4. 将 similarity_score 写入 state
    """
    current_node = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行LangGraph节点：{current_node}")
    logger.info("--- BGE-M3 语义相似度测算启动 ---")

    try:
        top_chunk = state.get("top_chunk")
        ground_truth = state.get("ground_truth")

        if not top_chunk or not ground_truth:
            logger.warning("向量化输入无效：top_chunk 或 ground_truth 缺失")
            state["similarity_score"] = 0.0
            return state

        logger.info(f"测算文本：top_chunk='{top_chunk}' | ground_truth='{ground_truth}'")

        embeddings = generate_embeddings([top_chunk, ground_truth])
        chunk_vec = np.array(embeddings["dense"][0])
        truth_vec = np.array(embeddings["dense"][1])

        cos_sim = float(np.dot(chunk_vec, truth_vec) / (np.linalg.norm(chunk_vec) * np.linalg.norm(truth_vec)))
        norm_score = float((cos_sim + 1.0) / 2.0)

        state["similarity_score"] = round(norm_score, 4)
        
        # 根据相似度分数判断是否需要 LLM 介入
        # 0.4~0.8 之间为灰色地带，需要 LLM 二次判定
        if 0.4 < norm_score < 0.8:
            state["needs_llm_check"] = True
        elif norm_score >= 0.8:
            state["needs_llm_check"] = False
            state["is_hit"] = True
            state["reasoning"] = "向量相似度>=0.8，语义高度一致，直接判定命中"
        else:
            state["needs_llm_check"] = False
            state["is_hit"] = False
            state["reasoning"] = "向量相似度<0.4，语义差异过大，直接判定未命中"
        
        logger.info(f"--- BGE-M3 测算完成 | 得分: {state['similarity_score']} | 需LLM判定: {state['needs_llm_check']} ---")

    except Exception as e:
        logger.error(f"BGE-M3 节点执行失败：{str(e)}", exc_info=True)
        state["similarity_score"] = 0.0
        state["needs_llm_check"] = False

    return state