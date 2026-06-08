from typing import TypedDict, List, Optional

class EvaluationState(TypedDict):
    # 1. 批量输入 (由 node_entry 写入)
    questions: List[str]            # 所有用户问题列表
    ground_truths: List[str]        # 所有标准答案列表
    total_count: int                # 数据总条数
    current_index: int              # 当前处理的数据索引

    # 2. 单条处理中 (逐条流转)
    question: str                   # 当前用户问题
    ground_truth: str               # 当前标准答案
    top_chunk: str                  # 当前检索片段

    # 3. 评分与路由 (由 BGE 节点写入)
    similarity_score: float         # BGE-M3 向量相似度得分
    needs_llm_check: bool           # 路由标志：是否需要 LLM 介入

    # 4. 评估结论
    is_hit: bool                    # 是否命中
    reasoning: str                  # 判定的理由/思路

    # 5. 最终输出 (每条数据一个 dict，共 total_count 条)
    final_report: List[dict]        # 所有评估报告列表
    accuracy_rate: float              # 整体检索准确率
