# 加载环境变量
from dotenv import load_dotenv
# 导入LangGraph核心依赖：StateGraph(状态图)、START/END(内置起始/结束节点常量)
from langgraph.graph import StateGraph, END

from app.process.agent.node.node_entry import node_entry
from app.process.agent.node.node_answer_output import node_answer_output
from app.process.agent.node.node_llm_server import node_llm_server
from app.process.agent.node.node_search_server import  node_search_server
from app.process.agent.node.node_bge_embedding import  node_bge_embedding
from app.process.agent.node.node_summary import node_summary
from app.process.agent.state import EvaluationState

# 初始化环境变量
load_dotenv()

# 初始化langgraph状态图
workflow = StateGraph(EvaluationState)
# 注册所有子节点
workflow.add_node("node_entry", node_entry)
workflow.add_node("node_search_server", node_search_server)
workflow.add_node("node_bge_embedding", node_bge_embedding)
workflow.add_node("node_llm_server", node_llm_server)
workflow.add_node("node_answer_output", node_answer_output)
workflow.add_node("node_summary", node_summary)

# 设置入口节点
workflow.set_entry_point("node_entry")

# 定义线性流转边：node_entry -> node_search_server -> node_bge_embedding
workflow.add_edge("node_entry", "node_search_server")
workflow.add_edge("node_search_server", "node_bge_embedding")

def router_after_embedding(state: EvaluationState) -> str:
    """
    根据BGE-M3 向量相似度得分判断下一个节点路线
    similarity_score>0.8 -> node_answer_output
    similarity_score<0.4 -> node_answer_output
    0.4<similarity_score<0.8 -> node_llm_server
    :param state:
    :return:
    """
    if not state["needs_llm_check"]:
        return "node_answer_output"
    else:
        return "node_llm_server"

workflow.add_conditional_edges("node_bge_embedding",
                               router_after_embedding,
                               {
                                   "node_answer_output": "node_answer_output",
                                   "node_llm_server": "node_llm_server"
                               }

)
workflow.add_edge("node_llm_server", "node_answer_output")

def router_after_output(state: EvaluationState) -> str:
    """
    node_answer_output 执行完毕后判断：
    - 还有下一条数据 -> 回到 node_search_server 继续处理
    - 所有数据处理完毕 -> 进入 node_summary 汇总
    """
    current = state.get("current_index", 0)
    total = state.get("total_count", 0)
    if current + 1 <= total:
        return "node_search_server"
    else:
        return "node_summary"

workflow.add_conditional_edges("node_answer_output",
                               router_after_output,
                               {
                                   "node_search_server": "node_search_server",
                                   "node_summary": "node_summary"
                               })

workflow.add_edge("node_summary", END)

# 编译图节点对象
app = workflow.compile()

# print(app.get_graph().draw_mermaid())