import json
from pathlib import Path


def ragServer() -> list:
    """
    模拟 RAG 检索服务
    直接读取 test_data.json 返回检索结果列表
    """
    data_path = Path(__file__).parent / "test_data.json"
    return json.loads(data_path.read_text(encoding="utf-8"))