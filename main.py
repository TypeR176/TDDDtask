import json
from app.process.agent.main_graph import app
from app.core.logger import logger


def main():
    logger.info("===== RAG检索质量评估流程启动 =====")

    initial_state = {}
    result = app.invoke(initial_state)

    final_report = result.get("final_report", [])
    accuracy_rate = result.get("accuracy_rate", 0.0)

    logger.info(f"整体检索准确率: {accuracy_rate}")

    output_path = "output_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy_rate": accuracy_rate,
            "detail": final_report
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"评估报告已写入: {output_path}")
    logger.info("===== RAG检索质量评估流程结束 =====")


if __name__ == "__main__":
    main()