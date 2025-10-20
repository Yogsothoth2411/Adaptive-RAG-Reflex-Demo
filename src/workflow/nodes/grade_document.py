from typing import Any, Dict
from src.workflow.chains.retrieval_grader import retrieval_grader
from src.workflow.state import GraphState


def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    """
    確定檢索到的文件是否與問題相關，如果不相關會導向網路搜索節點

    Args:
        state (State): 工作流狀態
    Returns:
        Dict[str, Any]: 過濾掉不相關的文件，更新圖中的web_search標誌狀態
    """

    question = state["question"]
    sub_questions = state["sub_questions"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    print("--正在進行文檔相關性評分--")
    for doc in documents:
        score = retrieval_grader.invoke(
            {
                "question": question,
                "sub_questions": sub_questions,
                "document": doc.page_content,
            }
        )
        if score.score:
            filtered_docs.append(doc)
        else:
            web_search = True
            continue
    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search,
        "prev_node": "grade_document",
    }
