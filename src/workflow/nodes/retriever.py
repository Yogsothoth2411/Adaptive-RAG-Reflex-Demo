from typing import Any, Dict, List
from src.workflow.state import GraphState
from data.ingestion import retriever


def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """
    向量檢索節點（支援多子問題）
    輸入:
        question: str
        sub_questions: List[str]
    輸出:
        documents: List[str] (聚合後的檢索結果)
    """
    question = state["question"]
    sub_questions: List[str] = state["sub_questions"].sub_question
    merge_questions = [question] + sub_questions

    aggregated_docs: List[str] = []
    print("--正在進行向量檢索--")
    aggregated_docs = []
    for sub_q in merge_questions:
        docs = retriever.invoke(sub_q)
        for doc in docs:
            if doc not in aggregated_docs:
                aggregated_docs.append(doc)

    return {
        "documents": aggregated_docs,
        "question": question,
        "sub_questions": sub_questions,
    }
