from typing import Any, Dict
from src.workflow.state import GraphState
from src.workflow.chains.query_split import question_splitter


def query_splitter_node(state: GraphState) -> Dict[str, Any]:
    """
    使用者查詢拆分節點，將使用者的提問輸入拆分為多個關聯子提問

    Args:
        question: str
        chat_history: Annotated[list[AnyMessage], add_messages]

    Returns:
        question: str
        sub_questions: List[str]
    """
    print("--正在拆分子問題--")
    question = state["question"]
    chat_history = state["chat_history"]
    sub_questions = question_splitter.invoke(
        {
            "question": question,
            "chat_history": chat_history,
        }
    )

    return {
        "question": question,
        "sub_questions": sub_questions,
    }
