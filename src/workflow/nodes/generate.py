from typing import Any, Dict
from src.workflow.chains.generation import generation_chain
from src.workflow.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    """使用檢索文檔及使用者提問生成回覆"""
    print("正在生成回覆Answer")
    reflex_prompt = state["reflex_prompt"]
    question = state["question"]
    sub_questions = state["sub_questions"]
    documents = state["documents"]
    chat_history = state["chat_history"]
    generation = generation_chain.invoke(
        {
            "reflex_prompt": reflex_prompt,
            "question": question,
            "sub_questions": sub_questions,
            "documents": documents,
            "chat_history": chat_history,
        }
    )
    return {
        "generation": generation,
        "prev_node": "generate",
        "chat_history": chat_history,
    }
