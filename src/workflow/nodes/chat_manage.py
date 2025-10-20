from typing import Any, Dict
from src.workflow.state import GraphState
from langchain_core.messages import AIMessage, SystemMessage


def manager(state: GraphState) -> Dict[str, Any]:
    """管理每輪的對話紀錄"""
    print("正在儲存本輪的對話歷史")
    reflex_prompt = state["reflex_prompt"]
    documents = state["documents"]
    chat_history = state["chat_history"]
    generation = state["generation"]

    # 保存反思提示
    if reflex_prompt:
        chat_history.append(SystemMessage(content=f"[Reflex Prompt]\n{reflex_prompt}"))

    # 保存檢索文件資訊
    if documents:
        doc_text = "\n".join([doc.page_content for doc in documents])
        chat_history.append(SystemMessage(content=f"[Retrieved Documents]\n{doc_text}"))

    # 保存模型生成
    chat_history.append(AIMessage(content=generation))

    return {
        "generation": generation,
        "chat_history": chat_history,
    }
