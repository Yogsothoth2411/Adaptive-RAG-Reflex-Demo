from typing import List, TypedDict, Annotated
from langgraph.graph.message import AnyMessage, add_messages


class GraphState(TypedDict):
    """包含 查詢、生成、對話歷史 和 文檔 的狀態"""

    question: str
    sub_questions: List[str]
    chat_history: Annotated[list[AnyMessage], add_messages]
    documents: List[str]
    generation: str
    reflex_prompt: str
    web_search: bool
    prev_node: str
    loop_count: int
