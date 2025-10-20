from typing import Any, Dict
from src.workflow.state import GraphState
from src.workflow.chains.reflex import reflex_router


def reflection_node(state: GraphState) -> Dict[str, Any]:
    """
    反思節點，反思前一節點，輸出修改建議。

    Args:
        prev_node: str
        question: str
        sub_questions: List[str]
        documents: List[str]
        generation: str
    Returns:
        reflex_prompt: str
    """
    print("--正在反思提供修改建議--")
    prev_node = state["prev_node"]
    question = state["question"]
    sub_questions = state["sub_questions"]
    documents = state["documents"]
    generation = state["generation"]
    loop_count = state["loop_count"]
    loop_count += 1
    reflex_prompt = reflex_router.invoke(
        {
            "prev_node": prev_node,
            "question": question,
            "sub_questions": sub_questions,
            "documents": documents,
            "generation": generation,
        }
    )

    return {
        "reflex_prompt": reflex_prompt,
        "loop_count": loop_count,
    }
