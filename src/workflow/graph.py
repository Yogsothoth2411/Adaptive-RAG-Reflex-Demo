from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from src.workflow.chains.hallucination_grader import hallucination_grader
from src.workflow.chains.answer_grader import answer_grader
from src.workflow.chains.router import RouteQuery, question_router
from src.workflow.consts import (
    GENERATE_NODE,
    GRADE_DOCUMENT_NODE,
    QUERY_SPLITTER_NODE,
    RETRIEVER_NODE,
    REFLECTION_NODE,
    WEB_SEARCH_NODE,
    CHAT_MANAGE_NODE,
)
from src.workflow.nodes.generate import generate
from src.workflow.nodes.grade_document import grade_documents_node
from src.workflow.nodes.retriever import retrieve_node
from src.workflow.nodes.web_search import web_search_node
from src.workflow.nodes.query_splitter import query_splitter_node
from src.workflow.nodes.reflection import reflection_node
from src.workflow.nodes.chat_manage import manager
from src.workflow.state import GraphState
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


def decide_generate(state):
    return REFLECTION_NODE if state["web_search"] else GENERATE_NODE


def grade_generation_grounder_in_documents_and_question(state):
    loop_count = state["loop_count"]

    if loop_count >= 3:
        print("--已達反思迴圈上限，直接返回當前答案--")
        return "max_loop"
    question = state["question"]
    sub_questions = state["sub_questions"]
    documents = state["documents"]
    generation = state["generation"]
    chat_history = state["chat_history"]
    print("--正在對答案進行評分--")
    hallucination_score = hallucination_grader.invoke(
        {
            "documents": documents,
            "generation": generation,
            "sub_questions": sub_questions,
            "chat_history": chat_history,
        }
    )

    if hallucination_score.score:
        score = answer_grader.invoke(
            {
                "question": question,
                "generation": generation,
                "sub_questions": sub_questions,
                "chat_history": chat_history,
            }
        )
        if score.score:
            return "useful"
        else:
            print(f"-- 當前第 {loop_count} 次 反思迴圈--")
            return "not useful"
    else:
        print("--未通過幻覺評分--")
        return "not supported"


def route_question(state: GraphState) -> str:
    source: RouteQuery = question_router.invoke({"question": state["question"]})
    route_map = {
        "generate": GENERATE_NODE,
        "vectorstore": QUERY_SPLITTER_NODE,
        "websearch": WEB_SEARCH_NODE,
    }
    print(f"--正在路由導向→{source.datasource}--")
    return route_map.get(source.datasource, GENERATE_NODE)


def route_reflex(state):
    return WEB_SEARCH_NODE if state["web_search"] else GENERATE_NODE


workflow = StateGraph(GraphState)
workflow.add_node(QUERY_SPLITTER_NODE, query_splitter_node)
workflow.add_node(RETRIEVER_NODE, retrieve_node)
workflow.add_node(GRADE_DOCUMENT_NODE, grade_documents_node)
workflow.add_node(REFLECTION_NODE, reflection_node)
workflow.add_node(GENERATE_NODE, generate)
workflow.add_node(WEB_SEARCH_NODE, web_search_node)
workflow.add_node(CHAT_MANAGE_NODE, manager)

workflow.set_conditional_entry_point(
    route_question,
    {
        # 函式回傳值 : 進入節點名
        WEB_SEARCH_NODE: WEB_SEARCH_NODE,
        QUERY_SPLITTER_NODE: QUERY_SPLITTER_NODE,
        GENERATE_NODE: GENERATE_NODE,
    },
)
workflow.add_edge(QUERY_SPLITTER_NODE, RETRIEVER_NODE)
workflow.add_edge(RETRIEVER_NODE, GRADE_DOCUMENT_NODE)
workflow.add_edge(CHAT_MANAGE_NODE, END)

workflow.add_conditional_edges(
    GRADE_DOCUMENT_NODE,
    decide_generate,
    {REFLECTION_NODE: REFLECTION_NODE, GENERATE_NODE: GENERATE_NODE},
)

workflow.add_conditional_edges(
    REFLECTION_NODE,
    route_reflex,
    {WEB_SEARCH_NODE: WEB_SEARCH_NODE, GENERATE_NODE: GENERATE_NODE},
)

workflow.add_conditional_edges(
    GENERATE_NODE,
    grade_generation_grounder_in_documents_and_question,
    {
        "not supported": GENERATE_NODE,
        "useful": CHAT_MANAGE_NODE,
        "not useful": REFLECTION_NODE,
        "max_loop": CHAT_MANAGE_NODE,
    },
)
workflow.add_edge(WEB_SEARCH_NODE, GENERATE_NODE)
memory = InMemorySaver()
app = workflow.compile(checkpointer=memory)
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
