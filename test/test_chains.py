from pprint import pprint
import pytest
from dotenv import load_dotenv

from src.workflow.chains.query_split import QuerySplit, question_splitter
from src.workflow.chains.generation import generation_chain
from src.workflow.chains.hallucination_grader import (
    HallucinationGrader,
    hallucination_grader,
)
from src.workflow.chains.retrieval_grader import GradeDocuments, retrieval_grader
from src.workflow.chains.router import RouteQuery, question_router
from src.workflow.chains.reflex import RouteReflex, reflex_router
from src.workflow.chains.search_prompt import SearchPromet, search_promet
from data.ingestion import retriever

load_dotenv()


@pytest.mark.parametrize(
    "question, expected_route",
    [
        pytest.param("hi?", "generate", id="簡短問候 → 回應生成"),
        pytest.param(
            "What is the difference between RAG and fine-tuning?",
            "vectorstore",
            id="技術問題 → 向量檢索",
        ),
        pytest.param(
            "What is the current weather in Tokyo?",
            "websearch",
            id="即時問題 → 網路搜尋",
        ),
    ],
)
def test_question_router(question: str, expected_route: str) -> None:
    """測試 router 導向正確的資料源(生成回應、向量檢索、網路搜尋)"""
    res: RouteQuery = question_router.invoke({"question": question})

    assert isinstance(res, RouteQuery)
    assert res.datasource == expected_route


def test_question_splitter() -> None:
    """測試使用者輸入拆分子問題"""
    question = "What is retrieval augmented generation?"
    result: QuerySplit = question_splitter.invoke(
        {"question": question, "chat_history": ""}
    )
    pprint(result)
    # 結構檢查
    assert isinstance(result, QuerySplit)
    # 切分數量檢查
    assert 1 <= len(result.sub_question) <= 5
    # 子問題內容檢查
    assert all(isinstance(q, str) and q.strip() for q in result.sub_question)


def run_retrieval_grader_test(question_to_check: str, expected: bool):
    """測試完整流程：子問題拆分 → 檢索 → 檢索評分 → 相關/不相關"""
    sub_questions = question_splitter.invoke(
        {"question": question_to_check, "chat_history": ""}
    ).sub_question
    merge_questions = [question_to_check] + sub_questions
    all_docs = {q: retriever.invoke(q) for q in merge_questions}

    for sq, docs in all_docs.items():
        if docs:
            doc = docs[0]
            print(doc.page_content[:200])
            res: GradeDocuments = retrieval_grader.invoke(
                {
                    "question": question_to_check,
                    "sub_questions": sq,
                    "document": doc.page_content,
                }
            )
            assert res.score == expected, (
                f"Failed for sub-question: {sq}, doc: {doc.page_content[:200]}"
            )


def test_retrival_grader_flow_yes() -> None:
    """測試檢索評分 → 相關"""
    run_retrieval_grader_test("what is LLM?", True)


def test_retrival_grader_flow_no() -> None:
    """測試檢索評分 → 不相關"""
    run_retrieval_grader_test("how to cook pasta with mushrooms", False)


def test_generation_chain() -> None:
    """測試生成器產生答案"""
    question = "How do language models work?"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke(
        {
            "reflex_prompt": "",
            "question": question,
            "sub_questions": "",
            "documents": docs,
            "chat_history": [],
        }
    )
    pprint(generation)
    assert generation is not None
    assert isinstance(generation, str)
    assert len(generation.strip()) > 0


def test_hallucination_grader_answer_yes() -> None:
    """測試幻覺評分器→答案基於事實"""
    question = "What are the benefits of vector databases?"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke(
        {
            "reflex_prompt": "",
            "question": question,
            "sub_questions": "",
            "documents": docs,
            "chat_history": [],
        }
    )
    res: HallucinationGrader = hallucination_grader.invoke(
        {
            "documents": docs,
            "chat_history": [],
            "generation": generation,
            "sub_questions": "",
        }
    )
    assert res.score


def test_hallucination_grader_answer_no() -> None:
    """測試幻覺評分器→答案不基於事實"""
    question = "What are the benefits of vector databases?"
    docs = retriever.invoke(question)

    res: HallucinationGrader = hallucination_grader.invoke(
        {
            "documents": docs,
            "chat_history": [],
            "generation": "To bake a perfect chocolate cake, you need to preheat the oven to 350 degrees",
            "sub_questions": "",
        }
    )
    assert not res.score


def test_reflex_router_generate_node() -> None:
    """測試 prev_node=generate 的情況"""
    prev_node = "generate"
    question = "What is retrieval augmented generation?"
    sub_questions = ["What is RAG?", "How does it work?"]
    documents = ["RAG combines retrieval and generation.", "It improves LLM accuracy."]
    generation = "RAG is a method to enhance LLM responses."

    result: RouteReflex = reflex_router.invoke(
        {
            "prev_node": prev_node,
            "question": question,
            "sub_questions": sub_questions,
            "documents": documents,
            "generation": generation,
        }
    )

    pprint(result)
    assert isinstance(result, RouteReflex)
    assert result.reflex_prompt is not None
    assert isinstance(result.reflex_prompt, str)
    assert len(result.reflex_prompt.strip()) > 0


def test_reflex_router_grade_document_node() -> None:
    """測試 prev_node=grade_document 的情況"""
    prev_node = "grade_document"
    question = "What is retrieval augmented generation?"
    sub_questions = ["What is RAG?", "How does it work?"]
    documents = ["RAG combines retrieval and generation.", "It improves LLM accuracy."]
    generation = ""  # 對 grade_document 不需要 generation

    result: RouteReflex = reflex_router.invoke(
        {
            "prev_node": prev_node,
            "question": question,
            "sub_questions": sub_questions,
            "documents": documents,
            "generation": generation,
        }
    )

    pprint(result)
    assert isinstance(result, RouteReflex)
    assert result.reflex_prompt is not None
    assert isinstance(result.reflex_prompt, str)
    assert len(result.reflex_prompt.strip()) > 0


def test_search_promet_basic() -> None:
    """測試 search_promet 生成查詢語句"""
    question = "What is retrieval augmented generation?"
    reflex_prompt = "生成回答缺少 documents 的支撐，需要補充相關資料"

    result: SearchPromet = search_promet.invoke(
        {"question": question, "reflex_prompt": reflex_prompt}
    )

    pprint(result)
    # 基本結構檢查
    assert isinstance(result, SearchPromet)
    assert result.search_prompt is not None
    assert isinstance(result.search_prompt, str)
    assert len(result.search_prompt.strip()) > 0
