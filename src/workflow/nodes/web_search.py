from typing import Any, Dict
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_tavily import TavilySearch
from src.workflow.state import GraphState
from src.workflow.chains.search_prompt import search_promet

load_dotenv()

web_search_tool = TavilySearch(max_results=3)


def web_search_node(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    reflex_prompt = state["reflex_prompt"]
    documents = state["documents"]
    print("--正在生成關鍵字進行網路搜尋--")
    web_search_query = search_promet.invoke(
        {
            "question": question,
            "reflex_prompt": reflex_prompt,
        }
    )
    web_search_query = web_search_query.search_prompt
    print(f"--關鍵字： {web_search_query}--")
    print("--正在進行網路搜尋--")
    tavily_results = web_search_tool.invoke({"query": web_search_query})["results"]
    joined_tavily_results = "\n".join([t["content"] for t in tavily_results])

    web_results = Document(page_content=joined_tavily_results)

    documents.append(web_results)

    return {"documents": documents, "question": question}
