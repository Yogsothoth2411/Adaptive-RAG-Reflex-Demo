from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.models.model import llm_model


class RouteQuery(BaseModel):
    """將使用者的查詢路由到適當的子鏈"""

    datasource: Literal["generate", "vectorstore", "websearch"] = Field(
        ...,
        description="根據使用者的提問，選擇路由至 生成回應(generate)、向量資料庫(vectorstore) 或 網頁搜尋(websearch) 的子鏈",
        json_schema_extra={"example": "vectorstore"},
    )


llm = llm_model

structured_llm_router = llm.with_structured_output(RouteQuery)

system = """你是一個路由助理。你的任務是將使用者的查詢路由到合適的子鏈：生成回應(generate)、向量資料庫(vectorstore) 或 網頁搜尋(websearch)。

路由規則：
1. 如果查詢是日常對話、簡單問答、情境對話等，可以直接路由到 generate。
2. 如果查詢是知識型問題（例如 agents、prompt engineering、adversarial attacks)時，路由到 vectorstore。
3. 除非必要，不要直接讓 generate 嘗試回答無法從向量資料庫獲得資料的問題。
4. 僅在完全無法回答且 vectorstore 沒有資料時，才路由到 websearch。"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "{question}",
        ),
    ]
)

question_router = prompt | structured_llm_router
