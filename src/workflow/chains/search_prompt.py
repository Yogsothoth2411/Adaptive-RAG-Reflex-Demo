from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.models.model import llm_model


class SearchPromet(BaseModel):
    """對反思節點提出的缺少資訊產生查詢語句"""

    search_prompt: str = Field(
        ...,
        description="根據使用者的提問與反思建議，生成明確且可直接用於搜尋引擎的查詢語句。",
        json_schema_extra={"example": "AI 模型在工業應用中的資料品質挑戰 解決方案"},
    )


llm = llm_model

structured_llm_promet = llm.with_structured_output(SearchPromet)

system = """
你是一個網路搜尋查詢語句生成助理。請根據使用者的問題 (question)
以及反思建議 (reflex_prompt)，生成最合適的搜尋引擎查詢語句。

請遵守以下原則：
1. 搜尋語句應具體、可直接用於 Google 或 Bing。
2. 若 reflex_prompt 提到缺失資訊，請聚焦於該缺失面向。
3. 不要解釋搜尋語句的意圖，也不要加上引號或多句話，只輸出單一查詢語句。
4. 若資訊不足，請推斷最可能的搜尋方向（如技術、年份、應用領域等）。
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "根據以下內容提出改進建議。\n\n使用者提問: {question}\n\n反思建議: {reflex_prompt}",
        ),
    ]
)

search_promet = prompt | structured_llm_promet
