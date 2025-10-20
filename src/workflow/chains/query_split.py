from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from src.models.model import llm_model, trimer


class QuerySplit(BaseModel):
    """將使用者的查詢拆分為多個可幫助解答提問的子問題"""

    sub_question: List[str] = Field(
        ...,
        description="將使用者的查詢拆分為多個可幫助解答提問的子問題。每個子問題應簡短且具體，方便後續處理。",
        min_length=1,
        max_length=5,
        json_schema_extra={
            "example": [
                "AI如何影響醫療產業？",
                "AI在醫療影像分析中的應用有哪些？",
                "AI技術如何改善診斷準確率？",
            ]
        },
    )


llm = llm_model

structured_llm_router = llm.with_structured_output(QuerySplit)

system = """
你是一個提問分析助理，負責將使用者的查詢語句拆分為可以幫助解答原始提問的多個具體子問題。
請確保：
1. 子問題彼此之間不重複、具體且有邏輯關聯。
2. 子問題的數量介於1到5之間。
3. 子問題應該能被檢索或用於搜尋來協助回答原始問題。
4. 回答時僅需輸出結構化的子問題列表，不需額外文字、說明或總結。
5. 若對話歷史（chat_history）中存在線索，可適度參考；若無，請完全依據使用者原始提問。
在輸出子問題前，請先在內部思考這些子問題是否共同能解答原始提問，但不要輸出任何思考過程。
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "使用者提問： {question} \n\n對話歷史： {chat_history}",
        ),
    ]
)

question_splitter: RunnableSequence = prompt | trimer | structured_llm_router
