from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from src.models.model import llm_model


class GradeDocuments(BaseModel):
    """使用布林分數對檢索到的文件進行評分"""

    score: bool = Field(
        ...,
        description="根據使用者的提問，對檢索到的文件進行布林評分，若文件相關則為 True，否則為 False",
        json_schema_extra={"example": True},
    )


llm = llm_model
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """你是一個評分助理，負責根據使用者的提問以及拆分的子問題對檢索到的文件進行布林評分。若文件包含與提問相關的關鍵字或語意內容 則為 True，否則為 False。"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "根據使用者的提問，對檢索到的文件進行布林評分。若文件相關則為 True，否則為 False。\n\n使用者的提問: {question}\n\n拆分的子問題: {sub_questions}\n\n檢索到的文件: {document}",
        ),
    ]
)

# 顯式定義RunnableSequence 提供靜態類型檢查
retrieval_grader: RunnableSequence = prompt | structured_llm_grader
