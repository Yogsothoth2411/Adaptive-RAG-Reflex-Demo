from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from src.models.model import llm_model


class HallucinationGrader(BaseModel):
    """使用布林分數來評估LLM輸出的回覆是否基於事實。"""

    score: bool = Field(
        ...,
        description="答案是否基於事實。True表示回覆基於事實，False表示回覆包含幻覺。",
        json_schema_extra={"example": True},
    )


llm = llm_model
structured_llm_grader = llm.with_structured_output(HallucinationGrader)

system = """
你是一位嚴謹的事實檢查員，負責判斷 LLM 生成的回覆是否基於事實或合理依據。
請根據下列原則回答 True 或 False：

- 如果回答屬於日常對話（如問候、寒暄、閒聊），請回答 True。
- 如果回答內容有清楚的證據基礎（出自文件或對話歷史），請回答 True。
- 如果回答明顯虛構、缺乏文件或對話依據，請回答 False。

注意：
1. 對話歷史（chat_history）可作為次要的事實參考來源。
2. 文件內容（documents）為主要依據來源。
3. 子問題（sub_questions）僅作為背景輔助，不影響最終 True/False 判斷。
4. 僅回答 True 或 False，不要附加說明。
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "根據以下內容進行判斷：\n\n"
            "【文件內容】:\n{documents}\n\n"
            "【對話歷史】:\n{chat_history}\n\n"
            "【LLM生成回覆】:\n{generation}\n\n"
            "【輔助子問題（僅供參考）】:\n{sub_questions}\n\n"
            "請根據上述資訊回答：True 或 False。",
        ),
    ]
)

# 顯式定義RunnableSequence 提供靜態類型檢查
hallucination_grader: RunnableSequence = prompt | structured_llm_grader
