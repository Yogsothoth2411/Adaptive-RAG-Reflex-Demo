from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from src.models.model import llm_model


class GradeAnswer(BaseModel):
    """對最終生成的答案進行評分，確認是否能回答使用者提問"""

    score: bool = Field(
        ..., description="生成的答案是否能回答使用者提問，True or False", example=True
    )


llm = llm_model
structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """
你是一個答案評分助理，負責判斷生成的答案是否有助於回答使用者當前的問題。
請根據下列原則回答 True 或 False：

- 如果答案能夠直接或間接解答使用者當前提問，請回答 True。
- 如果答案無關、模糊或未能回應問題，請回答 False。
- 如果對話屬於寒暄、社交互動等情境（如 "你好"、"謝謝"），請回答 True。

注意：
1. 對話歷史（chat_history）能提供上下文，請據此理解使用者當前意圖。
2. 子問題（sub_questions）僅作為輔助，不影響最終 True/False。
3. 僅回答 True 或 False，不要附加說明。
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "使用者原始提問：\n{question}\n\n"
            "生成的答案：\n{generation}\n\n"
            "對話歷史（可用於理解上下文）：\n{chat_history}\n\n"
            "輔助子問題（可參考）：\n{sub_questions}\n\n"
            "請根據上述內容回答 True 或 False。",
        ),
    ]
)

# 顯式定義RunnableSequence 提供靜態類型檢查
answer_grader: RunnableSequence = prompt | structured_llm_grader
