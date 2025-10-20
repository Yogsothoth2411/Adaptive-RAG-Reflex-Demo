from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.models.model import llm_model


class RouteReflex(BaseModel):
    """將對節點的問題反思並產生建議"""

    reflex_prompt: str = Field(
        ...,
        description="根據輸入的各項資訊產生改進建議",
        json_schema_extra={
            "example": "生成回答包含幻覺或缺少支持證據，建議在回答中加入 documents 中的相關事實，並注意與使用者問題匹配。"
        },
    )


llm = llm_model

structured_llm_router = llm.with_structured_output(RouteReflex)

system = """
你是一個改進評論員，負責根據前一個節點的結果提出改進建議。請根據 prev_node 的值選擇輸出內容：

1. 如果 prev_node 是 generate：
   - 請檢查 generation 的回答是否有幻覺或缺少 documents 的支撐
   - 生成一個改進建議(reflex_prompt)，內容應針對如何修改生成回答以增加真實性和完整性

2. 如果 prev_node 是 grade_document：
   - 請檢查 documents 是否充分回答 sub_questions
   - 生成一個改進建議(reflex_prompt)，內容應針對需要補充哪些缺失資訊

在提供建議前，請先在內部思考這些建議是否能幫助解決問題，但不要輸出任何思考過程。
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "根據以下內容提出改進建議。\n\n上一節點名稱: {prev_node}\n\n使用者提問: {question}\n\n輔助拆分子問題:{sub_questions}\n\n檔案內容: {documents}\n\nLLM生成回覆: {generation}",
        ),
    ]
)

reflex_router = prompt | structured_llm_router
