from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.models.model import llm_model


llm = llm_model

system = """
你是一個嚴謹且可驗證的回答者。根據提供的 documents 與改進建議 (reflex_prompt)，
生成對使用者問題的準確、可驗證回答。

請遵循以下原則：

1. 優先使用文件(documents)中的事實作為依據，次要使用對話歷史中的紀錄。
2. 若 reflex_prompt 不為空，請參考其指示調整回答策略；若 reflex_prompt 為空則忽略此欄。
3. 若文件不足以支持確切答案，請在 answer 欄位寫明「資料不足」，並說明需要補充的資訊方向。
4. 回覆應條理分明，必要時可引用文件中的關鍵句。
5. 若有多個子問題(sub_questions)，請整合它們的重點以形成完整回答。
6. 若使用者的提問屬於日常問候，可以簡單且禮貌的回覆。

"""

llm = llm_model

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "根據以下內容回答問題。\n\n改進建議: {reflex_prompt}\n\n使用者提問: {question}\n\n輔助拆分子問題:{sub_questions}\n\n檔案內容: {documents}\n\n對話歷史: {chat_history}",
        ),
    ]
)

generation_chain = prompt | llm | StrOutputParser()
