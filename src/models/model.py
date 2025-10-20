from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import trim_messages


load_dotenv()

trimer = trim_messages(
    # messages=prompt.messages,# 這裡不需要傳入messages，會自動從上游節點取得，如果傳入，會傳出一個訊息裁剪過的list，且無法達成動態處理。
    # 當len作為令牌計數器函數傳入時，# max_tokens 將計算聊天歷史記錄中的消息數。
    max_tokens=10,  # 總共的最大token數量，包含系統、使用者、助手
    token_counter=len,  # 計數函數
    strategy="last",  # 刪除策略，last保留最新，first保留最前（最舊訊息）
    include_system=False,  # token計算是否包含系統訊息，False時也不會刪除，僅不算入token，如果是True算入token，則在特定情況可能刪除，當策略為last時，預設為False
    allow_partial=False,  # 是否允許部分訊息，False表示要麼保留整個訊息，要麼刪除整個訊息
)

llm_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
