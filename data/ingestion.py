import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from src.models.model import embed_model
from bs4 import BeautifulSoup

load_dotenv()


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def create_vectorstore():
    faiss_path = "./faiss.index"
    if os.path.exists(faiss_path):
        print("--正在載入向量資料庫--")
        vectorstore = FAISS.load_local(
            faiss_path,
            embed_model,
            allow_dangerous_deserialization=True,
        )
        return vectorstore.as_retriever()
    print("--創建新向量資料庫--")
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    docs_list = []
    for url in urls:
        raw_docs = WebBaseLoader(url).load()
        for doc in raw_docs:
            doc.page_content = clean_html(doc.page_content)  # 清理 HTML
            docs_list.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=True,
    )
    doc_splits = text_splitter.split_documents(docs_list)
    doc_splits = [doc for doc in doc_splits if len(doc.page_content.strip()) > 50]

    vectorstore = FAISS.from_documents(doc_splits, embed_model)

    vectorstore.save_local(faiss_path)

    return vectorstore.as_retriever()


retriever = create_vectorstore()
