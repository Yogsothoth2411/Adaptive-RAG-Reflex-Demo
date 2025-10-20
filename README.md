# For English version, see [README_en.md](README_en.md)

# Enhanced Adaptive RAG

> 基於 LangGraph 的反思型與狀態管理 RAG 管線，擴展了Adaptive-RAG 架構，增加了多輪推理、查詢反思與對話狀態追蹤功能。


## 概述

本專案是基於 **LangGraph** 與 **LangChain** 所建的 **Adaptive-RAG 系統延伸實現**。
它透過引入 **查詢拆分**、**反思推理** 以及 **對話狀態管理**，增強了原始自適應 RAG 架構，使多輪互動更可控且可解釋。

系統引入了受控的反思迴圈與狀態管理機制，避免無限遞迴，同時保持自適應能力。


## 功能與改進

相比原始的 Adaptive-RAG 範例，本專案新增了以下功能與改進：

| # | 功能 / 節點                          | 說明                          |
| - | -------------------------------- | --------------------------- |
| 1 | **`QUERY_SPLITTER_NODE`**        | 將複雜用戶查詢拆分為子查詢，以提升檢索精度。      |
| 2 | **`REFLECTION_NODE`**            | 增加反思邏輯，用於提供高幻覺風險或缺失的網路檢索建議。 |
| 3 | **`search_prompt` Chain**        | 根據反思結果生成網路搜尋提示，以進行自適應外部檢索。  |
| 4 | **`CHAT_MANAGE_NODE`**           | 儲存每輪對話歷史（使用 InMemorySaver），確保狀態連續性與分析能力。 |
| 5 | **迴圈控制 (Loop Control)**          | 設置反思迴圈上限，以防止無限反思循環。         |
| 6 | **擴展測試 (Extended Tests)**        | 新增自定義節點與反思邏輯的單元測試。          |
| 7 | **擴展圖狀態 (Extended Graph State)** | 擴展狀態定義，以支援額外的節點資料流。         |


## 架構概覽

強化後的工作流程在原始自適應 RAG 迴圈中加入了反思與記憶節點：

![static\Graph_2.png](static\Graph_2.png)

主要設計理念：

* 引入受控的反思反饋機制
* 使用 `InMemorySaver` 保存 **狀態化對話歷史**
* 在迴圈約束下實現動態推理與反思


## 測試

新增的擴展測試可驗證：

* 查詢拆分與重構邏輯
* 反思決策流程


## 參考資料

本專案參考以下開源倉庫開發：

1. [LangGraph Adaptive RAG Example](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_adaptive_rag.ipynb) — MIT 授權
2. [LangGraph-AI Adaptive RAG by Piyush Agnihotri](https://github.com/piyushagni5/langgraph-ai/blob/main/agentic-rag/agentic-rag-systems/building-adaptive-rag/README.md) — MIT 授權

### 參考來源 (2) 的引用及致謝

> * 原始 LangChain 倉庫：[LangChain Cookbook](https://github.com/mistralai/cookbook/tree/main/third_party/langchain)
> * 作者：Sophia Young (Mistral) & Lance Martin (LangChain)
> * 基於 LangGraph 構建
> * Marco 重構的倉庫：[emarco177/langgraph-course](https://github.com/emarco177/langgraph-course/tree/project/agentic-rag)
