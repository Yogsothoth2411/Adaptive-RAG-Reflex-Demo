# Enhanced Adaptive RAG

> A reflective and state-managed RAG pipeline built on LangGraph, extending the adaptive RAG architecture with multi-turn reasoning, query reflection, and conversation state tracking.


## Overview

This project is an **extended implementation** of the Adaptive-RAG system built with **LangGraph** and **LangChain**.
It enhances the original adaptive RAG architecture by introducing **query decomposition**, **reflective reasoning**, and **conversation state management**, allowing more controlled multi-turn interactions.

The system introduces controlled reflective loops and state management to prevent infinite recursion while maintaining adaptivity.


## Features & Improvements

Compared to the referenced Adaptive-RAG examples, this project adds the following new components and improvements:

| # | Feature / Node            | Description                                                                                    |
| - | ------------------------- | -----------------------------------------------------------------------------------------------|
| 1 | **`QUERY_SPLITTER_NODE`** | Splits complex user queries into sub-queries for better retrieval precision.                   |
| 2 | **`REFLECTION_NODE`**     | Adds reflection logic to detect high hallucination risk or missing web retrieval suggestions.  |
| 3 | **`search_prompt` Chain** | Generates web-search prompts based on reflection results for adaptive external retrieval.      |
| 4 | **`CHAT_MANAGE_NODE`**    | Stores each round’s conversation history (use InMemorySaver) for state continuity and analysis.|
| 5 | **Loop Control**          | Adds reflection loop upper limit to prevent infinite reflection cycles.                        |
| 6 | **Extended Tests**        | Integrates new unit tests for custom nodes and reflection logic.                               |
| 7 | **Extended Graph State**  | Expands state definition to support additional node data flow.                                 |


## Architecture Overview

The enhanced workflow introduces reflection and memory nodes to the original adaptive RAG loop:

![static\graph.png](static\graph.png)

Key design philosophy:

* Introducing controlled reflective feedback.
* Preserve **stateful chat history** using `InMemorySaver`.
* Enable dynamic reasoning and reflection under loop constraints.


## Testing

Extended tests have been added to validate:

* Query splitting and reconstruction logic
* Reflection decision flow


## References

This project was developed with reference to the following repositories:

1. [LangGraph Adaptive RAG Example](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_adaptive_rag.ipynb) — MIT License
2. [LangGraph-AI Adaptive RAG by Piyush Agnihotri](https://github.com/piyushagni5/langgraph-ai/blob/main/agentic-rag/agentic-rag-systems/building-adaptive-rag/README.md) — MIT License

### Acknowledgements from Reference (2)

> * Original LangChain repository: [LangChain Cookbook](https://github.com/mistralai/cookbook/tree/main/third_party/langchain)
> * By Sophia Young (Mistral) & Lance Martin (LangChain)
> * Built with LangGraph
> * Marco’s refactored repository: [emarco177/langgraph-course](https://github.com/emarco177/langgraph-course/tree/project/agentic-rag)

