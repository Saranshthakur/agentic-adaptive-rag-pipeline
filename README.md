Adaptive Multi-Source RAG Pipeline with Self-Correction

A dynamic, agentic Retrieval-Augmented Generation (RAG) system that routes queries between a vector database and web search, applies iterative retrieval refinement, and uses LLM-based evaluation to ensure grounded, high-quality answers.

Built using LangGraph, FAISS, OpenAI GPT-4o-mini, and Tavily Search.

What this project does

This system goes beyond basic RAG. It behaves like a small reasoning pipeline that can:

Decide whether to use vector database or web search
Retrieve relevant documents from multiple sources
Filter irrelevant documents before generation
Detect hallucinations in generated answers
Rewrite queries automatically when retrieval quality is poor
Iterate until a high-quality answer is produced

Architecture
