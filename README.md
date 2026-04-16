# Adaptive Multi-Source RAG Pipeline with Self-Correction

A dynamic, agentic Retrieval-Augmented Generation (RAG) system that routes queries between a vector database and web search, applies iterative retrieval refinement, and uses LLM-based evaluation to ensure grounded and high-quality responses.

Built using **LangGraph, LangChain, FAISS, OpenAI GPT-4o-mini, and Tavily Search API**.

---

## What this project does

This system goes beyond standard RAG. It behaves like an adaptive reasoning pipeline that can:

- Route queries between **vector store (FAISS)** and **web search (Tavily)**
- Retrieve and filter relevant documents using LLM-based grading
- Generate answers grounded in retrieved context
- Detect hallucinations in generated responses
- Evaluate answer quality against the original question
- Automatically rewrite queries when retrieval quality is poor
- Iterate until a useful response is produced

In short: it doesn’t just retrieve and answer — it self-corrects when it fails.


---

## Key Features

### Adaptive Query Routing
Automatically decides whether a query should be answered using:
- Internal knowledge base (FAISS)
- External web search (Tavily)

### Multi-Source Retrieval
Combines structured vector search with real-time web search for broader coverage.

### Document Relevance Filtering
Each retrieved document is graded by an LLM to remove irrelevant context before generation.

### Self-Correction Loop
If retrieved context is weak or incomplete:
- Query is rewritten
- Retrieval is repeated
- System iterates until quality improves

### Hallucination Detection
Generated answers are validated against retrieved documents to ensure factual grounding.

### Answer Quality Evaluation
Checks whether the final response actually addresses the user’s question.

---

## Tech Stack

- Python
- LangGraph (agentic workflows)
- LangChain
- OpenAI GPT-4o-mini
- FAISS (vector database)
- Tavily Search API
- Pydantic (structured outputs)

---

## How it works

1. User submits a question  
2. LLM router decides best data source  
3. Relevant documents are retrieved  
4. Documents are filtered for relevance  
5. Answer is generated using RAG pipeline  
6. Answer is checked for hallucination  
7. Answer is evaluated for relevance  
8. If quality is poor → query is rewritten and retried  

---

## Example Queries

```python
"What is agent memory and how does it work?"
"What are the latest updates in Python?"
"What is retrieval augmented generation?"
