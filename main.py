import os
from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict, Annotated

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


# ── 2. VECTOR STORE (Knowledge Base) ─────────────────────
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

print("Loading documents...")
docs = []
for url in urls:
    docs.extend(WebBaseLoader(url).load())

splits = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100
).split_documents(docs)

vector_store = FAISS.from_documents(splits, OpenAIEmbeddings())
retriever    = vector_store.as_retriever(search_kwargs={"k": 4})
print(f"Vector store ready — {vector_store.index.ntotal} chunks indexed.")


# ── 3. LLM ────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── 4. GRAPH STATE ────────────────────────────────────────
class GraphState(TypedDict):
    question:   str
    generation: str
    documents:  List[Document]


# ── 5. PYDANTIC SCHEMAS (structured outputs) ──────────────

# 5a. Query router — which datasource to use
class RouteQuery(BaseModel):
    datasource: str = Field(
        description="Route to 'vectorstore' or 'web_search'"
    )

# 5b. Document grader — is the retrieved doc relevant?
class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Document relevance score: 'yes' or 'no'"
    )

# 5c. Hallucination grader — is the answer grounded?
class GradeHallucinations(BaseModel):
    binary_score: str = Field(
        description="Answer is grounded in facts: 'yes' or 'no'"
    )

# 5d. Answer grader — does the answer address the question?
class GradeAnswer(BaseModel):
    binary_score: str = Field(
        description="Answer addresses the question: 'yes' or 'no'"
    )


# ── 6. CHAINS ─────────────────────────────────────────────

# 6a. Query router
router_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert at routing questions. "
     "The vectorstore contains documents about agents, prompt engineering, "
     "and adversarial attacks on LLMs. "
     "Route questions on these topics to 'vectorstore'. "
     "Route all other questions to 'web_search'."),
    ("human", "{question}"),
])
question_router = router_prompt | llm.with_structured_output(RouteQuery)

# 6b. Document relevance grader
grade_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a grader assessing relevance of a retrieved document to a user question. "
     "Give a binary 'yes' or 'no' score to indicate whether the document is relevant."),
    ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
])
retrieval_grader = grade_prompt | llm.with_structured_output(GradeDocuments)

# 6c. RAG generation chain
rag_prompt = hub.pull("rlm/rag-prompt")
rag_chain   = rag_prompt | llm | StrOutputParser()

# 6d. Hallucination grader
hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a grader assessing whether an LLM generation is grounded in retrieved facts. "
     "Give a binary 'yes' or 'no'. 'Yes' means grounded, 'no' means hallucinated."),
    ("human", "Facts:\n\n{documents}\n\nGeneration: {generation}"),
])
hallucination_grader = hallucination_prompt | llm.with_structured_output(GradeHallucinations)

# 6e. Answer grader
answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a grader assessing whether an answer addresses a question. "
     "Give a binary 'yes' or 'no'."),
    ("human", "Question: {question}\n\nAnswer: {generation}"),
])
answer_grader = answer_prompt | llm.with_structured_output(GradeAnswer)

# 6f. Question rewriter
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a question re-writer that optimises a query for vectorstore retrieval. "
     "Reason about the underlying semantic intent and output only the improved question."),
    ("human", "Initial question: {question}"),
])
question_rewriter = rewrite_prompt | llm | StrOutputParser()

# 6g. Web search tool
web_search_tool = TavilySearchResults(k=3)


# ── 7. NODE FUNCTIONS ─────────────────────────────────────

def retrieve(state: GraphState) -> GraphState:
    """Retrieve documents from vector store."""
    print("--- NODE: retrieve ---")
    docs = retriever.invoke(state["question"])
    return {**state, "documents": docs}


def web_search(state: GraphState) -> GraphState:
    """Run web search and return results as documents."""
    print("--- NODE: web_search ---")
    results  = web_search_tool.invoke({"query": state["question"]})
    web_docs = [Document(page_content=r["content"]) for r in results]
    return {**state, "documents": web_docs}


def grade_documents(state: GraphState) -> GraphState:
    """
    Grade each retrieved document.
    Keep relevant ones. If any fail → flag for web search.
    """
    print("--- NODE: grade_documents ---")
    question  = state["question"]
    documents = state["documents"]
    filtered  = []
    web_needed = False

    for doc in documents:
        score = retrieval_grader.invoke({
            "question": question,
            "document": doc.page_content
        })
        if score.binary_score == "yes":
            print("  ✅ Document relevant")
            filtered.append(doc)
        else:
            print("  ❌ Document irrelevant — will trigger web search")
            web_needed = True

    return {
        **state,
        "documents":  filtered,
        "web_search": "Yes" if web_needed else "No"
    }


def generate(state: GraphState) -> GraphState:
    """Generate answer from retrieved context."""
    print("--- NODE: generate ---")
    context = "\n\n".join([doc.page_content for doc in state["documents"]])
    answer  = rag_chain.invoke({
        "context":  context,
        "question": state["question"]
    })
    return {**state, "generation": answer}


def transform_query(state: GraphState) -> GraphState:
    """Rewrite the question for better retrieval."""
    print("--- NODE: transform_query ---")
    better_q = question_rewriter.invoke({"question": state["question"]})
    print(f"  Rewritten: {better_q}")
    return {**state, "question": better_q}


# ── 8. ROUTING FUNCTIONS ──────────────────────────────────

def route_question(state: GraphState) -> str:
    """
    Query analysis — route to vectorstore or web search
    based on question complexity and topic.
    """
    print("--- ROUTER: route_question ---")
    source = question_router.invoke({"question": state["question"]}).datasource
    print(f"  Route → {source}")
    if source == "web_search":
        return "web_search"
    return "vectorstore"


def decide_to_generate(state: GraphState) -> str:
    """After grading — generate directly or rewrite query first?"""
    print("--- ROUTER: decide_to_generate ---")
    if state.get("web_search") == "Yes":
        print("  Some docs irrelevant → transform query")
        return "transform_query"
    print("  All docs relevant → generate")
    return "generate"


def grade_generation(state: GraphState) -> str:
    """
    After generation — three outcomes:
      useful       → done
      not useful   → rewrite query and retry
      not supported → hallucination → regenerate
    """
    print("--- ROUTER: grade_generation ---")

    # Check 1: is the answer grounded in the retrieved docs?
    h_score = hallucination_grader.invoke({
        "documents":  "\n\n".join([d.page_content for d in state["documents"]]),
        "generation": state["generation"]
    })

    if h_score.binary_score == "yes":
        print("  ✅ Grounded (no hallucination)")

        # Check 2: does the answer actually address the question?
        a_score = answer_grader.invoke({
            "question":   state["question"],
            "generation": state["generation"]
        })

        if a_score.binary_score == "yes":
            print("   Useful — done") 
            return "useful"
        else:
            print("   Doesn't address question → rewrite query") 
            return "not useful"

    else:
        print("   Hallucination detected → regenerate") 
        return "not supported"


# ── 9. BUILD THE GRAPH ────────────────────────────────────

builder = StateGraph(GraphState)

# Add nodes
builder.add_node("web_search",      web_search)
builder.add_node("retrieve",        retrieve)
builder.add_node("grade_documents", grade_documents)
builder.add_node("generate",        generate)
builder.add_node("transform_query", transform_query)

# Entry: route question to web_search or retrieve
builder.add_conditional_edges(
    START,
    route_question,
    {
        "web_search":  "web_search",
        "vectorstore": "retrieve",
    }
)

# Web search path: always goes to generate
builder.add_edge("web_search", "generate")

# Retrieve path: grade first
builder.add_edge("retrieve", "grade_documents")

# After grading: generate or rewrite
builder.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate":        "generate",
        "transform_query": "transform_query",
    }
)

# After rewrite: retrieve again with better query
builder.add_edge("transform_query", "retrieve")

# After generation: quality check
builder.add_conditional_edges(
    "generate",
    grade_generation,
    {
        "useful":         END,
        "not useful":     "transform_query",
        "not supported":  "generate",
    }
)

app = builder.compile()
print("\nGraph compiled successfully.\n")


# ── 10. RUN IT ────────────────────────────────────────────

def run_pipeline(question: str) -> str:
    """Run the full adaptive RAG pipeline on a question."""
    print("=" * 60)
    print(f"QUESTION: {question}")
    print("=" * 60)
    result = app.invoke({"question": question})
    print("\n" + "=" * 60)
    print("ANSWER:")
    print(result["generation"])
    print("=" * 60 + "\n")
    return result["generation"]


if __name__ == "__main__":
    # Test 1 — vectorstore question (agents blog)
    run_pipeline("What is agent memory and how does it work?")

    # Test 2 — general knowledge → web search
    run_pipeline("What is the latest version of Python?")

    # Test 3 — specific vectorstore topic
    run_pipeline("What are the types of agent memory?")

