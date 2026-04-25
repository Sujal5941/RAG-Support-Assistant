# graph.py
import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from retriever import retrieve_context
from config import GROQ_API_KEY, GROQ_MODEL, ESCALATION_KEYWORDS

load_dotenv()

# ─── State Object ───────────────────────────────────────────────
# This is the "baton" passed between every node in the graph
class GraphState(TypedDict):
    query: str
    context: str
    answer: str
    confidence: str          # "high" | "low" — set by retriever
    needs_escalation: bool
    escalation_reason: str   # NEW: why it was escalated
    human_response: str

# ─── LLM Setup (once at module level) ───────────────────────────
# Use config values — change model in config.py, updates everywhere
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL
)

# ─── Node 1: Retrieve Context ────────────────────────────────────
def retrieve_node(state: GraphState) -> GraphState:
    """Fetches relevant chunks from ChromaDB for the user query"""
    query = state["query"]
    context, confidence = retrieve_context(query)
    return {
        **state,
        "context": context,
        "confidence": confidence
    }

# ─── Node 2: Generate Answer ─────────────────────────────────────
def generate_node(state: GraphState) -> GraphState:
    """Sends retrieved context + query to LLM and gets an answer"""
    query = state["query"]
    context = state["context"]

    # Improved prompt — explicit rules, ShopEase branding, format instructions
    prompt = f"""You are a ShopEase customer support assistant.
ShopEase is an online retail platform serving customers across India, UAE and Southeast Asia.

Rules you must follow:
- Answer ONLY using the context provided below
- Never make up policies, prices or timelines not in the context
- If the context does not contain the answer, say: "I don't have enough information on this. Please contact support at support@shopease.com"
- Keep your answer concise, friendly and well structured
- Use bullet points for multi-step answers

Context from knowledge base:
{context}

Customer question: {query}

Your answer:"""

    response = llm.invoke(prompt)
    answer = response.content

    return {
        **state,
        "answer": answer,
        "needs_escalation": False,
        "escalation_reason": ""
    }

# ─── Node 3: Escalate to Human ───────────────────────────────────
def escalate_node(state: GraphState) -> GraphState:
    """Flags query for human agent — stores reason for HITL logs"""
    return {
        **state,
        "answer": (
            "Thank you for reaching out to ShopEase support. "
            "Your query has been forwarded to a human agent who will "
            "respond shortly. For urgent help call 1800-123-4567."
        ),
        "needs_escalation": True
    }

# ─── Conditional Router ──────────────────────────────────────────
def route_query(state: GraphState) -> Literal["generate", "escalate"]:
    """
    Decides whether to generate an AI answer or escalate to human.
    Two triggers: keyword match OR low retrieval confidence.
    """
    query = state["query"].lower()
    confidence = state["confidence"]

    # Trigger 1: sensitive keywords (imported from config.py)
    matched = [kw for kw in ESCALATION_KEYWORDS if kw in query]
    if matched:
        # Store reason before routing — update state via a workaround
        state["escalation_reason"] = f"Keyword match: {', '.join(matched)}"
        return "escalate"

    # Trigger 2: retriever found no confident matches
    if confidence == "low":
        state["escalation_reason"] = "Low retrieval confidence"
        return "escalate"

    return "generate"

# ─── Build and Compile Graph (once) ──────────────────────────────
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("escalate", escalate_node)

    graph.set_entry_point("retrieve")

    # After retrieval, route_query decides the next node
    graph.add_conditional_edges(
        "retrieve",
        route_query,
        {
            "generate": "generate",
            "escalate": "escalate"
        }
    )

    graph.add_edge("generate", END)
    graph.add_edge("escalate", END)

    return graph.compile()

# Compile once when module loads — not on every query
compiled_graph = build_graph()

# ─── Public entry point ───────────────────────────────────────────
def run_graph(query: str, human_response: str = ""):
    """Run the compiled RAG graph for a given user query"""
    initial_state = GraphState(
        query=query,
        context="",
        answer="",
        confidence="high",
        needs_escalation=False,
        escalation_reason="",
        human_response=human_response
    )
    result = compiled_graph.invoke(initial_state)
    return result