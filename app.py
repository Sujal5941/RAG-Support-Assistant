# app.py
import streamlit as st
import os
from graph import run_graph
from hitl import (
    log_escalation,
    get_pending_escalations,
    resolve_escalation,
    get_escalation_stats
)
from ingest import ingest_pdf
from config import CHROMA_PATH

# ─── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopEase Support Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("QuickBit Food Delivery Support Assistant")
st.caption("Powered by Groq LLM · ChromaDB · LangGraph · HITL")

# ─── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    # PDF Upload and Ingestion
    uploaded_file = st.file_uploader("Upload Knowledge Base PDF", type="pdf")
    if uploaded_file:
        os.makedirs("data", exist_ok=True)
        with open("data/knowledge_base.pdf", "wb") as f:
            f.write(uploaded_file.read())
        if st.button("📥 Ingest PDF into ChromaDB"):
            with st.spinner("Processing PDF — this may take a minute..."):
                ingest_pdf()
            st.success("✅ PDF ingested! You can now ask questions.")

    # Warn if ChromaDB not ready
    if not os.path.exists(CHROMA_PATH):
        st.warning("⚠️ No knowledge base found. Please upload and ingest a PDF first.")

    st.divider()

    # Escalation stats summary
    stats = get_escalation_stats()
    st.header("📋 Escalations")
    col1, col2 = st.columns(2)
    col1.metric("Pending", stats["pending"])
    col2.metric("Resolved", stats["resolved"])

    st.divider()

    # Pending escalations for human agents to resolve
    escalations = get_pending_escalations()
    if escalations:
        for esc in escalations:
            with st.expander(f"❗ {esc['query'][:40]}..."):
                st.write(f"**Query:** {esc['query']}")
                st.write(f"**Reason:** {esc.get('reason', 'Not specified')}")
                st.write(f"**Time:** {esc['timestamp']}")
                human_reply = st.text_area(
                    "Your Reply:", key=esc['timestamp']
                )
                if st.button("✅ Resolve", key=f"resolve_{esc['timestamp']}"):
                    # Use return value — only show success if it actually worked
                    success = resolve_escalation(esc['timestamp'], human_reply)
                    if success:
                        st.success("✅ Resolved and saved!")
                        st.rerun()
                    else:
                        st.error("❌ Could not find this escalation entry.")
    else:
        st.info("No pending escalations")

    st.divider()
    # Clear chat history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ─── Chat Interface ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("escalated"):
            st.warning("🔺 This query was escalated to a human agent")
        if msg.get("confidence"):
            label = "🟢 High confidence" if msg["confidence"] == "high" else "🟡 Low confidence"
            st.caption(label)
        if msg.get("context"):
            with st.expander("📄 Retrieved Context"):
                st.text(msg["context"])

# Block chat if ChromaDB not ready
if not os.path.exists(CHROMA_PATH):
    st.info("👆 Please upload and ingest a PDF using the sidebar before asking questions.")
    st.stop()

# Chat Input
if prompt := st.chat_input("Ask a QuickBite support question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            result = run_graph(prompt)

        answer = result["answer"]
        escalated = result["needs_escalation"]
        context = result.get("context", "")
        confidence = result.get("confidence", "high")
        escalation_reason = result.get("escalation_reason", "")

        st.markdown(answer)

        # Show confidence level on every answer
        if not escalated:
            label = "🟢 High confidence" if confidence == "high" else "🟡 Low confidence"
            st.caption(label)

        if escalated:
            st.warning("🔺 Escalated to human agent")
            # Log with the actual reason from graph state — not a hardcoded string
            log_escalation(prompt, escalation_reason or "Escalation triggered")

        with st.expander("📄 Retrieved Context"):
            st.text(context if context else "No context retrieved")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "escalated": escalated,
        "confidence": confidence,
        "context": context
    })