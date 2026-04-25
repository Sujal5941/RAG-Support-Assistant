# 🍔 QuickBite Food Delivery — RAG Customer Support Assistant

A production-style AI customer support assistant built using **Retrieval-Augmented Generation (RAG)**, **LangGraph**, and **Human-in-the-Loop (HITL)** escalation. The system answers customer queries by retrieving relevant information from a PDF knowledge base and routing complex queries to human agents.

---

## 📌 Project Overview

This project demonstrates a complete RAG pipeline for a food delivery platform (QuickBite). Instead of relying on a generic LLM that might hallucinate policies, the assistant retrieves answers **only from the official knowledge base PDF** — making it accurate, grounded, and controllable.

---

## 🏗️ System Architecture

```
PDF Knowledge Base
       │
       ▼
  ingest.py          ← Load, chunk, embed, store in ChromaDB
       │
       ▼
  ChromaDB           ← Vector store (persisted on disk)
       │
       ▼
  retriever.py       ← Similarity search, confidence scoring
       │
       ▼
  graph.py           ← LangGraph workflow (retrieve → route → generate/escalate)
       │
    ┌──┴──┐
    ▼     ▼
generate  escalate
  (LLM)  (hitl.py)
    │     │
    └──┬──┘
       ▼
    app.py            ← Streamlit UI
```

---

## 📁 Folder Structure

```
project/
│
├── data/
│   └── knowledge_base.pdf      # QuickBite support knowledge base
│
├── chroma_db/                  # Auto-created after ingestion
│
├── app.py                      # Streamlit frontend
├── graph.py                    # LangGraph workflow nodes and routing
├── retriever.py                # ChromaDB similarity search
├── ingest.py                   # PDF loading, chunking, embedding
├── hitl.py                     # Human-in-the-Loop escalation logic
├── config.py                   # Central configuration (chunk size, keys, etc.)
│
├── escalation_log.json         # Auto-created — logs escalated queries
├── .env                        # API keys (never commit this)
├── .gitignore
└── requirements.txt
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq API — LLaMA 3.3 70B |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (free, local) |
| Vector Store | ChromaDB (persisted locally) |
| Workflow Engine | LangGraph (StateGraph) |
| PDF Processing | LangChain PyPDFLoader |
| Frontend | Streamlit |
| HITL Logging | JSON file-based escalation log |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/quickbite-rag-support.git
cd quickbite-rag-support
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your `.env` file

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Add the knowledge base PDF

Place your PDF inside the `data/` folder:

```
data/knowledge_base.pdf
```

### 6. Ingest the PDF into ChromaDB

```bash
python ingest.py
```

Expected output:
```
📄 Loading QuickBite knowledge base PDF...
✅ Loaded 6 pages from PDF
✂️  Split into 38 chunks (size=600, overlap=80)
🔢 Generating embeddings...
💾 Storing embeddings in ChromaDB...
✅ ChromaDB ready — 38 chunks stored in collection 'shopease_kb'
```

### 7. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🔁 How the RAG Pipeline Works

1. **Ingestion** — `ingest.py` loads the PDF, splits it into 600-character chunks with 80-character overlap, generates embeddings using `all-MiniLM-L6-v2`, and stores them in ChromaDB.

2. **Retrieval** — When a user asks a question, `retriever.py` converts the query into an embedding and finds the top 3 most similar chunks using cosine similarity. It also computes an average relevance score to determine confidence.

3. **Routing** — `graph.py` uses LangGraph to decide the next step:
   - If a sensitive keyword is detected (fraud, legal, complaint, etc.) → **escalate**
   - If retrieval confidence is low → **escalate**
   - Otherwise → **generate answer with LLM**

4. **Generation** — The LLM (LLaMA 3.3 70B via Groq) receives the retrieved context and the user question, and generates a grounded answer.

5. **HITL** — Escalated queries are logged to `escalation_log.json`. Human agents can view and resolve them directly from the Streamlit sidebar.

---

## 🔀 LangGraph Workflow

```
[START]
   │
   ▼
retrieve_node          ← Fetches top-k chunks from ChromaDB
   │
   ▼
route_query()          ← Conditional router
   │
   ├── keyword match or low confidence ──▶ escalate_node ──▶ [END]
   │
   └── high confidence ──────────────────▶ generate_node ──▶ [END]
```

**GraphState fields:**

| Field | Type | Purpose |
|---|---|---|
| `query` | str | User's question |
| `context` | str | Retrieved chunks from ChromaDB |
| `answer` | str | Final answer shown to user |
| `confidence` | str | "high" or "low" based on retrieval score |
| `needs_escalation` | bool | Whether query was escalated |
| `escalation_reason` | str | Why it was escalated |
| `human_response` | str | Human agent's reply (if resolved) |

---

## 🧑‍💼 Human-in-the-Loop (HITL)

When a query is escalated, it is logged to `escalation_log.json` with:
- Timestamp
- Original query
- Escalation reason (keyword match or low confidence)
- Status (pending / resolved)

Human agents see pending escalations in the Streamlit sidebar, can type a reply, and mark them as resolved. Resolved entries are timestamped.

**Escalation triggers:**
- Keywords: `fraud`, `legal`, `lawsuit`, `complaint`, `chargeback`, `wrong item`, `lost package`, `speak to human`, `manager`, `supervisor`, `police`
- Low retrieval confidence (average similarity score below threshold)

---

## 🧪 Sample Test Queries

| Query | Expected Behaviour |
|---|---|
| `What are the delivery charges?` | AI answers from knowledge base |
| `How do I cancel my order?` | AI answers from knowledge base |
| `What is QuickBite Pro membership?` | AI answers from knowledge base |
| `How long does a refund take?` | AI answers from knowledge base |
| `I want to file a fraud complaint` | Escalated to human agent |
| `I need to speak to a manager` | Escalated to human agent |
| `What is the weather today?` | Low confidence → escalated |

---

## ⚙️ Configuration

All settings are centralized in `config.py`:

| Setting | Value | Purpose |
|---|---|---|
| `CHUNK_SIZE` | 600 | Characters per chunk |
| `CHUNK_OVERLAP` | 80 | Overlap between chunks |
| `TOP_K_RESULTS` | 3 | Chunks retrieved per query |
| `CONFIDENCE_THRESHOLD` | 0.3 | Min relevance score for high confidence |
| `GROQ_MODEL` | llama-3.3-70b-versatile | LLM model |

---

## 📦 Requirements

```
streamlit
langchain
langchain-community
langchain-chroma
langchain-groq
langchain-text-splitters
langgraph
sentence-transformers
chromadb
pypdf
python-dotenv
```

---

## 🔒 Security Notes

- Never commit your `.env` file — it contains your API key
- The `.gitignore` excludes `.env` and `chroma_db/`
- QuickBite will never ask for OTP or passwords — same principle applies here

---

## 🔮 Future Enhancements

- **Multi-document support** — ingest multiple PDFs into separate collections
- **Feedback loop** — let users rate answers to improve retrieval
- **Memory integration** — remember conversation history across turns
- **Authentication** — agent login for the HITL dashboard
- **Cloud deployment** — deploy on Streamlit Cloud or AWS

---

## 👨‍💻 Author

Built as part of the RAG Internship Project.
Designed and implemented following production system design principles.

---

*QuickBite Customer Support Assistant — RAG + LangGraph + HITL — 2024*
