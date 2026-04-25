# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ChromaDB Configuration
CHROMA_PATH = "chroma_db"
DATA_PATH = "data/knowledge_base.pdf"

# Chunking Configuration
# Larger chunk size keeps support policy paragraphs intact
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80

# Retrieval Configuration
# 3 precise results reduces noise in the LLM prompt
TOP_K_RESULTS = 3

# Confidence threshold — ChromaDB distance score above this = low confidence
CONFIDENCE_THRESHOLD = 0.3   

# Escalation Keywords — ShopEase specific triggers
ESCALATION_KEYWORDS = [
    "refund", "legal", "lawsuit", "urgent", "angry",
    "complaint", "fraud", "cancel account", "speak to human",
    "manager", "supervisor", "chargeback",
    "police", "wrong item", "lost package"
]