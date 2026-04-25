# retriever.py
from ingest import load_vectorstore
from config import TOP_K_RESULTS, CONFIDENCE_THRESHOLD

def retrieve_context(query: str, k: int = TOP_K_RESULTS):
    """
    Retrieve top-k relevant chunks from ChromaDB using similarity search.
    Uses relevance scores (not just doc count) to determine confidence.
    Returns (context_text, confidence_level)
    """
    vectorstore = load_vectorstore()

    # similarity_search_with_relevance_scores returns (doc, score) pairs
    # Score is 0 to 1 — higher means MORE relevant to the query
    results = vectorstore.similarity_search_with_relevance_scores(
        query, k=k
    )

    if not results:
        return "No relevant information found in the knowledge base.", "low"

    # Separate docs and scores
    docs = [doc for doc, score in results]
    scores = [score for doc, score in results]

    # Calculate average relevance score across retrieved chunks
    avg_score = sum(scores) / len(scores)

    # Build context — include source page for traceability
    context_parts = []
    for doc, score in results:
        page = doc.metadata.get("source_page", "unknown")
        context_parts.append(
            f"[Page {page} | Relevance: {score:.2f}]\n{doc.page_content}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Confidence based on actual relevance score, not just doc count
    # CONFIDENCE_THRESHOLD set in config.py (default 0.3)
    confidence = "high" if avg_score >= CONFIDENCE_THRESHOLD else "low"

    print(f"🔍 Retrieved {len(docs)} chunks | Avg score: {avg_score:.2f} | Confidence: {confidence}")

    return context, confidence