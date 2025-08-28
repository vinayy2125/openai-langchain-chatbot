from backend.retriever import retriever
from backend.llm_client import call_llm_with_context
from typing import List
from backend.search_client import search_site
from crawler.scraper import scrape_url


def _dedupe_chunks(docs) -> List[str]:
    seen = set()
    unique = []
    for d in docs:
        text = d.page_content.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique

def _maybe_expand_queries(query: str) -> List[str]:
    # Lightweight RAG fusion: expand the query to reduce “same answer” effect
    return list(dict.fromkeys([
        query,
        f"Details about {query}",
        f"In-depth explanation of {query}",
    ]))

def build_chatbot_response(query: str, chat_history: list, site: str="ditstek.com"):
    """
    Retrieves context for the query, calls LLM, and returns chatbot response.
    `chat_history` is a list of tuples: [(role, message), ...]
    """
    # 1) Retrieve with light fusion
    variant_queries = _maybe_expand_queries(query)
    pooled_docs = []
    for q in variant_queries:
        pooled_docs.extend(retriever.get_relevant_documents(q))

    # De-duplicate chunks
    unique_texts = _dedupe_chunks(pooled_docs)

    # Construct richer context (cap to avoid over-long prompts)
    MAX_CHUNKS = 10
    context_text = "\n\n---\n\n".join([
        f"Source {i+1}:\n{chunk}"
        for i, chunk in enumerate(
        unique_texts[:MAX_CHUNKS])
    ])

    # Debug logs
    print(f"[DEBUG] Retrieved {len(pooled_docs)} docs, {len(unique_texts)} unique. "
      f"Using {min(len(unique_texts), MAX_CHUNKS)} chunks.")
    print("\n[DEBUG] Final context passed to LLM:\n", context_text[:1500],
      "\n[...]" if len(context_text) > 1500 else "")

    # 2) If FAISS gave nothing, fallback to site-specific internet search
    if not context_text.strip():
        print("[DEBUG] No context from FAISS. Falling back to internet search...")
        search_results = search_site(query, site)
        scraped_texts = []

        for res in search_results:
            url = res.get("url")
            title = res.get("title") or url
            if url:
                text = scrape_url(url)  # ✅ sync call into Playwright
                if text:
                    scraped_texts.append(f"[{title}]({url}): {text}")

        context_text = "\n\n".join(scraped_texts[:MAX_CHUNKS])

        if not context_text.strip():
            return (
                "No relevant content found. Please visit the website directly "
                f"[{site}](https://{site}).",
                True
            )

    # 3) Format history
    history_text = "\n".join(
        f"{'User' if role == 'user' else 'Assistant'}: {msg}"
        for role, msg in chat_history
    )

    answer = call_llm_with_context(
    context=context_text,
    history=history_text,
    question=query,
    detail_level="high"  # Always request detailed responses
    )

    # 5) Fallback phrasing: relax strict check
    # Only fallback if the answer is *completely empty*
    if not answer.strip():
        return (
            "No relevant content found. Please submit your query via our "
            "[Contact Form](https://www.ditstek.com/contact).",
            True
        )

    return answer, True
