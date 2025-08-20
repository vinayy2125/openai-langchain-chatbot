import os
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage
from dotenv import load_dotenv
from googlesearch import search  # Import the Google search library

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("API_KEY not found in .env file")

# LLM setup
llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)

# Prompt template
prompt_template = PromptTemplate.from_template("""
You are a helpful assistant.

Conversation so far:
{history}

Relevant context from the knowledge base:
{context}

User's latest question:
{question}

Rules:
- Only use the provided context and conversation history.
- If the answer is not in the context, respond exactly:
  "No relevant content found. Please visit the website directly."
- Do not use outside knowledge.

Format your final answer with **bold** for key points and use structured formatting where helpful.
""")

def perform_google_search(query: str, num_results: int = 5) -> str:
    """
    Perform a Google search and return the top results as a formatted string.
    """
    try:
        results = search(query, num_results=num_results)
        formatted_results = "\n".join([f"{i+1}. {result}" for i, result in enumerate(results)])
        return formatted_results
    except Exception as e:
        return f"Error performing search: {e}"

def call_llm_with_context(context: str, history: str, question: str) -> str:
    """
    Calls the LLM with formatted prompt and returns the response content.
    """
    prompt = prompt_template.format(
        history=history,
        context=context,
        question=question
    )

    raw_answer = llm.invoke(prompt)
    return raw_answer.content if isinstance(raw_answer, AIMessage) else str(raw_answer)


def call_llm_with_retrieval_and_search(query: str, history: str, retriever, confidence_threshold: float = 0.7) -> str:
    """
    Implements LangGraph-style branching logic for retrieval and search.

    Args:
        query (str): User's query.
        history (str): Conversation history.
        retriever: FAISS retriever instance.
        confidence_threshold (float): Threshold for retrieval confidence.

    Returns:
        str: Final response from the LLM.
    """
    # Step 1: Retrieve from local context DB
    retrieved_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Debug: Print retrieved context
    print(f"[DEBUG] Retrieved context:\n{context}")

    # Step 2: Check retrieval confidence
    if not context.strip() or len(context.split()) < confidence_threshold * 100:  # Use confidence_threshold to evaluate sufficiency
        print("[DEBUG] Low confidence or no results. Performing Google search...")
        search_results = perform_google_search(query, num_results=5)
        print(f"[DEBUG] Google search results:\n{search_results}")  # Debug search results
        if context.strip():
            context = f"Extracted Knowledge Base Context:\n{context}\n\nAdditional search results:\n{search_results}"
        else:
            context = f"Additional search results:\n{search_results}"
        response_source = "from web"
    else:
        print("[DEBUG] High-confidence retrieval. Using local context.")
        response_source = "from knowledge base"

    # Debug: Print the final context being sent
    print(f"[DEBUG] Final context for LLM:\n{context}")

    # Step 3: Pass to LLM
    final_answer = call_llm_with_context(context, history, query)

    # Step 4: Mark response source
    if response_source == "from web" and "No relevant content found" in final_answer:
        final_answer += f"\n\n[Response {response_source}]"
    else:
        final_answer += f"\n\n[Response prioritized from knowledge base]"

    return final_answer


# Run the test function
if __name__ == "__main__":
    test_google_search()
