import os
import re
import openai
from modules.db_models import ChatMessage
from modules.file_utils import extract_text_from_file, chunk_text
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def handle_pdf_chat(st, db, session_id, user_input):
    uploaded_file = st.file_uploader(
        "Upload a file (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"], key="pdf_chat_uploader"
    )

    if uploaded_file is not None and "pdf_extracted_text" not in st.session_state:
        with st.spinner("Extracting and indexing document..."):
            try:
                extracted_text = extract_text_from_file(uploaded_file)
                chunks = chunk_text(extracted_text)
                st.session_state["pdf_extracted_text"] = extracted_text
                st.session_state["pdf_chunks"] = chunks

                # Build FAISS index from chunks
                embeddings = OpenAIEmbeddings()
                vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                st.session_state["vector_store"] = vector_store

            except Exception as e:
                st.error(f"An error occurred during file processing: {e}")
                return

    extracted_text = st.session_state.get("pdf_extracted_text", "")
    vector_store = st.session_state.get("vector_store")

    if extracted_text:
        st.subheader("📄 Extracted Document Content")
        st.code(extracted_text[:3000] + ("..." if len(extracted_text) > 3000 else ""))
        st.info("✅ Document uploaded and indexed. You can now ask questions below.")
        st.markdown("---")
        st.subheader("💬 Active Chat on Uploaded Document")


        if "pdf_chat_history" not in st.session_state:
            st.session_state.pdf_chat_history = []

        if st.button("🧹 Clear Chat History"):
            st.session_state.pdf_chat_history.clear()
            db.query(ChatMessage).filter_by(session_id=session_id).delete()
            db.commit()
            st.session_state.pop("pdf_extracted_text", None)
            st.session_state.pop("pdf_chunks", None)
            st.session_state.pop("vector_store", None)

        if user_input:
            with st.spinner("Searching the document..."):
                keyword_matches = [
                    line for line in extracted_text.split("\n")
                    if user_input.lower() in line.lower()
                ]

                if keyword_matches:
                    def highlight_keywords(text, keyword):
                        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                        return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)

                    reply = "<b>Here are the matches I found in the document:</b><br><br>"
                    reply += "<br>".join(f"• {highlight_keywords(line, user_input)}" for line in keyword_matches[:5])
                    if len(keyword_matches) > 5:
                        reply += f"<br>...and {len(keyword_matches) - 5} more."
                elif vector_store:
                    # Semantic search using FAISS vector store
                    docs = vector_store.similarity_search(user_input, k=3)
                    context = "\n\n".join([doc.page_content for doc in docs])

                    full_prompt = (
                        f"The user uploaded the following document content:\n\n{context}\n\n"
                        f"The user now asks: {user_input}\n\nRespond based on the document above. If not found, use general knowledge."
                    )

                    try:
                        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                        response = client.chat.completions.create(
                        model="gpt-3.5-turbo",messages=[
                                {"role": "system", "content": "You are a helpful assistant that answers based on uploaded document content if available, otherwise based on general knowledge."},
                                {"role": "user", "content": full_prompt},
                            ],)
                        reply = response.choices[0].message.content
                    except Exception as e:
                        reply = f"Error generating response: {e}"
                else:
                    reply = "No indexed data available to search."

                st.session_state.pdf_chat_history.append({
                    "user": user_input,
                    "bot": reply
                })

                new_msg_user = ChatMessage(session_id=session_id, sender="You", content=user_input)
                new_msg_bot = ChatMessage(session_id=session_id, sender="OPENAI", content=reply)
                db.add_all([new_msg_user, new_msg_bot])
                db.commit()

        for msg in st.session_state.pdf_chat_history:
            st.markdown(
                f"<div style='text-align:right; background:#e6f7ff; padding:10px; border-radius:8px; margin:6px 0;'>🧑 {msg['user']}</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='text-align:left; background:#f6f6f6; padding:10px; border-radius:8px; margin:6px 0;'>🤖 {msg['bot']}</div>",
                unsafe_allow_html=True
            )

    elif uploaded_file is not None:
        st.warning("⚠️ No text could be extracted from the uploaded file. Please check the file format and content.")
