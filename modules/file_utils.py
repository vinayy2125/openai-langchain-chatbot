import PyPDF2
import docx
import streamlit as st


def extract_text_from_file(uploaded_file):
    """
    Extract text from an uploaded file (PDF, DOCX, or TXT).
    Returns the extracted text as a string.
    Shows Streamlit error/warning messages if extraction fails or file type is unsupported.
    """
    extracted_text = ""
    if uploaded_file.type == "application/pdf":
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                extracted_text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    elif (
        uploaded_file.type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        try:
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                extracted_text += para.text + "\n"
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
    elif uploaded_file.type == "text/plain":
        try:
            extracted_text = uploaded_file.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading TXT: {e}")
    else:
        st.warning("Unsupported file type.")
    return extracted_text


def chunk_text(text):
    """
    Split the input text into non-empty paragraphs (chunks) by newline.
    Returns a list of string chunks.
    """
    return [chunk.strip() for chunk in text.split("\n") if chunk.strip()]
