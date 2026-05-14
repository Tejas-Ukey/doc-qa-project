import streamlit as st

import config
from src.loader import load_pdfs
from src.splitter import split_documents
from src.embeddings import get_embeddings
from src.vectorstore import create_vectorstore, load_vectorstore
from src.rag_chain import get_rag_chain, format_docs


def process_pdfs(pdf_files):
    documents = load_pdfs(pdf_files)
    chunks = split_documents(documents)
    embeddings = get_embeddings()
    create_vectorstore(chunks, embeddings)


def answer_question(question):
    embeddings = get_embeddings()
    db = load_vectorstore(embeddings)

    docs = db.max_marginal_relevance_search(
        question,
        k=5,
        fetch_k=10
    )

    chain = get_rag_chain()

    response = chain.invoke({
        "chat_history": st.session_state.chat_history,
        "context": format_docs(docs),
        "question": question
    })

    # Save conversation
    st.session_state.chat_history += f"\nUser: {question}\nAssistant: {response}\n"

    # Extract sources
    sources = []

    for doc in docs:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")

        sources.append(f"{source} - Page {page}")

    unique_sources = list(set(sources))

    source_text = "\n\n".join([f"• {s}" for s in unique_sources])

    final_response = f"""
{response}

---
📚 Sources:
{source_text}
"""

    return final_response


def main():
    st.set_page_config("Doc QA Chatbot")
    st.header("Chat with PDFs 💬")

    # Initialize memory
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ""

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.text_input("Ask a question")

    if question:
        # Show user message
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        # Generate response
        response = answer_question(question)

        # Show assistant message
        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

    with st.sidebar:
        st.title("Upload PDFs")

        pdf_files = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True
        )

        if st.button("Process PDFs"):
            with st.spinner("Processing..."):
                process_pdfs(pdf_files)
                st.success("Processing Complete!")


if __name__ == "__main__":
    main()