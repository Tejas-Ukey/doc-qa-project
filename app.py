import streamlit as st

import config
from src.loader import load_pdfs
from src.splitter import split_text
from src.embeddings import get_embeddings
from src.vectorstore import create_vectorstore, load_vectorstore
from src.rag_chain import get_rag_chain, format_docs


def process_pdfs(pdf_files):
    text = load_pdfs(pdf_files)
    chunks = split_text(text)
    embeddings = get_embeddings()
    create_vectorstore(chunks, embeddings)


def answer_question(question):
    embeddings = get_embeddings()
    db = load_vectorstore(embeddings)

    docs = db.similarity_search(question, k=3)

    chain = get_rag_chain()

    response = chain.invoke({
        "context": format_docs(docs),
        "question": question
    })

    return response


def main():
    st.set_page_config("Doc QA")
    st.header("Chat with PDFs 💬")

    question = st.text_input("Ask a question")

    if question:
        answer = answer_question(question)
        st.write(answer)

    with st.sidebar:
        st.title("Upload PDFs")

        pdf_files = st.file_uploader(
            "Upload",
            accept_multiple_files=True
        )

        if st.button("Process"):
            with st.spinner("Processing..."):
                process_pdfs(pdf_files)
                st.success("Done!")


if __name__ == "__main__":
    main()