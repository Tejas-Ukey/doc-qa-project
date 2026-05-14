from langchain_community.vectorstores import FAISS

def create_vectorstore(documents, embeddings):
    db = FAISS.from_documents(documents, embedding=embeddings)
    db.save_local("data/faiss_index")


def load_vectorstore(embeddings):
    return FAISS.load_local(
        "data/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )