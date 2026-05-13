from langchain_community.vectorstores import FAISS

def create_vectorstore(chunks, embeddings):
    db = FAISS.from_texts(chunks, embedding=embeddings)
    db.save_local("data/faiss_index")


def load_vectorstore(embeddings):
    return FAISS.load_local(
        "data/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )