from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        # model="text-embedding-004"
        model = "gemini-embedding-001"
    )