from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_rag_chain():

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful PDF assistant.

    Use the provided context and chat history to answer the question.

    If answer is not available in context, say:
    "Answer is not available in the context."

    Chat History:
    {chat_history}

    Context:
    {context}

    Question:
    {question}
    """)

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    chain = prompt | model

    return chain


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)