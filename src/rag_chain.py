from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_rag_chain():

    prompt = PromptTemplate.from_template("""
    Answer the question based ONLY on the context below.

    If answer is not available, say:
    "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}
    """)

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    return prompt | model | StrOutputParser()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)