import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# PAGE CONFIG


st.set_page_config(
    page_title="DigiSailor AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 DigiSailor AI Assistant")
st.write("Ask anything about DigiSailor.")


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

# LOAD PDF


PDF_PATH = "Digisailor_Company_Profile.pdf"

loader = PyPDFLoader(PDF_PATH)
pages = loader.load()


# SPLIT DOCUMENTS


splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=150
)

chunks = splitter.split_documents(pages)

# =====================================================
# EMBEDDINGS
# =====================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =====================================================
# VECTOR DATABASE
# =====================================================

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

# =====================================================
# RETRIEVER
# =====================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k":3}
)

# =====================================================
# FORMAT DOCUMENTS
# =====================================================

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# =====================================================
# LLM
# =====================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0.5
)

# =====================================================
# PROMPT
# =====================================================

prompt = ChatPromptTemplate.from_template("""
You are DigiSailor AI Assistant.

Use the following context from the DigiSailor PDF to answer the user's question.

If the context contains the answer:
- Answer using only the provided context.
- Be clear, concise, and accurate.
- Mention that the source is the DigiSailor PDF.

If the context is empty, insufficient, or does not answer the question:
- Use your available web search results or general knowledge.
- Prefer official and trustworthy sources.
- Mention that the source is the Web.

If both the PDF context and web results are useful:
- Combine them into one complete answer.
- Clearly distinguish which information comes from the PDF and which comes from the Web.

If you cannot find the answer anywhere:
- Say that you don't have enough information.
- Do not make up facts.

PDF Context:
{context}

User Question:
{input}

Answer:
""")

# =====================================================
# RAG CHAIN
# =====================================================

chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# =====================================================
# CHAT HISTORY
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =====================================================
# USER INPUT
# =====================================================

question = st.chat_input("Ask a question about DigiSailor...")

if question:

    # Display user message
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Searching documents..."):

            response = chain.invoke(question)

            st.markdown(response)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":response
        }
    )