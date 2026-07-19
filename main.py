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

PDF_PATH = "Digisailor_Company_Profile.pdf"

# CACHE PDF


@st.cache_data
def load_pdf():
    loader = PyPDFLoader(PDF_PATH)
    return loader.load()

# CACHE CHUNKS


@st.cache_data
def split_documents():
    pages = load_pdf()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150
    )

    return splitter.split_documents(pages)


# CACHE EMBEDDINGS


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# CACHE VECTOR STORE


@st.cache_resource
def load_vector_store():
    chunks = split_documents()
    embeddings = load_embeddings()

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )


# CACHE RETRIEVER


@st.cache_resource
def load_retriever():
    return load_vector_store().as_retriever(
        search_kwargs={"k": 3}
    )


# CACHE LLM


@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.5
    )


# FORMAT DOCS


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# PROMPT


prompt = ChatPromptTemplate.from_template("""
You are DigiSailor AI Assistant.

Use the following context from the DigiSailor PDF to answer the user's question.

If the context contains the answer:
- Answer using only the provided context.
- Mention that the source is the DigiSailor PDF.

If the context is insufficient:
- Use your general knowledge.
- Mention that the source is general knowledge.

Context:
{context}

Question:
{question}

Answer:
""")

# CACHE CHAIN


@st.cache_resource
def load_chain():

    retriever = load_retriever()
    llm = load_llm()

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

chain = load_chain()


# CHAT HISTORY


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# USER INPUT


question = st.chat_input("Ask a question about DigiSailor...")

if question:

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("assistant"):

        with st.spinner("Searching..."):

            response = chain.invoke(question)

            st.markdown(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )