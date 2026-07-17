# 🤖 DigiSailor AI Assistant (RAG Chatbot)

An AI-powered chatbot built using **LangChain**, **Groq LLM**, **ChromaDB**, **HuggingFace Embeddings**, and **Streamlit**. The chatbot answers user queries using Retrieval-Augmented Generation (RAG) based on DigiSailor's company documents.

---

## 🚀 Features

- 📄 PDF-based Question Answering
- 🔍 Retrieval-Augmented Generation (RAG)
- 🧠 HuggingFace Sentence Embeddings
- 🗂️ Chroma Vector Database
- ⚡ Groq Llama 3.3 70B Model
- 💬 ChatGPT-style Streamlit Interface
- 📚 Context-aware responses
- ❌ Prevents hallucinations by answering only from the uploaded document

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- Groq
- ChromaDB
- HuggingFace Embeddings
- Sentence Transformers
- PyPDF

---

## 📂 Project Structure

```
Digisailor_Chatbot/
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
├── README.md
└── Digisailor_Company_Profile.pdf
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Digisailor_Chatbot.git

cd Digisailor_Chatbot
```

---

### 2. Create Virtual Environment

Windows

```bash
python -m venv .venv
```

Activate

```bash
.venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Create `.env`

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
```

---

### 5. Add Company PDF

Place your company PDF in the project folder.

Example:

```
Digisailor_Company_Profile.pdf
```

---

### 6. Run the Application

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. Load the company PDF.
2. Split the PDF into chunks.
3. Generate embeddings using HuggingFace.
4. Store embeddings in ChromaDB.
5. Retrieve relevant chunks based on the user's question.
6. Send the retrieved context and question to the Groq LLM.
7. Display the generated answer in Streamlit.

---

## 🔄 RAG Pipeline

```
PDF
   │
   ▼
PyPDFLoader
   │
   ▼
Text Splitter
   │
   ▼
Chunks
   │
   ▼
Embeddings
   │
   ▼
ChromaDB
   │
   ▼
Retriever
   │
   ▼
Context
   │
   ▼
Prompt
   │
   ▼
Groq LLM
   │
   ▼
Answer
```

---

## 📦 Dependencies

- Streamlit
- LangChain
- LangChain Community
- LangChain Groq
- LangChain Chroma
- LangChain HuggingFace
- Sentence Transformers
- ChromaDB
- PyPDF
- Python Dotenv

---

## 📸 Demo

You can add screenshots here after running the application.

Example:

```
screenshots/home.png
screenshots/chat.png
```

---

## 🎯 Future Improvements

- Upload PDF from the UI
- Conversation memory
- Multiple PDF support
- Source citation for answers
- Voice input and speech output
- Docker deployment

---

## 👨‍💻 Author

**V Nisanth**

Computer Science Engineering Student

AI | Machine Learning | Generative AI | RAG | Agentic AI

---

## 📜 License

This project is intended for educational and portfolio purposes.