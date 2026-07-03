# 📚 RAG Book Chat

An AI-powered **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and interact with them through natural language. The application retrieves relevant information from the uploaded document using semantic search and generates accurate responses using the **Mistral AI** Large Language Model.

---

## 🚀 Features

* 📄 Upload any PDF document
* ✂️ Automatic document chunking
* 🧠 Semantic search using vector embeddings
* 🗂️ ChromaDB vector database for efficient retrieval
* 🤖 AI-powered responses using Mistral AI
* 💬 Interactive Streamlit chat interface
* 🎨 Custom forest green themed UI
* 🔍 Uses Maximum Marginal Relevance (MMR) retrieval for improved search results

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **LangChain**
* **ChromaDB**
* **Hugging Face Embeddings**
* **Mistral AI**
* **PyPDF**
* **python-dotenv**

---

## 📂 Project Structure

```text
RAG-Book-Chat/
│
├── app.py
├── main.py
├── create_database.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
├── chroma-db/
├── document loaders/
├── retrievers/
├── vector store/
└── assets/
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/vaibhaviimahajan/RAG-Book-Chat.git
```

### 2. Navigate to the project

```bash
cd RAG-Book-Chat
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows**

```bash
.venv\Scripts\activate
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Create a `.env` file

Add your Mistral API key:

```env
MISTRAL_API_KEY=your_api_key_here
```

---

## ▶️ Run the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 💡 How It Works

1. Upload a PDF document.
2. The PDF is loaded using **PyPDFLoader**.
3. The document is split into smaller chunks using **RecursiveCharacterTextSplitter**.
4. Each chunk is converted into vector embeddings using **Hugging Face Embeddings**.
5. The embeddings are stored in **ChromaDB**.
6. When a question is asked, the retriever finds the most relevant chunks.
7. The retrieved context is sent to **Mistral AI**.
8. The generated answer is displayed in the chat interface.


