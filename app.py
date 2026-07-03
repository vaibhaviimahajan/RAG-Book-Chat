import tempfile
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ---------------------------
# Models
# ---------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
llm = ChatMistralAI(model="mistral-small-2506")

# ---------------------------
# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="📚 RAG Book Chat",
    page_icon="📚",
    layout="wide"
)

# ---------------------------
# Custom CSS
# ---------------------------

st.markdown("""
<style>

/* ===========================
BACKGROUND
=========================== */

.stApp{
    background:#0B3D2E;
}

/* Main container */

.block-container{
    max-width:950px;
    padding-top:5rem;
}

/* ===========================
TITLE
=========================== */

h1{
    color:white;
    text-align:center;
    font-size:48px;
    margin-bottom:5px;
}

.subtitle{
    text-align:center;
    color:white;
    font-size:25px;
    margin-bottom:35px;
}

/* ===========================
UPLOAD BOX
=========================== */



/* Streamlit uploader */

[data-testid="stFileUploader"]{

    background:#0B3D2E !important;

    border:2px dashed white !important;

    border-radius:15px;

    padding:20px;
}

/* remove white */

[data-testid="stFileUploader"] *{

    color:white !important;

    background:transparent !important;
}

/* Browse button */

.stButton>button{

    background:#145A32;

    color:white;

    border:2px solid white;

    border-radius:12px;

    font-weight:bold;

    height:50px;
}

/* ===========================
CHAT INPUT
=========================== */

[data-testid="stChatInput"]{

    background:#0B3D2E !important;

    border:2px solid white !important;

    border-radius:25px !important;
}

[data-testid="stChatInput"] textarea{

    background:#0B3D2E !important;

    color:white !important;

    caret-color:white !important;
}

[data-testid="stChatInput"] textarea::placeholder{

    color:#dddddd !important;
}
/* ===========================
CHAT BOXES
=========================== */

[data-testid="stChatMessage"]{

    background:#145A32;

    border-radius:18px;

    padding:18px;

    margin-bottom:15px;
}

/* text */

[data-testid="stMarkdownContainer"]{

    color:white;
}

/* ===========================
USER AVATAR
=========================== */

[data-testid="stChatMessageAvatarUser"]{

    background:#2E8B57 !important;

    border-radius:50%;
}

/* BOT AVATAR */

[data-testid="stChatMessageAvatarAssistant"]{

    background:#8B5A2B !important;

    border-radius:50%;
}

/* ===========================
SUCCESS
=========================== */

.stSuccess{

    background:#145A32;
    color:white;
}

/* ===========================
SPINNER
=========================== */

.stSpinner{

    color:white;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<h1>📚 Chat with Your Book</h1>", unsafe_allow_html=True)

st.markdown(
    "<p class='subtitle'>Upload your PDF and ask questions using RAG.</p>",
    unsafe_allow_html=True
)
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📄 Upload PDF",
    type=["pdf"]
)
# ---------------------------
# Session State
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    if st.button("🚀 Create Vector Database"):

        with st.spinner("Processing PDF..."):

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)
            print("Number of chunks:", len(chunks))
            print("Chunk type:", type(chunks[0]))
            print("Page content type:", type(chunks[0].page_content))
            print("Page content value:", repr(chunks[0].page_content[:200]))
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory="chroma-db"
            )

        st.success("✅ Database Created Successfully!")
# ---------------------------
# Load Vector DB
# ---------------------------
vectorstore = Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# ---------------------------
# Prompt
# ---------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say:

"I could not find the answer in the document."
"""
        ),

        (
            "human",
            """
Context:
{context}

Question:
{question}
"""
        )
    ]
)

# ---------------------------
# Display Chat History
# ---------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------
# User Input
# ---------------------------
query = st.chat_input("Ask a question about the book...")

if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    docs = retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    response = llm.invoke(final_prompt)

    answer = response.content

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )