from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

docs= [ 
       Document(page_content = "Python is widely used in Artificial Intelligence.", metadata={"source": "AI-book"}),
       Document(page_content = "Pandas is used for data analysis in Python.", metadata ={"source": "DataScience-book"}),
       Document(page_content = "Neural networks are used in deep learning.", metadata = {"source": "DL-book"}),
       ]

#embedding_model = OpenAIEmbeddings()
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = Chroma.from_documents(
    documents= docs,
    embedding= embedding_model,
    persist_directory = "chroma-db"
)

result= vectorstore.similarity_search("what is used for data analysis?", k= 2) #k is how many result ...like 2 embeddings

for r in result:
    print(r.page_content)
    print(r.metadata)
    
retriever = vectorstore.as_retriever()

docs = retriever.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)