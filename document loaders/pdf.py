from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader("document loaders/GRU.pdf")

docs = data.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 10
)

chunks = splitter.split_documents(docs)

#print(len(chunks))
print(chunks[0].page_content)

#print(docs[14]) #in this pdf GRU there are 15 pages and each page makes 1 doc so 15 doc so we give 14 no. to extract last page 
#metadata and page content