from langchain_community.document_loaders import WebBaseloader

url = "https://www.apple.com/in/macbook-pro/"


#obj
data= WebBaseloader(url)

docs= data.load()

print(len(docs))