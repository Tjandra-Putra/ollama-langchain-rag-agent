from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Vector Search, a database hosted locally called ChromaDB
# This will allow us to perform vector searches over our documents
# Pass to our model so that data can be used to give contextual information

# load data
df = pd.read_csv("data/realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# location to store vector db
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)  # vectorise once

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(page_content=row["Title"] + " " + row["Review"], metadata={"rating": row["Rating"], "date": row["Date"]}, id=str(i))

    ids.append(str(i))
    documents.append(document)

vector_store = Chroma(collection_name="restaurant_reviews", persist_directory=db_location, embedding_function=embeddings)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# search_kwargs - searches 5 reviews similar to the query
