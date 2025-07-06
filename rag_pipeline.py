import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def load_and_translate_documents(pdf_folder):
    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, filename)
            with fitz.open(file_path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                documents.append(text)
    return documents

def create_vectorstore(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents(texts)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = FAISS.from_documents(docs, embedding_model)
    return vectorstore

def retrieve_relevant_chunks(vectorstore, query):
    results = vectorstore.similarity_search(query, k=3)
    return [doc.page_content for doc in results]
