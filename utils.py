import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)
from pinecone import Pinecone, ServerlessSpec
from io import BytesIO
from PyPDF2 import PdfReader
from langchain_core.documents.base import Document

TOKEN = os.environ.get('TOKEN')

class PineconeDB:
    def __init__(self, api_key: str, index_name: str, dim=768): # in future: change the dimensionality and setup based on a chosen encoder
        self.pinecone = Pinecone(api_key=api_key)
        try:
            self.pinecone.create_index(index_name, 
                                metric="cosine", 
                                dimension=dim, 
                                spec=ServerlessSpec(cloud="aws",
                                                    region="us-east-1"
                                                    )
            )
        except Exception:
            pass
        self.index = self.pinecone.Index(index_name)

    def similarity_search(self, query: list, top_k: int):
        results = self.index.query(vector=query, top_k=top_k, namespace="", include_metadata=True)
        return ''.join(result.metadata['text'] for result in results.matches)
    
    def exit(self):
        self.pinecone.delete_index(self.index.name)

class Encoder():
    def __init__(self, model_name, model_kwargs) -> None:
        self.embedding_function = HuggingFaceEmbeddings(
            model_name = model_name,
            model_kwargs = model_kwargs,
            token=TOKEN
        )

def load_pdf_bytes(pdf_bytes):
    pdf_reader = PdfReader(BytesIO(pdf_bytes.getvalue()))
    num_pages = len(pdf_reader.pages)
    page_content = ""
    for i in range(num_pages):
        page = pdf_reader.pages[i]
        page_content += page.extract_text()
    return page_content

def loader_splitter(files: list, chunk_size: int=128, model='sentence-transformers/all-MiniLM-L12-v2'):
    pages = [Document(load_pdf_bytes(f), metadata={'source': f.name}) for f in files]

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=AutoTokenizer.from_pretrained(model),
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 20,
        strip_whitespace=True,
    )
    return text_splitter.split_documents(pages)
