import os
import torch
import streamlit as st
from model import ChatModel
from utils import Encoder, loader_splitter, PineconeDB

API_KEY = os.environ.get('API_KEY')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.title("PDF Chatbot")

@st.cache_resource
def load_model(model="google/gemma-2b-it"):
    return ChatModel(model_id=model, device=DEVICE)

@st.cache_resource
def load_encoder(model="sentence-transformers/sentence-t5-base"):
    return Encoder(model_name=model, model_kwargs={})

@st.cache_resource
def initialize_db():
    return PineconeDB(api_key=API_KEY, index_name="pdf-chatbot")

@st.cache_resource
def load_doc(files):
    documents = loader_splitter(files)
    for doc in documents:
        vector = encoder.embedding_function.embed_query(doc.page_content)
        db.index.upsert(vectors=[{
            "id": doc.metadata["source"], 
            "values": vector, 
            "metadata": {"text": doc.page_content}
            }])
    
model = load_model(model="google/flan-t5-base")
encoder = load_encoder()
db = initialize_db()

with st.sidebar:
    # inputs and parameters in the sidebar
    max_new_tokens = st.number_input("max_new_tokens", 128, 4096, 2048)
    k = st.number_input("k", 1, 10, 6)
    uploaded_files = st.file_uploader(
        "Upload PDFs for context", type=["PDF", "pdf"], accept_multiple_files=True
    )
    
    # Upload files into a database
    if uploaded_files: load_doc(uploaded_files)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me anything!"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        user_prompt = st.session_state.messages[-1]["content"]
        vector = encoder.embedding_function.embed_query(user_prompt)
        context = (
            None if uploaded_files == [] else db.similarity_search(query=vector, 
                                                                   top_k=k, 
                                                                   )
        )
        answer = model.generate(
            user_prompt, context=context, max_new_tokens=max_new_tokens
        )
        response = st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})