import os
import tempfile
import streamlit as st
import numpy as np
np.float_ = np.float64
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Page Layout
st.set_page_config(page_title="Gemini Document Assistant", page_icon="✨", layout="wide")
st.title("✨ RAG Document Assistant (Powered by Gemini API)")

# Sidebar Setup
with st.sidebar:
    st.header("1. API Configuration")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    
    st.divider()
    st.header("2. Document Upload")
    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt"])
    
    if st.button("Clear Chat Memory"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

@st.cache_resource
def load_embedding_model():
    """Loads local HuggingFace embeddings once to optimize performance and prevent API limit errors."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_file(uploaded_file):
    """Parses document and generates vector database locally."""
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    if file_extension == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif file_extension == ".docx":
        loader = Docx2txtLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path)

    documents = loader.load()
    os.remove(tmp_path)

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(documents)

    # Local Vector Storage via ChromaDB
    embeddings = load_embedding_model()
    vectorstore = Chroma.from_documents(splits, embeddings)
    return vectorstore.as_retriever()

# State Management
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Document Processing Listener
retriever = None
if uploaded_file:
    with st.spinner("Processing document embeddings..."):
        if "retriever" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
            try:
                st.session_state.retriever = process_file(uploaded_file)
                st.session_state.current_file = uploaded_file.name
                st.session_state.chat_history = []
                st.session_state.messages = []
                st.success("Document indexed successfully!")
            except Exception as e:
                st.error(f"Error processing document: {e}")
    retriever = st.session_state.get("retriever")

# Render Active Conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Query Processing
if prompt := st.chat_input("Ask a question about your document..."):
    clean_api_key = api_key.strip() if api_key else ""
    
    if not clean_api_key:
        st.error("Please enter a valid Gemini API Key in the sidebar.")
    elif not retriever:
        st.error("Please upload a document before asking questions.")
    else:
        # Display User Input
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Environment Key Injection to prevent authorization errors
        os.environ["GOOGLE_API_KEY"] = clean_api_key

        # Active Gemini LLM Initialization
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash", 
            temperature=0, 
            google_api_key=clean_api_key
        )

        # 1. Query Reformulation
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # 2. Answer Construction
        qa_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know.\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # 3. Stream Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document with Gemini..."):
                try:
                    response = rag_chain.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.chat_history
                    })
                    answer = response["answer"]
                    st.markdown(answer)
                    
                    # Update Memory
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.chat_history.append(HumanMessage(content=prompt))
                    st.session_state.chat_history.append(AIMessage(content=answer))
                except Exception as e:
                    st.error(f"API Execution Error: {e}")