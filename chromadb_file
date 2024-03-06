import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
import os

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = 'your_key' #add your open ai api key here

# Function to get vector store from URL
def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(document_chunks, embedding=embeddings, persist_directory='./vector_data_openAI')
    vector_store.persist()
    return vector_store

# Function to create or load vector store without ingesting docs
def no_ingest_docs():
    persist_directory = './vector_data_openAI'
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vector_store

# Function to create a history-aware retriever chain
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo')
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up for relevant information")
    ])
    retriever = vector_store.as_retriever()
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

# Function to create a conversational RAG chain
def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# Function to get response based on user input
def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response['answer']

# App configuration
st.set_page_config(page_title="Chat bot ")
st.title("Chatbot")

# Sidebar
with st.sidebar:
    st.header("Knowledge base")
    website_url = st.text_input("URL")

    if website_url is None or website_url == "":
        st.info("Add a URL to Process New Data")

# Process New Data Button
if st.button("Process New Data"):
    # Initialize or load session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello, How can I help you?")]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    # User input
    user_query = st.text_input("Type your message here...")
    if user_query:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Display conversation
    for message in st.session_state.chat_history:
        role = "AI" if isinstance(message, AIMessage) else "Human"
        with st.container():
            st.write(role + ": " + message.content)

# If Process New Data button is not clicked
else:
    # Initialize or load session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello, How can I help you?")]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = no_ingest_docs()

    # User input
    user_query = st.text_input("Type your message here...")
    if user_query:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Display conversation
    for message in st.session_state.chat_history:
        role = "AI" if isinstance(message, AIMessage) else "Human"
        with st.container():
            st.write(role + ": " + message.content)
