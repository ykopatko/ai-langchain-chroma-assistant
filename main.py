import os
import chromadb
import streamlit as st

from dotenv import load_dotenv
from uuid import uuid4

from chromadb.config import Settings

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate
)
from langchain.schema import HumanMessage, AIMessage

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

project_directory = r"Your local directory"  # Change this to your local directory
persist_directory = os.path.join(project_directory, "db.chromadb")

# Create ChromaDB client and OpenAI embedding function
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=persist_directory
))
embeddings = OpenAIEmbeddings()

st.markdown("# 🦜🔗 AI Assistant")

collection_name = "conversation_history"

try:
    collection = client.get_collection(collection_name)
except ValueError:
    collection = client.create_collection(collection_name)


# Set up the initial prompt for the chat
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "The following is a friendly conversation between a human and an AI. "
        "The AI is talkative and provides lots of specific details from its context. "
        "If the AI does not know the answer to a question, it truthfully says it does not know."
    ),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# Initialize the chat model and language model chain
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
chain = LLMChain(llm=chat, prompt=prompt)

# Set up the Streamlit app interface
if "history" not in st.session_state:
    st.session_state.history = []

if "conversation" not in st.session_state:
    st.session_state.conversation = []


input_text = st.text_area("You: ")

# Send user input to the chat model when the "Send" button is clicked
if st.button("Send"):
    response = chain.run(input=input_text, history=st.session_state.history)
    st.session_state.conversation.append(f"You: {input_text}\n")
    st.session_state.conversation.append(f"\nAI: {response}\n")
    st.session_state.history.append(HumanMessage(content=input_text))
    st.session_state.history.append(AIMessage(content=response))

    # Add conversation history to ChromaDB
    full_conversation = "\n".join(st.session_state.conversation)
    embeddings_full_conversation = embeddings.embed_documents([full_conversation])[0]
    collection.add(
        ids=[str(uuid4())],
        embeddings=[embeddings_full_conversation],
        metadatas=[{"type": "conversation"}],
        documents=[full_conversation]
    )

# Reset the conversation when the "Reset" button is clicked
if st.button("Reset"):
    st.session_state.conversation = []
    st.session_state.history = []


# Display the conversation in the Streamlit app
st.markdown("\n".join(st.session_state.conversation))
