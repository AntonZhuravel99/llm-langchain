from openai import OpenAI
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
import pandas as pd
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
pinecone_env = os.getenv("PINECONE_ENV")

    # Step 3: Initialize Embeddings and Pinecone
embeddings = OpenAIEmbeddings()
# st.info("🚀 Embeddings initialized successfully!")
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

st.title("💬 Wiki bot")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")
button = st.button("Run")
# uploaded_file = st.file_uploader("Upload an article", type=("pdf"))
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    if "file" in msg:
        file = msg["file"]
        st.file_uploader("Upload File", type=file["type"])


if button:
    query = "Tell me about Artisan"
    docs = vector_store.similarity_search(query, k=2)
    st.info("🚀 Query executed successfully!")
    for doc in docs:
        st.info(f"Text: {doc.page_content}")
    # # Step 1: Read Excel File
    # df = pd.read_excel('data/Italy_short.xlsx')
    # # Step 2: Convert DataFrame Rows to Text
    df['text'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)


    # st.info("🚀 Embeddings initialized successfully!")
    # # Step 4: Add Data to Vector Store
    for idx, row in df.iterrows():
        text = row['text']
        vector_store.add_texts([text], ids=[str(idx)])

    # st.info("🚀 Data added to the Vector Store successfully!")
    # # # Step 5: Query the Vector Store
    # query = "Tell me about Artisan"
    # docs = vector_store.similarity_search(query, top_k=5)
    # st.info("🚀 Query executed successfully!")
    # for doc in docs:
    #     st.info(f"Text: {doc.page_content}")


if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    client = OpenAI(api_key=openai_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
