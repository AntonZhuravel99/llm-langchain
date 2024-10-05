

from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from pinecone import Pinecone

import os
import numpy as np


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
pinecone_env = os.getenv("PINECONE_ENV")


embeddings = OpenAIEmbeddings()
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
retriever = vector_store.as_retriever()


print("RUN")


select = None
with st.sidebar:
    select =st.selectbox("Select a type", ["GPT-only", "Data based"])



# print("retriever", retiver)
st.title("💬 Data bot")
st.caption("🚀 A chatbot powered by OpenAI")
uploaded_file = st.file_uploader("Upload an file", type=("xlsx"))

if uploaded_file is not None:
    df= pd.read_excel(uploaded_file)
    # Define the number of rows per chunk
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    # Split the DataFrame into chunks of size `chunk_size`
    df['text'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    
    for idx, row in df.iterrows():
        percent_complete = (idx + 1) / len(df)
        my_bar.progress(percent_complete, text=progress_text)
        text = row['text']
        vector_store.add_texts([text], ids=[str(idx)])

    st.rerun()
    st.info("🚀 Data added to the Vector Store successfully!")    


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])



if prompt := st.chat_input():
    # Initialize memory to store conversation history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    # response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=retriever)

    response = qa_chain.run(prompt)
    # st.info("🚀 Query executed successfully!")

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)



# def get_response(prompt):
#     extendedPrompt = PromptTemplate(
#         input_variables=["chat_history", "human_input", "context"], template=template
#     )
#     # Initialize memory to store conversation history
#     memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
