import os
import pickle
import random
import streamlit as st


from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatAnthropic
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores.qdrant import Qdrant

from streamlit.web.server import websocket_headers
from streamlit_chat import message


qdrant_url = 'https://58de2381-e750-4aed-8eb2-7b08d8faf30b.us-east4-0.gcp.cloud.qdrant.io:6333'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/mnt/data/RAG-mktg/model_cache/'


model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en",
                                      model_kwargs=model_kwargs,
                                      encode_kwargs=encode_kwargs
                                     )

doc_store = Qdrant.from_texts(texts,
                          metadatas=metadatas,
                          embedding=embeddings,
                          url=qdrant_url,
                          api_key=os.environ['QDRANT_API_KEY'],
                          collection_name=f"medical_qa_search")


prompt_template = """Use the following pieces of context to answer the question enclosed within  3 backticks at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
Please provide an answer which is factually correct and based on the information retrieved from the vector store.
Please also mention any quotes supporting the answer if any present in the context supplied within two double quotes "" .

{context}

QUESTION:```{question}```
ANSWER:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context","question"])
#
chain_type_kwargs = {"prompt": PROMPT}


# Uncomment if you want to store and use the OpenAI key stored in an environment variable
anthropic_key = os.getenv('ANTHROPIC_API_KEY') 
qdrant_key = os.environ['QDRANT_API_KEY']

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]


st.set_page_config(initial_sidebar_state='collapsed')
anthropic_key = st.sidebar.text_input("Enter your Anthropic API key", type="password")
qdrant_key = st.sidebar.text_input("Enter your Anthropic API key", type="password")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

qa_chain = None


if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    memory.clear()


if doc_store and anthropic_key:
    rag_llm = ChatAnthropic(temperature=0,
                            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"])
    
    qa_chain = RetrievalQA.from_chain_type(llm=rag_llm,
                                       chain_type="stuff",
                                       chain_type_kwargs={"prompt": PROMPT},
                                       retriever=doc_store.as_retriever(search_kwargs={"k": 5}),
                                       return_source_documents=True
                                      )

# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')
    if submit_button and user_input and qa_chain and anthropic_key:
        with st.spinner("Searching for the answer..."):
            result = qa_chain(user_question)
        answer = result["result"]
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(answer)
        
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))