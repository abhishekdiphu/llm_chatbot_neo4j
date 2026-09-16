import streamlit as st

# Create the LLM
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

custom_llm = False

openai_api_key = st.secrets['OPENAI_API_KEY']
openai_model = st.secrets['OPENAI_MODEL']
llm = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"],
                 model=st.secrets["OPENAI_MODEL"],
                 max_completion_tokens=4096)


if custom_llm:
    from custom_llms.custom_llm import cus_llm
    cus_llm = cus_llm 
else:
    cus_llm = llm






embeddings = OpenAIEmbeddings(
    model=st.secrets.get("EMBEDDING_MODEL_NAME", "embeddinggemma-300m-GGUF:BF16"),
    base_url=st.secrets.get("LLAMA_EMBEDDING_BASE_URL", "http://localhost:8081/v1"),
    #base_url=st.secrets.get("LLAMA_EMBEDDING_BASE_URL", "http://host.docker.internal:8081/v1"),
    api_key=st.secrets.get("LLAMA_EMBEDDING_API_KEY", "not-needed"),
)

