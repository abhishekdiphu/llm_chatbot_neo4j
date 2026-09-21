import streamlit as st

# Create the LLM
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from neo4j_agent_memory.config import EmbeddingConfig
from neo4j_agent_memory.config.settings import EmbeddingProvider


custom_llm = False

openai_api_key = st.secrets['OPENAI_API_KEY']
openai_model = st.secrets['OPENAI_MODEL']
llm = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"],
                 model=st.secrets["OPENAI_MODEL"],
                 max_completion_tokens=4096)


if custom_llm:
    from custom_llms.custom_llm import cus_llm
    cus_llm = cus_llm 
    llm = cus_llm
else:
    cus_llm = llm





try:
    embeddings = OpenAIEmbeddings(
        model=st.secrets.get("EMBEDDING_MODEL_NAME", "embeddinggemma-300m-GGUF:BF16"),
        base_url=st.secrets.get("LLAMA_EMBEDDING_BASE_URL", "http://localhost:8081/v1"),
        #base_url=st.secrets.get("LLAMA_EMBEDDING_BASE_URL", "http://host.docker.internal:8081/v1"),
        api_key=st.secrets.get("LLAMA_EMBEDDING_API_KEY", "not-needed"),
    )
except Exception as e:
    print("Error initializing embeddings:")
    embeddings = None

try:
    mem_embeddings =EmbeddingConfig(provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
                                    model="sentence-transformers/all-MiniLM-L6-v2",
                                    dimensions=384,
                                    device="mps")
except Exception as e:

    print("Error initializing mem embeddings:")

