import streamlit as st
from langchain_neo4j import Neo4jGraph

# Connect to Neo4j
try:
    graph = Neo4jGraph(
        url=st.secrets["NEO4J_URI"],
        username=st.secrets["NEO4J_USERNAME"],
        password=st.secrets["NEO4J_PASSWORD"],
        database=st.secrets["NEO4J_DATABASE"],
    )
except Exception as error:
    raise RuntimeError(
        "Neo4j initialization failed"
    ) from error