import streamlit as st

from llm import cypher_llm, vector_llm
from graph import graph

from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

from tools.tools_prompts import CYPHER_GENERATION_TEMPLATE_03


class CypherTool():
    def __init__(self, graph=graph):
        super().__init__()
        self.graph = graph
        self.CYPHER_GENERATION_TEMPLATE = CYPHER_GENERATION_TEMPLATE_03

        self.cypher_prompt = PromptTemplate.from_template(self.CYPHER_GENERATION_TEMPLATE)


        # Create the Cypher QA chain.
        # This chain is intended for structured graph questions about scientific papers.
        self.cypher_qa = GraphCypherQAChain.from_llm(
            #llm=cypher_llm,
            cypher_llm=cypher_llm,
            qa_llm=vector_llm,
            include_types=["Paper", "Author", "Category", "Version", "Chunk", 
                           "AUTHORED_BY", "IN_CATEGORY", "HAS_VERSION", "HAS_CHUNK"],
            graph=graph,
            verbose=True,
            allow_dangerous_requests=True,
            return_direct=True,
            cypher_prompt= self.cypher_prompt)

if __name__ == "__main__":
    tool = CypherTool()
    question = "Find papers on the category 'math.AG' "
    print("Question:", question)
    result = tool.cypher_qa.invoke({"query": question})
    print("Response:", result)
