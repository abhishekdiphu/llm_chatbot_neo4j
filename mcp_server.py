from mcp.server.fastmcp import FastMCP
from llm import llm
from langchain_core.output_parsers import StrOutputParser

from tools.vector_tool import VectorTool
from tools.cypher_tool import CypherTool

from langchain_core.prompts import ChatPromptTemplate
from dataclasses import dataclass
from neo4j import AsyncDriver, AsyncGraphDatabase
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import os
from mcp.server.fastmcp import Context



@dataclass
class AppContext:
    """Application context with Neo4j driver."""
    driver: AsyncDriver
    database: str

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage Neo4j driver lifecycle."""

    # Read connection details from environment
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    #uri = os.getenv("NEO4J_URI_LOCAL")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    # Initialize driver on startup
    driver = AsyncGraphDatabase.driver(uri, auth=(username, password))

    try:
        # Yield context with driver
        yield AppContext(driver=driver, database=database)
    finally:
        # Close driver on shutdown
        await driver.close()







class MCPServer():
    def __init__(self):

        self.mcp = FastMCP("scientific-paper-tools", lifespan=app_lifespan)
        

        self.vector_tool = VectorTool()
        self.cypher_tool = CypherTool()
        self.general_chat_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
            You are a scientific paper assistant.

            Use this tool only for general academic-paper explanations.
            Examples:
            - What is a DOI?
            - What is an abstract?
            - What is peer review?
            - What is semantic search?
            - What is an arXiv category?

            Do not use this tool to find, recommend, retrieve, or summarize papers from the database.
            For database questions, the agent should use vector_search or cypher_search.
            """
                    ),
                    ("human", "{input}"),
                ]
            )
        self.general_chat_chain = self.general_chat_prompt | llm | StrOutputParser()

        self.register_tools()
        self.register_resources()

    
    def run(self):
        
        self.mcp.run()




    def register_resources(self):
        
        @self.mcp.resource("neo4j://schema")
        def kg_sc_schema() -> str:
    
            return """
                    Scientific Paper Knowledge Graph Schema

                    Node labels:
                    - Paper
                    Properties:
                    - id
                    - submitter
                    - authors_raw
                    - title
                    - comments
                    - journal_ref
                    - doi
                    - report_no
                    - license
                    - abstract
                    - update_date

                    - Author
                    Properties:
                    - name

                    - Category
                    Properties:
                    - name

                    - Version
                    Properties:
                    - id
                    - paper_id
                    - version
                    - created

                    - Chunk
                    Properties:
                    - id
                    - paper_id
                    - chunk_type
                    - text
                    - embedding
                    - embedding_model
                    - embedding_dimensions

                    Relationships:
                    - (:Paper)-[:AUTHORED_BY]->(:Author)
                    - (:Paper)-[:IN_CATEGORY]->(:Category)
                    - (:Paper)-[:HAS_VERSION]->(:Version)
                    - (:Paper)-[:HAS_CHUNK]->(:Chunk)
                    """
        


    def register_tools(self):
            
        @self.mcp.tool()
        def vector_search(question: str) -> str:
            """
            Semantic search over scientific papers using Neo4j vector search.
            Use this for topic-based search, recommendations, and similarity search.
            """
            response = self.vector_tool.get_scientific_paper_answer(question)

            if isinstance(response, dict):
                return response.get("answer") or response.get("result") or str(response)

            return str(response)


        @self.mcp.tool()
        def cypher_search(question: str) -> str:
            """
            Structured search over the scientific paper Neo4j graph using Cypher.
            Use this for authors, categories, paper IDs, DOI, versions, and metadata.
            """
            response = self.cypher_tool.cypher_qa.invoke({"query": question})

            if isinstance(response, dict):
                return response.get("result") or response.get("answer") or str(response)

            return str(response)


        


        @self.mcp.tool()
        def general_paper_chat(question: str) -> str:
            """
            General scientific-paper chat.

            Use this only for general academic concept explanations.
            Do not use this for database retrieval, paper recommendations, author lookup,
            category lookup, or paper metadata questions.
            """
            return self.general_chat_chain.invoke({"input": question})

        
        @self.mcp.tool()
        async def graph_statistics(ctx: Context) -> dict[str, int]:
            """Count the number of nodes and 
            relationships in the graph."""

            # Access the driver from lifespan context
            driver = ctx.request_context.lifespan_context.driver
            database = ctx.request_context.lifespan_context.database

            # Use the driver to query Neo4j with the correct database
            records, summary, keys = await driver.execute_query(
                r"RETURN COUNT {()} AS nodes, COUNT {()-[]->()} AS relationships",
                database_=database
            )

            # Process the results
            if records:
                return dict(records[0])
            return {"nodes": 0, "relationships": 0}
