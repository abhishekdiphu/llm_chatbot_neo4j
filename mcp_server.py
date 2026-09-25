from mcp.server.fastmcp import FastMCP
from llm import memory_llm, mem_embeddings 
from tools.vector_tool import VectorTool
from tools.cypher_tool import CypherTool


from dataclasses import dataclass
from neo4j import AsyncDriver, AsyncGraphDatabase
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import os
from mcp.server.fastmcp import Context


from neo4j_agent_memory import (
    MemoryClient,
    MemorySettings,
    ExtractionConfig,
)
from neo4j_agent_memory.integrations.langchain import (
    llm_provider_from_langchain,
)
from neo4j_agent_memory.config.settings import ExtractorType
from neo4j_agent_memory.config import Neo4jConfig

from tools.arvix_tool import arvix_search
import asyncio

@dataclass
class AppContext:
    """Application context with Neo4j driver."""
    driver: AsyncDriver
    database: str
    memory_client: MemoryClient

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage Neo4j driver lifecycle."""

    # Read connection details from environment
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    # Initialize driver on startup
    driver = AsyncGraphDatabase.driver(uri, auth=(username, password))

    memory_settings = MemorySettings(neo4j=Neo4jConfig(uri=uri, username=username, password=password, database=database),
        extraction=ExtractionConfig(enable_spacy=False, enable_gliner=False,),
        llm=memory_llm,
        embedding=mem_embeddings,
    )

    non_llm_memory_settings = MemorySettings(neo4j=Neo4jConfig(uri=uri, username=username, password=password, database=database),
            extraction=ExtractionConfig(extractor_type=ExtractorType.SPACY, enable_llm_fallback=False),
            llm= None,
            embedding=mem_embeddings)

    memory_client = MemoryClient(memory_settings)
    await memory_client.connect()

    try:
        # Yield context with driver
         yield AppContext(
            driver=driver,
            database=database,
            memory_client=memory_client,
        )
    finally:
        await memory_client.close()
        await driver.close()







class MCPServer():
    def __init__(self):

        self.mcp = FastMCP("scientific-paper-tools", lifespan=app_lifespan)
        

        self.vector_tool = VectorTool()
        self.cypher_tool = CypherTool()
        self.register_tools()
        self.register_resources()

    
    def run(self,transport="stdio"):
        self.mcp.run(transport=transport)




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
    #-----------------------------      
        @self.mcp.tool()
        async def get_chat_history(
            session_id: str,
            ctx: Context,
        ) -> list[dict[str, str]]:
            """Return messages for one chat session."""

            memory_client = (
                ctx.request_context
                .lifespan_context
                .memory_client
            )

            conversation = (
                await memory_client.short_term.get_conversation(
                    session_id=session_id,
                )
            )

            return [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in conversation.messages
            ]



        @self.mcp.tool()
        async def add_chat_message(
            session_id: str,
            role: str,
            content: str,
            user_id: str | None,
            ctx: Context,
        ) -> str:
            """Save one message."""

            memory_client = (
                ctx.request_context
                .lifespan_context
                .memory_client
            )

            await memory_client.short_term.add_message(
                session_id=session_id,
                role=role,
                content=content,
                metadata=(
                    {"user_id": user_id}
                    if user_id
                    else None
                ),
            )

            return "Message saved"


        @self.mcp.tool()
        async def clear_chat_history(
            session_id: str,
            ctx: Context,
        ) -> str:
            """Delete one chat session."""

            memory_client = (
                ctx.request_context
                .lifespan_context
                .memory_client
            )

            await memory_client.short_term.delete_session(
                session_id=session_id,
            )

            return "History cleared"
      #-----------------------------      
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


        


    #   @self.mcp.tool()
    #    def general_paper_chat(question: str) -> str:
    #        """
    #        General scientific-paper chat.
    #
    #        Use this only for general academic concept explanations.
    #        Do not use this for database retrieval, paper recommendations, author lookup,
    #        category lookup, or paper metadata questions.
    #        """
    #        return self.general_chat_chain.invoke({"input": question})

        
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
            else:
                return {"nodes": 0, "relationships": 0}
#----------------long term memory tools-----------------------------
        @self.mcp.tool()
        async def get_user_preferences( user_id: str, ctx: Context, ) -> dict[str, list[str]]:
            """Return long-term preferences for a user."""

            memory_client = (ctx.request_context.lifespan_context.memory_client)

            preferences = await memory_client.long_term.get_preferences_for( user_identifier=user_id)
            grouped: dict[str, list[str]] = {}
            for item in preferences:
                grouped.setdefault(item.category,[],).append(item.preference)

            return grouped


        @self.mcp.tool()
        async def set_user_preferences( preference: str, category: str, user_id: str, ctx: Context,) -> str:
                    """Return long-term preferences for a user."""
        
                    memory_client = (ctx.request_context.lifespan_context.memory_client)
        
                    preferences_to_save = await memory_client.long_term.add_preference(category=category, preference=preference, confidence=0.85, user_identifier=user_id)

                    return (f"Saved preference: " f"{preferences_to_save.category} - {preferences_to_save.preference}")


        @self.mcp.tool()
        async def arvix_paper_chat(query: str,  max_results: int = 5) -> str:

            try:
                result = await asyncio.to_thread( arvix_search, query, max_results)
            except Exception as e:
                print(f"Error occurred while searching arXiv: {e}")
                result = "Error occurred while searching arXiv."

            print("result length:", len(result))
            return result






#------------------------tracing------------------------------#
        @self.mcp.tool()
        async def start_reasoning_trace(
                                        session_id: str,
                                        user_id: str,
                                        task: str,
                                        ctx: Context,
                                        ) -> dict[str, str]:

                memory_client = (ctx.request_context.lifespan_context.memory_client)
                trace = await memory_client.reasoning.start_trace(session_id=session_id,task=task,metadata={"user_id": user_id})
                return {"trace_id": str(trace.id)}
                pass

        @self.mcp.tool()
        async def add_reasoning_step(
                                    trace_id: str,
                                    thought: str,
                                    action: str | None,
                                    ctx: Context,
                                    ) -> dict[str, str]:

                memory_client = (ctx.request_context.lifespan_context.memory_client)
                step = await memory_client.reasoning.add_step(trace_id=trace_id,thought=thought, action=action)
                return {"step_id": str(step.id)}
                pass

        @self.mcp.tool()
        async def record_reasoning_tool_call(
                                            step_id: str,
                                            tool_name: str,
                                            arguments: dict,
                                            result: dict | None,
                                            status: str,
                                            error: str | None,
                                            ctx: Context,
                                        ) -> str:
                memory_client = (ctx.request_context.lifespan_context.memory_client)
                await memory_client.reasoning.record_tool_call( step_id=step_id, tool_name=tool_name,
                                                                arguments=arguments, result=result,
                                                                status=status, error=error)

                return "Tool call recorded"
                pass

        @self.mcp.tool()
        async def complete_reasoning_trace(
                                            trace_id: str,
                                            outcome: str,
                                            success: bool,
                                            ctx: Context,
                                        ) -> str:
                memory_client = (ctx.request_context.lifespan_context.memory_client)
                await memory_client.reasoning.complete_trace(trace_id=trace_id, outcome=outcome, success=success)
                pass
        #------------------------tracing------------------------------#

           