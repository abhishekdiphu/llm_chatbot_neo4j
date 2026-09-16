import asyncio
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
import json


class MCPClient:
    """
    MCP client for the scientific paper chatbot.

    This class connects to the local MCP server and calls MCP tools.
    It does not know anything about Neo4j, VectorTool, or CypherTool directly.
    """

    def __init__(self):
        self.server_params = StdioServerParameters(
            command="python",
            args=["-m", "mcp_server.main"],
        )

        self.mcp_server_url = os.getenv(
            "MCP_SERVER_URL")

    async def _call_tool_async(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        Internal async method that:
        1. starts/connects to the MCP server
        2. initializes an MCP session
        3. calls the selected MCP tool
        4. returns plain text
        """
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.call_tool(
                    tool_name,
                    arguments=arguments,
                )

                return self._extract_text(result)

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        Synchronous wrapper.

        Your LangChain Agent is synchronous, so it calls this normal method.
        Internally, this method runs the async MCP call.
        """
        try:
            existing_loop = asyncio.get_event_loop()
        except RuntimeError:
            existing_loop = None

        try:
            return asyncio.run(
                self._call_tool_async(
                    tool_name=tool_name,
                    arguments=arguments,
                )
            )
        finally:
            if existing_loop is not None and not existing_loop.is_closed():
                asyncio.set_event_loop(existing_loop)



    def vector_search(self, question: str) -> str:
        """
        Calls the MCP server tool named: vector_search

        Use this for semantic paper search:
        - topic search
        - recommendations
        - similarity search
        """
        return self.call_tool(
            tool_name="vector_search",
            arguments={"question": question},
        )

    def cypher_search(self, question: str) -> str:
        """
        Calls the MCP server tool named: cypher_search

        Use this for structured graph questions:
        - authors
        - categories
        - paper IDs
        - DOI
        - versions
        - exact metadata
        """
        return self.call_tool(
            tool_name="cypher_search",
            arguments={"question": question},
        )
    
    def general_paper_chat(self, question: str) -> str:
        return self.call_tool(
            tool_name="general_paper_chat",
            arguments={"question": question},
    )


    def graph_statistics(self, question: str) -> str:
        return self.call_tool(
            tool_name="graph_statistics",
            arguments={"question": question},
    )

    def _extract_text(self, result) -> str:
        """
        MCP returns content blocks.
        LangChain tools need a simple string.
        This method converts MCP result content into plain text.
        """
        if not hasattr(result, "content"):
            return str(result)

        output_parts = []

        for item in result.content:
            if hasattr(item, "text"):
                output_parts.append(item.text)
            else:
                output_parts.append(str(item))

        return "\n".join(output_parts)
    # RESEOURCE READING
    async def _read_resource_async(self, uri: str) -> str:
        """
        Internal async method to read an MCP resource.
        Example:
        - neo4j://schema
        """
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.read_resource(uri)

                return self._extract_resource_text(result)

    def read_resource(self, uri: str) -> str:
        """
        Synchronous wrapper for reading an MCP resource.
        """
        try:
            existing_loop = asyncio.get_event_loop()
        except RuntimeError:
            existing_loop = None

        try:
            return asyncio.run(
                self._read_resource_async(uri)
            )
        finally:
            if existing_loop is not None and not existing_loop.is_closed():
                asyncio.set_event_loop(existing_loop)

    def get_neo4j_schema(self, _: str = "") -> str:
        """
        Read the Neo4j schema resource from the MCP server.

        The unused argument exists because LangChain tools often pass
        a string input to functions.
        """
        return self.read_resource("neo4j://schema")

    def _extract_resource_text(self, result) -> str:
        """
        MCP resource results usually return contents blocks.
        Convert them into plain text.
        """
        if not hasattr(result, "contents"):
            return str(result)

        output_parts = []

        for item in result.contents:
            if hasattr(item, "text"):
                output_parts.append(item.text)
            else:
                output_parts.append(str(item))

        return "\n".join(output_parts)
    def get_neo4j_schema_tool(self, _: str = "") -> str:
        return self.read_resource("neo4j://schema")






    def get_chat_history(
    self,
    session_id: str,
    ) -> list[dict[str, str]]:
        
        result = self.call_tool(
            "get_chat_history",
            {
                "session_id": session_id,
            },
        )

        try:
            data = json.loads(result)
        except (TypeError, json.JSONDecodeError):
            return []

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            return data.get("messages", [])

        return []

    def add_chat_message(
                        self,
                        session_id: str,
                        role: str,
                        content: str,
                        user_id: str | None = None,
                        ) -> str:
                        return self.call_tool(
                            "add_chat_message",
                            {
                                "session_id": session_id,
                                "role": role,
                                "content": content,
                                "user_id": user_id,
                            },
                        )


    def clear_chat_history(
                        self,
                        session_id: str,
                        ) -> str:
                            return self.call_tool(
                                "clear_chat_history",
                                {
                                    "session_id": session_id,
                                },
                            )

    def get_user_preferences(self, user_id: str, ) -> str:
            return self.call_tool(tool_name="get_user_preferences", arguments={ "user_id": user_id, },)

    
    def save_user_preferences(self, user_id: str,preference: str, category: str,) -> str:
            return self.call_tool(tool_name="get_user_preferences", arguments={ "user_id": user_id, preference: str, category: str},)



    def arvix_paper_chat(self, query: str, max_results: int = 2) -> str:
        return self.call_tool(
            tool_name="arvix_paper_chat",
            arguments={"query": query, "max_results": max_results},
        )