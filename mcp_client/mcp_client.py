import asyncio
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os


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
            "MCP_SERVER_URL",
            "http://host.docker.internal:8000/sse"
        )

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
        return asyncio.run(
            self._call_tool_async(
                tool_name=tool_name,
                arguments=arguments,
            )
        )

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
        return asyncio.run(
            self._read_resource_async(uri)
        )

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

