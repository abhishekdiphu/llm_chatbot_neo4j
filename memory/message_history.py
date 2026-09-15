from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from neo4j_agent_memory import MemoryClient, MemorySettings
import logging  

logger = logging.getLogger(__name__)    
from mcp_client.mcp_client import MCPClient    



class Neo4jChatMessageHistory(BaseChatMessageHistory):
    """
    Neo4j history accessed through MCP.
    """

    def __init__(self, mcp_client: MCPClient, session_id: str, user_id: str | None = None,):

        self.mcp_client = mcp_client
        self.session_id = session_id
        self.user_id = user_id

    @property
    def messages(self) -> List[BaseMessage]:
        records = self.mcp_client.get_chat_history( session_id=self.session_id,)

        result = []

        for message in records:
            role = message.get("role")
            content = message.get("content", "")

            if role == "user":
                result.append(
                    HumanMessage(content=content)
                )
            elif role == "assistant":
                result.append(
                    AIMessage(content=content)
                )

        return result

    def add_message(
        self,
        message: BaseMessage,
    ) -> None:
        role = (
            "user"
            if isinstance(message, HumanMessage)
            else "assistant"
        )

        self.mcp_client.add_chat_message(
            session_id=self.session_id,
            role=role,
            content=str(message.content),
            user_id=self.user_id,
        )

    def clear(self) -> None:
        self.mcp_client.clear_chat_history(
            session_id=self.session_id,
        )



