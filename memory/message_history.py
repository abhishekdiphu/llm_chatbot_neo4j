from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from neo4j_agent_memory import MemoryClient, MemorySettings
import logging  

logger = logging.getLogger(__name__)        


class Neo4jChatMessageHistory(BaseChatMessageHistory):
    """LangChain message history backed by neo4j-agent-memory."""

    def __init__(
        self,
        memory_client: MemoryClient,
        session_id: str,
        user_id: str | None = None,
    ):
        self.memory_client = memory_client
        self.session_id = session_id
        self.user_id = user_id

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve all messages for this session."""
        import asyncio

        # Run async method synchronously
        loop = asyncio.get_event_loop()
        conversation = loop.run_until_complete(
            self.memory_client.short_term.get_conversation(
                session_id=self.session_id,
            )
        )

        result = []
        for msg in conversation.messages:
            if msg.role == "user":
                result.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                result.append(AIMessage(content=msg.content))

        return result

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the session."""
        import asyncio

        role = "user" if isinstance(message, HumanMessage) else "assistant"

        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            self.memory_client.short_term.add_message(
                session_id=self.session_id,
                role=role,
                content=message.content,
                metadata={"user_id": self.user_id} if self.user_id else None,
            )
        )

    def clear(self) -> None:
        """Clear all messages in the session."""
        import asyncio

        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            self.memory_client.short_term.delete_session(
                session_id=self.session_id,
            )
        )


# Async version for async chains
class AsyncNeo4jChatMessageHistory(BaseChatMessageHistory):
    """Async LangChain message history backed by neo4j-agent-memory."""

    def __init__(
        self,
        memory_client: MemoryClient,
        session_id: str,
        user_id: str | None = None,
    ):
        self.memory_client = memory_client
        self.session_id = session_id
        self.user_id = user_id
        self._messages: List[BaseMessage] = []

    async def aget_messages(self) -> List[BaseMessage]:
        """Async retrieve all messages."""
        conversation = await self.memory_client.short_term.get_conversation(
            session_id=self.session_id,
        )

        result = []
        for msg in conversation.messages:
            if msg.role == "user":
                result.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                result.append(AIMessage(content=msg.content))

        return result

    async def aadd_message(self, message: BaseMessage) -> None:
        """Async add a message."""
        role = "user" if isinstance(message, HumanMessage) else "assistant"

        await self.memory_client.short_term.add_message(
            session_id=self.session_id,
            role=role,
            content=message.content,
            metadata={"user_id": self.user_id} if self.user_id else None,
        )

    async def aclear(self) -> None:
        """Async clear messages."""
        await self.memory_client.short_term.delete_session(
            session_id=self.session_id,
        )

    # Sync methods that wrap async
    @property
    def messages(self) -> List[BaseMessage]:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.aget_messages())

    def add_message(self, message: BaseMessage) -> None:
        import asyncio
        try:
            asyncio.get_event_loop().run_until_complete(self.aadd_message(message))
        except Exception as e:
            # Log but don't fail the chain
            logger.warning(f"Failed to save message to memory: {e}")
    def clear(self) -> None:
        import asyncio
        asyncio.get_event_loop().run_until_complete(self.aclear())