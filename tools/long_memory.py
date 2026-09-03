
import json

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field
from neo4j_agent_memory import MemoryClient, MemorySettings



class UserPreferencesTool(BaseTool):
    name: str = "get_user_preferences"
    description: str = """
    Get stored preferences for the current user.
    Returns domain preferences, author, paper preferences.
    """
    memory_client: MemoryClient
    user_id: str

    def _run(self) -> str:
        import asyncio

        prefs = asyncio.get_event_loop().run_until_complete(
            self.memory_client.long_term.get_preferences(
                user_id=self.user_id,
            )
        )

        by_category = {}
        for p in prefs:
            if p.category not in by_category:
                by_category[p.category] = []
            by_category[p.category].append(p.preference)

        return json.dumps(by_category)


class SavePreferenceInput(BaseModel):
    preference: str = Field(description="The preference to save")
    category: str = Field(description="Category: category, domain, subject,author etc.")


class SavePreferenceTool(BaseTool):
    name: str = "save_user_preference"
    description: str = """
    Save a new user preference learned during the conversation.
    Use when customer expresses a like, dislike, or preference.
    """
    args_schema: type[BaseModel] = SavePreferenceInput
    memory_client: MemoryClient
    user_id: str

    def _run(self, preference: str, category: str) -> str:
        import asyncio

        asyncio.get_event_loop().run_until_complete(
            self.memory_client.long_term.add_preference(
                preference=preference,
                category=category,
                confidence=0.85,
            )
        )

        return f"Saved preference: {category} - {preference}"