
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from mcp_client.mcp_client import MCPClient


class SavePreferenceInput(BaseModel):

    preference: str = Field(description="The preference to save")
    category: str = Field(description="Category: category, domain, subject,author etc.")


class UserPreferencesTool(BaseTool):
    name: str = "get_user_preferences"

    description: str = """
    Get long-term preferences stored for the current user.

    Use this before making personalized paper recommendations
    or when the user refers to their existing preferences.
    """

    mcp_client: MCPClient
    user_id: str

    def _run(self) -> str:
        return self.mcp_client.get_user_preferences(user_id=self.user_id)





class SavePreferenceTool(BaseTool):
    name: str = "save_user_preference"
    description: str = """
    Save a long-term user preference.
    Use this when the user explicitly says they like,
    dislike, prefer, avoid, or are interested in something.
    Do not use it for temporary questions or assumptions.
    """
    args_schema: type[BaseModel] = SavePreferenceInput
    mcp_client: MCPClient
    user_id: str

    def _run(self, preference: str, category: str) -> str:
        return self.mcp_client.save_user_preferences(user_id=self.user_id, preference=preference, category=category)