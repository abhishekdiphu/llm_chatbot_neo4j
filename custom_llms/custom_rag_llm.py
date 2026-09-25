import re
from typing import Any
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult




class QwenRagChatModel(BaseChatModel):
    """Plain chat model for RAG, Cypher generation, and memory."""

    qwen: Any

    @property
    def _llm_type(self) -> str:
        return "local-qwen-chat"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs, ) -> ChatResult:
        conversation = self._convert_messages(messages)
        output = self.qwen.invoke(conversation)
        generated_output = ChatResult(generations=[ChatGeneration(message=AIMessage(content=output.strip()))])
        return generated_output
    

    def _convert_messages(self, messages: list[BaseMessage]) -> list[dict[str, str]]:
        conversation = []

        for message in messages:
                if isinstance(message, SystemMessage):
                    role = "system"
                elif isinstance(message, HumanMessage):
                    role = "user"
                elif isinstance(message, AIMessage):
                    role = "assistant"
                elif isinstance(message, ToolMessage):
                    role = "tool"
                else:
                    continue

                conversation.append({"role": role, "content": str(message.content),})

        return conversation