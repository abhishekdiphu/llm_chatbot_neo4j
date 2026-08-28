from langchain.callbacks.base import BaseCallbackHandler,AsyncCallbackHandler
from neo4j_agent_memory import MemoryClient
from langchain_core.messages import BaseMessage

def extract_text(value) -> str:
    """Safely convert LangChain callback inputs/outputs to text."""

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, BaseMessage):
        return str(value.content)

    if isinstance(value, dict):
        for key in ["input", "output", "text", "question"]:
            if key in value:
                return extract_text(value[key])

        return str(value)

    return str(value)



class ReasoningTraceCallback(AsyncCallbackHandler):
    """Record LangChain execution as reasoning traces."""

    def __init__(self, memory_client: MemoryClient, user_id: str , session_id: str):
        self.memory_client = memory_client
        self.user_id = user_id
        self.session_id = session_id
        self.trace_id = None
        self.current_step_id = None




    async def on_chain_start(self, serialized, inputs, **kwargs):
        """Start trace when chain begins."""

        """import asyncio

        trace = asyncio.get_event_loop().run_until_complete(
            self.memory_client.reasoning.start_trace(
                task=str(inputs.get("input", ""))[:200],
                user_id=self.user_id,
                metadata={"chain_type": serialized.get("name", "unknown")},
            )
        )
        self.trace_id = trace.id
        """
        if kwargs.get("parent_run_id") is not None:
            return

        # serialized can legitimately be None
        serialized = serialized or {}

        task = extract_text(inputs)[:200]

        chain_name = serialized.get("name") or "AgentExecutor"

        trace = await self.memory_client.reasoning.start_trace(
            session_id=self.session_id,
            task=task,
            metadata={
                "chain_type": chain_name,
                "user_id": self.user_id,
            },
        )

        self.trace_id = trace.id






    async def on_tool_start(self, serialized, input_str, **kwargs):
        """Record tool invocation."""


        """import asyncio

        if not self.trace_id:
            return

        tool_name = serialized.get("name", "unknown")

        step = asyncio.get_event_loop().run_until_complete(
            self.memory_client.reasoning.add_step(
                trace_id=self.trace_id,
                description=f"Using tool: {tool_name}",
                reasoning=f"Input: {input_str[:100]}",
            )
        )
        self.current_step_id = step.id
        """
        if not self.trace_id:
            return

        serialized = serialized or {}

        self.current_tool_name = (
            serialized.get("name")
            or "unknown"
        )

        step = await self.memory_client.reasoning.add_step(
            trace_id=self.trace_id,
            description=f"Using tool: {self.current_tool_name}",
            reasoning=f"Input: {extract_text(input_str)[:100]}",
        )

        self.current_step_id = step.id


    async def on_chat_model_start(
        self,
        serialized,
        messages,
        **kwargs,
        ):

        pass

    async def on_llm_start(
        self,
        serialized,
        prompts,
        **kwargs,
        ):
        pass


    async def on_tool_end(self, output, **kwargs):
        """Record tool result."""
        """
        import asyncio

        if not self.current_step_id:
            return

        asyncio.get_event_loop().run_until_complete(
            self.memory_client.reasoning.add_tool_call(
                step_id=self.current_step_id,
                tool_name=kwargs.get("name", "unknown"),
                arguments={},
                result={"output": str(output)[:500]},
                success=True,
            )
        )
        """
        if not self.current_step_id:
            return

        await self.memory_client.reasoning.add_tool_call(
            step_id=self.current_step_id,
            tool_name=self.current_tool_name or "unknown",
            arguments={},
            result={"output": extract_text(output)[:500]},
            success=True,
        )

        self.current_step_id = None
        self.current_tool_name = None


    async def on_chain_end(self, outputs, **kwargs):
        """Complete trace when chain finishes."""
        """
        import asyncio

        if not self.trace_id:
            return

        asyncio.get_event_loop().run_until_complete(
            self.memory_client.reasoning.complete_trace(
                trace_id=self.trace_id,
                outcome="success",
                result={"output": str(outputs)[:500]},
            )
        )
        """
        # Ignore nested chain completions
        if kwargs.get("parent_run_id") is not None:
            return

        if not self.trace_id:
            return

        await self.memory_client.reasoning.complete_trace(
            trace_id=self.trace_id,
            outcome="success",
            result={
                "output": extract_text(outputs)[:500]
            },
        )

        self.trace_id = None

    async def on_chain_error(self, error, **kwargs):
        """Record chain failure."""
        """
        import asyncio

        if not self.trace_id:
            return

        asyncio.get_event_loop().run_until_complete(
            self.memory_client.reasoning.complete_trace(
                trace_id=self.trace_id,
                outcome="failure",
                error=str(error),
            )
        )
        """
        if kwargs.get("parent_run_id") is not None:
            return

        if not self.trace_id:
            return

        await self.memory_client.reasoning.complete_trace(
            trace_id=self.trace_id,
            outcome="failure",
            error=str(error),
        )

        self.trace_id = None

