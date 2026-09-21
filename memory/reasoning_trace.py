import json
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from mcp_client.mcp_client import MCPClient




class ReasoningTraceCallback(BaseCallbackHandler):
    """LangChain callback for reasoning traces."""

    def __init__(self, mcp_client: MCPClient, user_id: str,  session_id: str,):
        self.mcp_client = mcp_client
        self.user_id = user_id
        self.session_id = session_id
        self.trace_id = None
        self.current_step_id = None
        self.tool_steps: dict[UUID, dict] = {}
        self.root_run_id: UUID | None = None


    @staticmethod
    def _parse_response(response: str) -> dict:
        try:
            return json.loads(response)
        except (TypeError, json.JSONDecodeError):
            return {}

    def on_chain_start(self, serialized,inputs, *, run_id, parent_run_id=None, **kwargs):
        # Only create a trace for the top-level chain.
        if parent_run_id is not None or self.trace_id is not None:
            return

        messages = inputs.get("messages", [])
        task = str(messages[-1].content if messages else "")[:2000]

        response = self.mcp_client.start_reasoning_trace(
            session_id=self.session_id,
            user_id=self.user_id,
            task=task,
        )
        data = self._parse_response(response)
        self.trace_id = data.get("trace_id")
        self.root_run_id = run_id




    def on_tool_start(
        self,
        serialized,
        input_str,
        *,
        run_id,
        **kwargs,
    ):
        if not self.trace_id:
            return

        tool_name = serialized.get("name", "unknown")

        response = self.mcp_client.add_reasoning_step(
            trace_id=self.trace_id,
            thought=f"Calling tool: {tool_name}",
            action=tool_name,
        )

        data = self._parse_response(response)
        step_id = data.get("step_id")

        if step_id:
            self.tool_steps[run_id] = {
                "step_id": step_id,
                "tool_name": tool_name,
                "input": input_str,
            }

    def on_tool_end(
        self,
        output,
        *,
        run_id,
        **kwargs,
    ):
        tool = self.tool_steps.pop(run_id, None)

        if not tool:
            return

        self.mcp_client.record_reasoning_tool_call(
            step_id=tool["step_id"],
            tool_name=tool["tool_name"],
            arguments={
                "input": str(tool["input"])[:2000]
            },
            result={
                "output": str(output)[:5000]
            },
            status="success",
        )



    def on_chain_end(
        self,
        outputs,
        *,
        run_id,
        **kwargs,
    ):
        if run_id != self.root_run_id or not self.trace_id:
            return

        messages = outputs.get("messages", [])
        outcome = str(messages[-1].content if messages else "Chain completed")[:5000]

        self.mcp_client.complete_reasoning_trace(
            trace_id=self.trace_id,
            outcome=outcome,
            success=True,
        )

        self._reset()

    def _reset(self):
        self.root_run_id = None
        self.trace_id = None
        self.tool_steps.clear()
