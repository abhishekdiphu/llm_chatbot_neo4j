import asyncio
from collections.abc import Sequence
from typing import Any

from neo4j_agent_memory.llm import ChatMessage, Completion
from neo4j_agent_memory.llm.errors import ProviderServiceError, ProviderTimeoutError


class QwenMemoryProvider:
    """Neo4j-agent-memory provider for the locally loaded Qwen model."""

    def __init__(self, qwen: Any, model: str = "local/Qwen2.5-3B-Instruct", default_max_tokens: int = 512) -> None:
        self.qwen = qwen
        self.model = model
        self.default_max_tokens = default_max_tokens

        # Prevent simultaneous generation calls against the same model.
        self._generation_lock = asyncio.Lock()

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> Completion:



        
        conversation = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]

        generation_limit = max_tokens or self.default_max_tokens

        try:
            async with self._generation_lock:
                generation = asyncio.to_thread(
                    self.qwen.invoke,
                    conversation,
                    max_new_tokens=generation_limit,
                )

                if timeout is not None:
                    output = await asyncio.wait_for(
                        generation,
                        timeout=timeout,
                    )
                else:
                    output = await generation

        except TimeoutError as error:
            raise ProviderTimeoutError(
                f"Local Qwen generation exceeded {timeout} seconds."
            ) from error
        except Exception as error:
            raise ProviderServiceError(
                f"Local Qwen generation failed: {error}"
            ) from error

        # The current Qwen implementation is deterministic and therefore
        # ignores temperature. Apply textual stop sequences after generation.
        if stop:
            stop_positions = [
                output.find(sequence)
                for sequence in stop
                if sequence and sequence in output
            ]

            if stop_positions:
                output = output[:min(stop_positions)]

        return Completion(
            content=output.strip(),
            model=self.model,
            finish_reason="stop",
        )