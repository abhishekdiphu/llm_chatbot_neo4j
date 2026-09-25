


import re
from typing import Any
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (AIMessage,
                                    BaseMessage,
                                    HumanMessage,
                                    SystemMessage,
                                    ToolMessage)

from langchain_core.outputs import ChatGeneration, ChatResult

class QwenReActChatModel(BaseChatModel):
    qwen: Any
    @property

    def _llm_type(self) -> str:
        return "local-qwen-react"


    def bind_tools(self, tools, **kwargs)-> str:
        return self.bind(tools=list(tools), **kwargs)


    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        tools = kwargs.get("tools", [])
        tool_by_name = {tool.name: tool for tool in tools}
        tool_instructions = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
        allowed_names = ", ".join(tool_by_name)


        incoming_system_text = "\n\n".join(str(message.content) for message in messages if isinstance(message, SystemMessage))

        combined_system_prompt = f"""
                                    You are an academic assistant with access to tools.

                                    {incoming_system_text}

                                    For questions that require database or scientific-paper information,
                                    you must call an appropriate tool. Do not answer from memory and do not
                                    invent tool results.

                                    Available tools:

                                    {tool_instructions}

                                    Valid tool names:

                                    {allowed_names}

                                    When calling a tool, output exactly two lines:

                                    Action: one_valid_tool_name

                                    Action Input: input to the Action 

                                    Rules:

                                    - Replace one_valid_tool_name with one name from the valid tool names.
                                    - Do not output the literal text one_valid_tool_name. this is the function name
                                    - Do not write anything before Action.
                                    - Do not write anything after Action Input.
                                    - Do not generate the tool result.
                                    - Do not include Observation or Answer with a tool request.
                                    - For schema_of_databse_lookup, use an empty Action Input.
                                    - For graph_statistics, use an empty Action Input.
                                    - For search tools, pass the input to the Action as Action Input.these are the arguments for Action.

                                    If a tool has already executed and its result is included in the conversation
                                    as an Observation.You must now answer the user's original question using only 
                                    that Observation.

                                    Do not request another tool.
                                    Do not output Action or Action Input.
                                    Do not repeat the Observation verbatim.

                                    After the application returns a tool result, answer using:
                                    Answer: answer based only on the tool result
                                    """.strip()


        conversation = [{"role": "system",
                         "content": combined_system_prompt}
                    ]


        
        for message in messages:
            if isinstance(message, SystemMessage):
                continue

            elif isinstance(message, HumanMessage):
                conversation.append({
                    "role": "user",
                    "content": str(message.content),
                })

            elif isinstance(message, AIMessage):
                if message.tool_calls:
                    tool_call = message.tool_calls[0]
                    arguments = tool_call.get("args", {})

                    tool_input = (
                        next(iter(arguments.values()))
                        if arguments
                        else ""
                    )

                    conversation.append({
                        "role": "assistant",
                        "content": (
                            f"Action: {tool_call['name']}\n"
                            f"Action Input: {tool_input}"
                        ),
                    })

                elif message.content:
                    conversation.append({
                        "role": "assistant",
                        "content": str(message.content),
                    })

            elif isinstance(message, ToolMessage):
                conversation.append({
                    "role": "user",
                    "content": (
                        f"Observation from tool {message.name or 'unknown'}:\n"
                        f"{message.content}\n\n"
                        "The tool has completed successfully. "
                        "Answer the original question using this observation. "
                        "Do not call another tool."
                    ),
                })



        output = self.qwen.invoke(conversation,max_new_tokens=512)

        action_pattern = re.compile(r"Action:\s*"
                                    r"(?:tool_name\s*:?\s*)?"
                                    r"([A-Za-z_][A-Za-z0-9_-]*)"
                                    r"\s+Action\s*Input:\s*"
                                    r"(.*?)"
                                    r"(?=\s+(?:Thought|Action|Observation|Answer|Final\s+Answer):|$)",
                                    flags=re.IGNORECASE | re.DOTALL)

      
        matches = list(action_pattern.finditer(output))
        print(matches)
        valid_action = None
        
        for match in reversed(matches):
            candidate_name = match.group(1).strip()

            if candidate_name in tool_by_name:
                valid_action = match
                break

        if valid_action is not None:
            name = valid_action.group(1).strip()
            tool_input = valid_action.group(2).strip()

            input_fields = tool_by_name[name].args
            input_key = next(iter(input_fields))
            print("Parsed tool name:", repr(name))
            print("Parsed tool input:", repr(tool_input))
         
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": name,
                        "args": {input_key: tool_input},
                        "id": f"call_{uuid4().hex}",
                    }
                ],
            )
        else:
            answer = re.split(r"(?:Final\s+Answer|Answer):", output, maxsplit=1, flags=re.IGNORECASE)[-1].strip()
            message = AIMessage(content=answer)



        

        final_answer= ChatResult(generations=[ChatGeneration(message=message)])

        return final_answer