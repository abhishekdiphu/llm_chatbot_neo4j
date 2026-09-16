from openai import chat

from llm import llm , embeddings, cus_llm
from graph import graph

from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

from langchain.tools import Tool

from langchain.agents import AgentExecutor, create_react_agent

from langchain import hub
from langchain_core.prompts import PromptTemplate

from utils import ExtractedPreference, get_session_id
from mcp_client.mcp_client import MCPClient

import os
from memory.message_history import Neo4jChatMessageHistory

from datetime import datetime

from langchain.memory import ConversationBufferWindowMemory

from tools.long_memory import UserPreferencesTool, SavePreferenceTool
from utils import ExtractedPreferences


"""
┌─────────────────────────────────────────────────────┐
│                   LangChain Agent                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │    Tools    │  │   Memory    │  │    Chain    │ │
│  │             │  │  Interface  │  │             │ │
│  └──────┬──────┘  └──────┬──────┘  └─────────────┘ │
│         │                │                          │
│         ▼                ▼                          │
│  ┌─────────────────────────────────────────────┐   │
│  │        Neo4jAgentMemory Adapter             │   │
│  │  • Implements BaseChatMessageHistory        │   │
│  │  • Provides entity/preference tools         │   │
│  │  • Callbacks for reasoning traces           │   │
│  └──────────────────────┬──────────────────────┘   │
└─────────────────────────┼───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│               Neo4j Context Graph                   │
│  Messages ──── Entities ──── Preferences            │
│      │            │              │                  │
│      └────────────┴──────────────┘                  │
│            Reasoning Traces                         │
└─────────────────────────────────────────────────────┘


User
 │
 │ "papers about graph sparsity"
 ▼
Embedding model
 │
 ▼
query embedding
 │
 ▼
chunk_embedding_index
 │
 │ vector similarity
 ▼
(:Chunk)
 │
 │ graph traversal
 ▼
(:Paper) ──► (:Author)
 │
 └────────► (:Category)
 │
 ▼
PAPER_RETRIEVAL_QUERY
 │
 ▼
Rich retrieval result
 │
 ▼
LLM
 │
 ▼
Answer

"""



class Agent():
    def __init__(self, llm=None, cus_llm=llm, graph=graph, embeddings=embeddings, user_id="abhishek"):    
        self.user_id = user_id
        self.session_id = f"user-{self.user_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.graph = graph
        self.embeddings = embeddings
        self.mcp_client = MCPClient()
        # Convert it to an LLMProvider for the memory client.
       

        
        self.agent_prompt_local = PromptTemplate.from_template("""
        You are a academic expert providing information about scientific papers.
        Be as helpful as possible and return as much information as possible.
        Do not answer any questions using your pre-trained knowledge, only use 
        the information provided in the context.

        Do not answer any questions that do not relate to scientific papers.
                                                    

        TOOLS:
        ------

        You have access to the following tools:

        {tools}

        To use a tool, please use the following format:

        ```
        Thought: Do I need to use a tool? Yes
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
         ... (this Thought/Action/Action Input/Observation can repeat N times)
        ```

        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

        ```
        Thought: Do I need to use a tool? No
        Final Answer: the final answer to the original input question
        ```

        Begin!

        Previous conversation history:
        {chat_history}

        Question: {input}
        Thought:{agent_scratchpad}
        """)
        #New input: {input}
        #{agent_scratchpad}
                
        




       

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a scientific paper expert providing information about academic papers."),
                ("human", "{input}"),
            ])

        

        # -------------------------
        # longMemory tools
        # -------------------------
        self.user_preferences_tool = UserPreferencesTool( mcp_client=self.mcp_client, user_id=self.user_id)

        self.save_preference_tool = SavePreferenceTool( mcp_client=self.mcp_client, user_id=self.user_id,)


        #-------------------------
        
        # LangChain tools.
        # These tools call MCP client methods, not VectorTool/CypherTool directly.
        self.tools = [
            Tool.from_function(
                name="paper_vector_search",
                description=(
                    "Use     this tool for semantic scientific paper search. "
                    "Best for finding papers by topic, meaning, similarity, "
                    "or recommendations."
                ),
                func=self.mcp_client.vector_search,
            ),
            Tool.from_function(
                name="paper_cypher_search",
                description=(
                    "Use this tool for structured scientific paper questions. "
                    "Best for authors, categories, paper IDs, DOI, versions, "
                    "update dates, and exact metadata."
                ),
                func=self.mcp_client.cypher_search,
            ),
       
            Tool.from_function(
                name="schema_of_databse_lookup",
                description=(
                    "Use this tool to look up the schema of the Neo4j database."
                ),
                func=self.mcp_client.get_neo4j_schema_tool,
            ),
            Tool.from_function(
                name="graph_statistics",
                description=(
                    "Use this tool to get statistics about the Neo4j graph, such as number of nodes, relationships"
                ),
                func=self.mcp_client.graph_statistics,
            ),
            Tool.from_function(
                            name="arvix_paper_chat",
                            description=("Use this tool to search for papers on arXiv.It is a fallback tool for external paper search."),
                            func=self.mcp_client.arvix_paper_chat,
                            
                        ),
         
    ]



        # Create the agent
        self.agent_prompt =  self.agent_prompt_local#hub.pull("hwchase17/react-chat")


        """
        agent start ---->

        1. Step 1: LLM Decision-Making Process

        a. The LLM suggests the ‘Action’ and ‘Action Input’ during the decision-making process.

        
        
        
        Step 2: Agent Response

        a. The agent then responds by launching the recommended tool (Action).

        b. The agent supplies the input parameters (Action Input) to the function.


        
        
        Step 3: Observation

        a. The function’s output (tool) is presented as an ‘Observation’ by the agent to the LLM.

        
        
        Step 4: LLM Feedback Loop

        a. Subsequently, the LLM provides a ‘Thought’.

        b. The LLM suggests the next series of ‘Action’ & ‘Action Input’ for the agent.

        


        Step 5: Repetition

        a. This entire procedure repeats in a cycle until the LLM provides a ‘Final Answer’ response.

        Step 6: Communication to User

        a. The final response is communicated to the user.



        """
        self.agent = create_react_agent(cus_llm, self.tools, self.agent_prompt)

        # Usage
        

        self.message_history = None
        self.memory = None
        self.agent_executor = None
        self.callback = None
        

        


        # Create a handler to call the agent

        #self.chat_agent = RunnableWithMessageHistory(
        #    self.agent_executor,
        #    #self.get_memory,
        #    input_messages_key="input",
        #    history_messages_key="chat_history",
        #)

    # Create chat history callback
    #def get_memory(self, session_id):
    #    return Neo4jChatMessageHistory(session_id=session_id, graph=self.graph)

    def generate_response(self, user_input):
        """
        Create a handler that calls the Conversational agent
        and returns a response to be rendered in the UI
        """
    

        try:
            response = self.agent_executor.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": self.session_id},
                #"callbacks": [self.callback],
                })
            
            return response['output']


        
        except Exception as e:

            return str(e)

            #raise


    async def initialize(self):
       

        # Create message history
        self.message_history = Neo4jChatMessageHistory(
        mcp_client=self.mcp_client,
        session_id=self.session_id,
        user_id=self.user_id,
    )

        # Create memory
        self.memory = ConversationBufferWindowMemory(
            chat_memory=self.message_history,
            k=3,  # Last 3 messages
            memory_key="chat_history",
            return_messages=True,
        )

        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            max_iterations=3
            )

    async def close(self):
        pass

    def finalize_discussion(self) -> dict:
        records = self.mcp_client.get_chat_history( session_id=self.session_id)

        if not records:return {"saved_count": 0, "preferences": [],}

        conversation = "\n".join(
            f"{record.get('role', 'unknown')}: "
            f"{record.get('content', '')}"
            for record in records
        )


                # Temporary replacement for LLM-based extraction.
        extracted = ExtractedPreferences(
            preferences=[
                        ExtractedPreference(
                            category="subject",
                            preference=(
                                "The user prefers papers about "
                                "knowledge graphs."),),
                        ExtractedPreference(
                            category="format",
                            preference=(
                                "The user prefers concise explanations."),),
                        ])

        saved = []
        for item in extracted.preferences:
            result = self.save_preference_tool.invoke({"preference": item.preference, "category": item.category})
            saved.append({"category": item.category, "preference": item.preference, "result": result,})
            
        return {"saved_count": len(saved),"preferences": saved,}



    def delete_conversation(self) -> dict:
            if self.message_history:
                try:
                    self.message_history.clear(session_id=self.session_id)
                except Exception as e:
                    return {"status": "error", "message": str(e)}

            else:
                return {"status": "error", "message": "No message history to clear."}
            

