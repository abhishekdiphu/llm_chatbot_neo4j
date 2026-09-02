from openai import chat

from llm import llm , embeddings, cus_llm
from graph import graph

from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
#from langchain_neo4j import Neo4jChatMessageHistory
from langchain.tools import Tool

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from langchain_core.prompts import PromptTemplate
from memory.reasoning_trace import ReasoningTraceCallback
from utils import get_session_id
from mcp_client.mcp_client import MCPClient


from neo4j_agent_memory.integrations.langchain import (
    llm_provider_from_langchain,
)
from neo4j_agent_memory import MemoryClient, MemorySettings,ExtractionConfig
from neo4j_agent_memory.config import Neo4jConfig
import os
from memory.message_history import Neo4jChatMessageHistory, AsyncNeo4jChatMessageHistory
import asyncio
from datetime import datetime
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory

from tools.long_term_memory_tool import UserPreferencesTool, SavePreferenceTool


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
    def __init__(self, llm=llm, cus_llm=cus_llm, graph=graph, embeddings=embeddings, user_id="abhishek"):    
        self.user_id = user_id
        self.session_id = f"user-{self.user_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.graph = graph
        self.embeddings = embeddings
        self.mcp_client = MCPClient()
        # Convert it to an LLMProvider for the memory client.
        self.provider = llm_provider_from_langchain(llm)
        self.mem_settings = MemorySettings(neo4j=Neo4jConfig(
                                            uri=os.environ["NEO4J_URI"],
                                            username=os.environ["NEO4J_USERNAME"],
                                            password=os.environ["NEO4J_PASSWORD"],
                                            database=os.environ["NEO4J_DATABASE"],
                                            ),
                                            extraction=ExtractionConfig(enable_spacy=False,
                                                                         enable_gliner=False),
                                            llm=self.provider,
                                            embedding="openai/text-embedding-3-small",
                                            )

        self.memory_client = MemoryClient(self.mem_settings)




        


        
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
        ```

        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

        ```
        Thought: Do I need to use a tool? No
        Final Answer: [your response here]
        ```

        Begin!

        Previous conversation history:
        {chat_history}

        New input: {input}
        {agent_scratchpad}
        """)





       

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a scientific paper expert providing information about academic papers."),
                ("human", "{input}"),
            ])

        self.arvix_chat = self.chat_prompt | cus_llm | StrOutputParser()

         # -------------------------
        # Memory tools
        # -------------------------
        self.user_preferences_tool = UserPreferencesTool(
            memory_client=self.memory_client,
            user_id=self.user_id,
        )

        self.save_preference_tool = SavePreferenceTool(
            memory_client=self.memory_client,
            user_id=self.user_id,
        )
        # Create a set of tools


        
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
                name="general_paper_chat",
                description=(
                    "Use this For general chat on scientific papers not covered by other tools "
                ),
                func=self.mcp_client.general_paper_chat,
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
            self.user_preferences_tool,
            self.save_preference_tool,
    ]



        # Create the agent
        self.agent_prompt =  self.agent_prompt_local#hub.pull("hwchase17/react-chat")

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
        await self.memory_client.connect()

        # Create message history
        self.message_history = AsyncNeo4jChatMessageHistory(
            memory_client=self.memory_client,
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

        #self.callback = ReasoningTraceCallback(self.memory_client, self.user_id, self.session_id)
        

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            max_iterations=3
            )

    async def close(self):
        await self.memory_client.close()

