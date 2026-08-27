from llm import llm , embeddings
from graph import graph

from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_neo4j import Neo4jChatMessageHistory
from langchain.tools import Tool

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from langchain_core.prompts import PromptTemplate
from utils import get_session_id
from mcp_client.mcp_client import MCPClient






class Agent():
    def __init__(self, llm=llm, graph=graph, embeddings=embeddings):    

        self.graph = graph
        self.embeddings = embeddings
        self.mcp_client = MCPClient()

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





        # Create a movie chat chain


        self.chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a scientific paper expert providing information about academic papers."),
                ("human", "{input}"),
            ])

        self.arvix_chat = self.chat_prompt | llm | StrOutputParser()

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
    ]



        # Create the agent
        self.agent_prompt =  self.agent_prompt_local#hub.pull("hwchase17/react-chat")

        self.agent = create_react_agent(llm, self.tools, self.agent_prompt)

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
            )


        # Create a handler to call the agent

        self.chat_agent = RunnableWithMessageHistory(
            self.agent_executor,
            self.get_memory,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    # Create chat history callback
    def get_memory(self, session_id):
        return Neo4jChatMessageHistory(session_id=session_id, graph=self.graph)

    def generate_response(self, user_input):
        """
        Create a handler that calls the Conversational agent
        and returns a response to be rendered in the UI
        """

        response = self.chat_agent.invoke(
            {"input": user_input},
            {"configurable": {"session_id": get_session_id()}},)

        return response['output']




