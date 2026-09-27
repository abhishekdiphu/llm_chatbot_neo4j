
from llm import llm , embeddings, cus_llm
from graph import graph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from utils import ExtractedPreference
from mcp_client.mcp_client import MCPClient
from memory.message_history import Neo4jChatMessageHistory
from datetime import datetime
from tools.long_memory import UserPreferencesTool, SavePreferenceTool
from utils import ExtractedPreferences
from memory.reasoning_trace import ReasoningTraceCallback
from agents.prompts import REACT_PROMPT_CREATE_AGENT
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel





class Answer(BaseModel):
    summary: str
    

class Planner:
    def __init__(self, llm=None, cus_llm=cus_llm, graph=graph, embeddings=embeddings, user_id="abhishek"):    

        self.user_id = user_id.strip()
        self.session_id = f"user-{self.user_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.graph = graph
        self.embeddings = embeddings
        self.mcp_client = MCPClient()
        # Convert it to an LLMProvider for the memory client.
       

        
        self.agent_prompt_local = REACT_PROMPT_CREATE_AGENT
        #self.agent_prompt_local = PromptTemplate.from_template(REACT_PROMPT)

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
        self.agent_prompt =  self.agent_prompt_local
        self.agent = create_agent(model = cus_llm,  
                                  tools=self.tools, 
                                  system_prompt= self.agent_prompt)
        self.message_history = None
        self.memory = None
        self.agent_executor = None
        self.callback = ReasoningTraceCallback( mcp_client=self.mcp_client,
                                                user_id=self.user_id,
                                                session_id=self.session_id)
        


    def generate_response(self, user_input):
        """
        Create a handler that calls the Conversational agent
        and returns a response to be rendered in the UI
        """
    
        if self.message_history is None:
            raise RuntimeError("planner not properly initilaized")
        history = self.message_history.messages[-3:]
        try:
            response = self.agent.invoke({"messages": [*history, HumanMessage(content=user_input)]},
                                          config={"callbacks": [self.callback],
                                                  "recursion_limit": 6})
            answer = response["messages"][-1].content
            self.message_history.add_message(HumanMessage(content=user_input))
            self.message_history.add_message(AIMessage(content=answer))

            return answer

        except Exception as e:
            return str(e)



    async def initialize(self):

        status= await self.mcp_client.test_connection()
        if status:

            # Create message history
            self.message_history = Neo4jChatMessageHistory( mcp_client=self.mcp_client,
                                                            session_id=self.session_id,
                                                            user_id=self.user_id,
                                                        )
        else:
            print("mcp connection failed")






    def finalize_discussion(self) -> dict:
        """
        long term memory usage
        
        """
        records = self.mcp_client.get_chat_history( session_id=self.session_id)

        if not records:
            return {"saved_count": 0, "preferences": [],}

        #conversation = "\n".join(f"{record.get('role', 'unknown')}: "f"{record.get('content', '')}" for record in records)


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
            """delete short term message after usage"""
            
            if self.message_history:
                try:
                    self.message_history.clear(session_id=self.session_id)
                except Exception as e:
                    return {"status": "error", "message": str(e)}

            else:
                return {"status": "error", "message": "No message history to clear."}
            



    
