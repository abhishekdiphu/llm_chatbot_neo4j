import streamlit as st
from utils import write_message

import os
import asyncio
import logging


os.environ["LANGSMITH_TRACING"] = "false"

@st.cache_resource
def get_agent(user_id):
    # Create a persistent event loop
    from agents.planner_agent import Planner
    loop = asyncio.new_event_loop()
    # Make it the current loop for this thread
    asyncio.set_event_loop(loop)
    agent = Planner(user_id=user_id)
    loop.run_until_complete(agent.initialize())

    return agent, loop


class Chatbot:
    def __init__(self, username:str):

        self.agent = None
        self.loop = None
        self.error_msg = None
        self.user_id = username

        self.logger = logging.getLogger(__name__)
        try:
            self.agent, self.loop = get_agent(self.user_id)
        except Exception:
            self.logger.exception("Agent initialization failed")

            self.error_msg = ("The chatbot is temporarily unavailable. ""Please try again later.")

    # Submit handler
    def handle_submit(self, message):
        """
        Submit handler:

        You will modify this method to talk with an LLM and provide
        context using data from Neo4j.
        """

        # Handle the response
        with st.spinner('Thinking...'):
            # Call the agent
             # Streamlit may execute this on a ScriptRunner thread.
            # Make our persistent loop available in this thread.
            asyncio.set_event_loop(self.loop)
            try:
                response =  self.agent.generate_response(message)
                
            except Exception as e:
                response = f"Error: {str(self.error_msg)}" 

            write_message('assistant', response)
             
            


if __name__ == "__main__":
    # Page Config
    st.set_page_config("Ebert", page_icon=":movie_camera:")


    if "username" not in st.session_state:
        st.title(" Database Chatbot")

        with st.form("username_form"):
            username = st.text_input("Username", placeholder="Enter your username", max_chars=80,)
            submitted = st.form_submit_button("Start chatting")

        if submitted:
            username = username.strip()
            if username:
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Please enter a username.")

        st.stop()
    # Set up Session State
    username = st.session_state.username
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role":    "assistant", 
             "content": f"Hi **{username}**, I'm the GraphAcademy Chatbot! ""How can I help you?"},]
    # Persist chatbot across Streamlit reruns
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = Chatbot(username=username)

    chatbot = st.session_state.chatbot

    


    # Display messages in Session State
    for message in st.session_state.messages:
        write_message(message['role'], message['content'], save=False)

    # Handle any user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        write_message('user', prompt)

        # Generate a response
        chatbot.handle_submit(prompt)

    with st.sidebar:
    
            if st.sidebar.button("Finalize discussion", use_container_width=True):
                if chatbot.agent is None:
                    st.sidebar.error(
                        chatbot.error_msg
                        or "The agent is unavailable."
                    )
                else:
                    try:
                        with st.spinner("Saving preferences..."):
                            result = (chatbot.agent.finalize_discussion())
    
                        saved_count = result.get("saved_count",0,)
                        if saved_count > 0:
                            st.sidebar.success(f"Saved {saved_count} "f"dummy preferences.")
    
                            for preference in result.get("preferences", [],):
                                st.sidebar.write( f"**{preference['category']}**: " f"{preference['preference']}")
                        else:
                            st.sidebar.warning( "No preferences were saved.")
    
                    except Exception as error:
                        st.sidebar.error(f"Finalization failed: {error}")
    
            if st.button("🗑️ Delete conversation",use_container_width=True):
                try:
                    chatbot.agent.delete_conversation()  
                    write_message('assistant', "Conversation deleted.", save=False)   
                except Exception as e:
                    st.sidebar.error(f"Failed to delete conversation: {str(e)}") 

            st.markdown(f"Signed in as **{username}**")
            st.caption("Chat memory and preferences are stored for this user.")

            if st.button("Change user",use_container_width=True,):
                st.session_state.clear()
                st.rerun()