import streamlit as st
from utils import write_message
from agent import Agent 
import os

os.environ["LANGSMITH_TRACING"] = "false"




class Chatbot:
    def __init__(self):
        self.agent = Agent()
        self.generate_response = self.agent.generate_response

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
            response = self.generate_response(message)
            write_message('assistant', response)
            

    


if __name__ == "__main__":
    # Page Config
    st.set_page_config("Ebert", page_icon=":movie_camera:")

    # Set up Session State
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, I'm the GraphAcademy Chatbot!  How can I help you?"},
        ]
    chatbot = Chatbot()
    # Display messages in Session State
    for message in st.session_state.messages:
        write_message(message['role'], message['content'], save=False)

    # Handle any user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        write_message('user', prompt)

        # Generate a response
        chatbot.handle_submit(prompt)