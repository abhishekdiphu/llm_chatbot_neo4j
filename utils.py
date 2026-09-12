import streamlit as st
from streamlit.runtime.scriptrunner.script_runner import get_script_run_ctx

from pydantic import BaseModel, Field

def write_message(role, content, save = True):
    """
    This is a helper function that saves a message to the
     session state and then writes a message to the UI
    """
    # Append to session state
    if save:
        st.session_state.messages.append({"role": role, "content": content})

    # Write to UI
    with st.chat_message(role):
        st.markdown(content)

def get_session_id():
    return get_script_run_ctx().session_id


class ExtractedPreference(BaseModel):
    category: str = Field(
        description="Preference category"
    )

    preference: str = Field(
        description=(
            "A concise statement of the user's "
            "lasting preference"
        )
    )


class ExtractedPreferences(BaseModel):
    preferences: list[ExtractedPreference] = Field(
        default_factory=list
    )


