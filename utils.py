import os
import streamlit as st


def reset_conversation():
    st.session_state.conversation = None
    st.session_state.chat_history = None
    st.session_state.messages = []
