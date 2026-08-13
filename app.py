import streamlit as st
import datetime
from utils.styles import apply_custom_styles
from utils.case_db import MOCK_CASES, process_query, DEFAULT_PROMPT_RESPONSES
from components.vay_orb_ui import render_vay_ui

# Configure Page Setup
st.set_page_config(
    page_title="VAY — Multilingual Voice Assistant",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Dark Cosmic Theme CSS
apply_custom_styles()

# Initialize Session State Variables
if "app_state" not in st.session_state:
    st.session_state.app_state = "LISTENING"

if "caption_text" not in st.session_state:
    st.session_state.caption_text = "Certainly, I am looking into your case. Please provide your case ID."

if "language" not in st.session_state:
    st.session_state.language = "en"

if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "assistant", "text": st.session_state.caption_text, "time": datetime.datetime.now().strftime("%H:%M:%S")}
    ]

# Sidebar Controls
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <h2 style="font-family: 'Cinzel', serif; letter-spacing: 4px; color: #ff6ec7; margin: 0;">VAY</h2>
        <span style="font-size: 11px; letter-spacing: 3px; color: #a491ca;">VOICE ASSISTANT HUD</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("### 🌐 Multilingual Voice Settings")
lang_options = {
    "English (en-US)": "en",
    "Spanish (es-ES)": "es",
    "French (fr-FR)": "fr",
    "German (de-DE)": "de",
    "Hindi (hi-IN)": "hi",
    "Japanese (ja-JP)": "ja"
}

selected_lang_name = st.sidebar.selectbox(
    "Target Language",
    options=list(lang_options.keys()),
    index=0
)

new_lang = lang_options[selected_lang_name]
if new_lang != st.session_state.language:
    st.session_state.language = new_lang
    # Update default caption to localized default prompt
    st.session_state.caption_text = DEFAULT_PROMPT_RESPONSES.get(new_lang, DEFAULT_PROMPT_RESPONSES["en"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎙️ Manual Query & Voice Input")

user_input = st.sidebar.text_input(
    "Type spoken query / Case ID:",
    placeholder="e.g. Check my case status for CASE-1092",
    key="text_query_input"
)

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Submit Query", use_container_width=True):
        if user_input.strip():
            st.session_state.app_state = "PROCESSING"
            st.session_state.history.append({"role": "user", "text": user_input, "time": datetime.datetime.now().strftime("%H:%M:%S")})
            
            # Process query via case_db
            response = process_query(user_input, lang=st.session_state.language)
            st.session_state.caption_text = response
            st.session_state.app_state = "RESPONDING"
            st.session_state.history.append({"role": "assistant", "text": response, "time": datetime.datetime.now().strftime("%H:%M:%S")})
            st.rerun()

with col2:
    if st.button("Reset State", use_container_width=True):
        st.session_state.app_state = "IDLE"
        st.session_state.caption_text = DEFAULT_PROMPT_RESPONSES.get(st.session_state.language, DEFAULT_PROMPT_RESPONSES["en"])
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔮 State Machine Visual Testing")
st.sidebar.caption("Toggle Orb Visualizer states directly to inspect animations:")

state_col1, state_col2 = st.sidebar.columns(2)
with state_col1:
    if st.button("Idle State", use_container_width=True):
        st.session_state.app_state = "IDLE"
        st.rerun()
    if st.button("Listening State", use_container_width=True):
        st.session_state.app_state = "LISTENING"
        st.rerun()

with state_col2:
    if st.button("Processing State", use_container_width=True):
        st.session_state.app_state = "PROCESSING"
        st.rerun()
    if st.button("Responding State", use_container_width=True):
        st.session_state.app_state = "RESPONDING"
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Demo Case Records")
selected_case = st.sidebar.selectbox(
    "Select Case to Test:",
    options=list(MOCK_CASES.keys()),
    format_func=lambda cid: f"{cid} - {MOCK_CASES[cid]['title']}"
)

if st.sidebar.button("Test Selected Case Query", use_container_width=True):
    st.session_state.app_state = "PROCESSING"
    case_query = f"Where is my case {selected_case}?"
    st.session_state.history.append({"role": "user", "text": case_query, "time": datetime.datetime.now().strftime("%H:%M:%S")})
    
    resp = process_query(case_query, lang=st.session_state.language)
    st.session_state.caption_text = resp
    st.session_state.app_state = "RESPONDING"
    st.session_state.history.append({"role": "assistant", "text": resp, "time": datetime.datetime.now().strftime("%H:%M:%S")})
    st.rerun()

with st.sidebar.expander("🔍 View Case Record Details"):
    cdata = MOCK_CASES[selected_case]
    st.markdown(f"**Title**: {cdata['title']}")
    st.markdown(f"**Category**: {cdata['category']}")
    st.markdown(f"**Status**: `{cdata['status']}`")
    st.markdown(f"**ETA**: {cdata['eta']}")
    st.markdown(f"**Details**: {cdata['details']}")

st.sidebar.markdown("---")
with st.sidebar.expander("💬 Voice Session Log", expanded=False):
    for turn in reversed(st.session_state.history):
        role_icon = "👤" if turn["role"] == "user" else "🔮"
        st.caption(f"{role_icon} [{turn['time']}] {turn['role'].upper()}")
        st.markdown(f"*{turn['text']}*")
        st.markdown("---")

# Main Application Layout
render_vay_ui(
    state=st.session_state.app_state,
    caption_text=st.session_state.caption_text,
    current_lang=st.session_state.language,
    key="vay_main_component"
)
