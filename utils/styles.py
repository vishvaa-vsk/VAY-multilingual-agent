import streamlit as st

def apply_custom_styles():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=Montserrat:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;600&display=swap');

            /* Hide Streamlit default header, footer, padding */
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}

            html, body, [data-testid="stAppViewContainer"] {
                background-color: #080313 !important;
                color: #ffffff;
                font-family: 'Montserrat', sans-serif;
                overflow-x: hidden;
            }

            [data-testid="stAppViewContainer"] > .main {
                padding: 0rem 0rem 1rem 0rem !important;
            }

            .block-container {
                padding-top: 0.5rem !important;
                padding-bottom: 0rem !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
                max-width: 100% !important;
            }

            /* Custom scrollbars */
            ::-webkit-scrollbar {
                width: 6px;
                height: 6px;
            }
            ::-webkit-scrollbar-track {
                background: #080313;
            }
            ::-webkit-scrollbar-thumb {
                background: #2a1b4e;
                border-radius: 3px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #ff6ec7;
            }

            /* Sidebar custom dark styling */
            [data-testid="stSidebar"] {
                background-color: #0d0620 !important;
                border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
            }

            [data-testid="stSidebar"] * {
                color: #d1c9e8 !important;
            }

            /* Streamlit button custom styles */
            .stButton > button {
                background: linear-gradient(135deg, #1f1238, #3b1d64) !important;
                color: #e2d9f3 !important;
                border: 1px solid rgba(255, 255, 255, 0.15) !important;
                border-radius: 8px !important;
                transition: all 0.3s ease !important;
                font-weight: 500 !important;
            }
            .stButton > button:hover {
                border-color: #ff6ec7 !important;
                box-shadow: 0 0 12px rgba(255, 110, 199, 0.4) !important;
                color: #ffffff !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
