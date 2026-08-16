import sys
import asyncio
import warnings

# Suppress DeprecationWarnings from event loop policy on Windows
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Force Selector event loop on Windows to prevent asyncio Proactor socket assertion crashes
if sys.platform == 'win32':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

import streamlit as st
import time
import json
import os
os.environ["STREAMLIT_UI"] = "1"
from audio_handler import decode_audio, encode_audio_bytes
from component_strands import strands_component
from component_galaxy import galaxy_component
from component_aurora import aurora_component

import io
import soundfile as sf
import torch
from langchain_core.messages import AIMessage, HumanMessage

from vay.asr.router import ASRRouter
from vay.graph.state import GraphState
from vay.graph.workflow import build_graph
from vay.tools.session import SessionContext
from vay.graph.utils import trim_history

# Ensure page config is set first
st.set_page_config(
    page_title="VAY Voice Assistant For You",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Try to import gTTS, fallback to mock if not installed yet
try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False

# Inject custom global CSS for premium UI styling
st.markdown("""
<style>
    /* Hide Streamlit top header bar, decoration line, and toolbar */
    header[data-testid="stHeader"] {
        display: none !important;
        background: transparent !important;
        height: 0 !important;
        visibility: hidden !important;
    }
    div[data-testid="stDecoration"] {
        display: none !important;
        height: 0 !important;
        visibility: hidden !important;
    }
    div[data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
    }
    #MainMenu, footer {
        display: none !important;
        visibility: hidden !important;
    }
    .block-container {
        padding-top: 1.5rem !important;
    }

    /* Dark Theme Core Styles */
    .stApp {
        background: radial-gradient(circle at center, #111115 0%, #070708 100%);
        color: #f3f4f6;
    }
    
    /* Custom Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(12, 12, 16, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Card Container styles */
    .premium-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(12px);
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
    }
    
    .premium-card h3 {
        margin-top: 0;
        font-weight: 600;
        letter-spacing: 0.5px;
        color: #f3f4f6;
    }
    
    /* Pipeline node styles */
    .pipeline-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 20px 0;
        padding: 10px;
        background: rgba(255,255,255,0.01);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.03);
    }
    
    .pipeline-node {
        flex: 1;
        text-align: center;
        padding: 10px 5px;
        margin: 0 4px;
        border-radius: 8px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        border: 1px solid rgba(255,255,255,0.05);
        background: rgba(255,255,255,0.02);
        color: #6b7280;
        transition: all 0.3s ease;
    }
    
    .pipeline-node.active {
        color: #f97316;
        border-color: rgba(249, 115, 22, 0.4);
        background: rgba(249, 115, 22, 0.1);
        box-shadow: 0 0 10px rgba(249, 115, 22, 0.2);
    }
    
    .pipeline-node.active-llm {
        color: #7c3aed;
        border-color: rgba(124, 58, 237, 0.4);
        background: rgba(124, 58, 237, 0.1);
        box-shadow: 0 0 10px rgba(124, 58, 237, 0.2);
    }
    
    .pipeline-node.active-tts {
        color: #06b6d4;
        border-color: rgba(6, 182, 212, 0.4);
        background: rgba(6, 182, 212, 0.1);
        box-shadow: 0 0 10px rgba(6, 182, 212, 0.2);
    }
    
    /* Metric styling */
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        background: linear-gradient(135deg, #f97316, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Escalated session item styling */
    .escalated-item {
        border-left: 3px solid #ef4444;
        background: rgba(239, 68, 68, 0.03);
        padding: 10px 15px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
        font-size: 13px;
    }

    /* ==================================================== */
    /* COUNTRY CODE & PHONE INPUT COLOR SETTINGS            */
    /* ==================================================== */
    
    /* 1. "Country Code" Label Color */
    div[data-testid="stSelectbox"] label p {
        color: #f97316 !important; /* <--- CHANGE COUNTRY CODE LABEL COLOR HERE */
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.5px;
    }

    /* 2. Country Code Dropdown Box (Background, Border, Selected Text) */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.1) !important; /* <--- CHANGE BOX BACKGROUND HERE */
        border: 1px solid rgba(249, 115, 22, 0.4) !important;   /* <--- CHANGE BORDER COLOR HERE */
        border-radius: 10px !important;
    }

    /* 3. Country Code Selected Value Text Color */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] * {
        color: #000!important;                                 /* <--- CHANGE TEXT VALUE COLOR HERE */
    }

    /* 4. "Phone Number" Label Color */
    div[data-testid="stTextInput"] label p {
        color: #06b6d4 !important; /* <--- CHANGE PHONE NUMBER LABEL COLOR HERE */
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.5px;
    }

    /* 5. Phone Number Input Box (Background & Border) */
    div[data-testid="stTextInput"] input {
        background-color: rgba(0, 0, 0, 0) !important;
        border: 1px solid rgba(6, 182, 212, 0.4) !important;
        color: #00000 !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- KNOWLEDGE BASE -----------------
OPERATOR_KB = [
    {
        "id": "KB001",
        "title": "Super 5G Unlimited Plan",
        "content": "The Super 5G Unlimited Plan costs 599 rupees per month. It includes unlimited 5G data, 100 SMS per day, and a 1-year Disney+ Hotstar subscription. Bill cycle starts on the 1st of every month.",
        "keywords": ["plan", "unlimited", "5g", "cost", "recharge", "monthly", "pack", "599", "super"]
    },
    {
        "id": "KB002",
        "title": "International Roaming Pack",
        "content": "The International Roaming Pack costs 1499 rupees. It is valid for 28 days and includes 5GB high-speed roaming data and 100 minutes of voice calls to India. Outgoing SMS costs 5 rupees per message.",
        "keywords": ["international", "roaming", "abroad", "travel", "calling", "outside india", "1499", "pack", "roam"]
    },
    {
        "id": "KB003",
        "title": "Bill Dispute and Grace Period",
        "content": "Telecom invoice disputes must be registered within 15 days of bill generation. Payments have a 3-day grace period. Late payments incur a fee of 1.5% of the total bill amount or 50 rupees, whichever is higher.",
        "keywords": ["bill", "dispute", "invoice", "late fee", "payment due", "grace period", "pay", "charge"]
    },
    {
        "id": "KB004",
        "title": "Account Closure / Cancellation (RESTRICTED)",
        "content": "Request for connection termination or account closure is a sensitive process. It requires verification of customer identification (Aadhaar/PAN) and must be routed immediately to a live senior manager for authentication.",
        "keywords": ["close account", "terminate", "cancel connection", "deactivate", "delete account", "disconnect"]
    }
]

# ----------------- SESSION STATE INITIALIZATION -----------------
if "session_started" not in st.session_state:
    st.session_state.session_started = False
if "phone_number" not in st.session_state:
    st.session_state.phone_number = ""
if "status" not in st.session_state:
    st.session_state.status = "idle"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "escalation_queue" not in st.session_state:
    st.session_state.escalation_queue = []
if "audio_to_play" not in st.session_state:
    st.session_state.audio_to_play = None
if "current_pipeline_data" not in st.session_state:
    st.session_state.current_pipeline_data = {
        "language": "N/A",
        "route": "N/A",
        "intent": "N/A",
        "confidence": 0.0,
        "raw_text": "N/A",
        "clean_text": "N/A",
        "action": "N/A",
        "entities": {}
    }

if "last_processed_event_id" not in st.session_state:
    st.session_state.last_processed_event_id = None

# Track toggle changes or button actions
if "component_key" not in st.session_state:
    st.session_state.component_key = 0

# ----------------- HELPER FUNCTIONS -----------------
def generate_text_to_speech(text, lang_code="en"):
    """
    Generates a TTS audio response using Microsoft Edge Neural voices (edge-tts).
    """
    import tempfile
    import os
    from vay.tts.engine import speak
    
    if not text:
        return b""
        
    fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    
    try:
        # Call the project's edge-tts wrapper (play=False because UI plays it)
        speak(text, lang=lang_code, play=False, output_path=tmp_path)
        
        # Read the generated MP3 bytes
        with open(tmp_path, "rb") as f:
            return f.read()
    except Exception as e:
        print(f"TTS generation error: {e}")
        return b""
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

def query_hybrid_rag(query_text):
    """
    Simulates a keyword hybrid RAG lookup on the knowledge base.
    Returns: best_match_doc, confidence_score
    """
    query_words = set(query_text.lower().replace("?", "").replace(".", "").split())
    best_doc = None
    best_score = 0.0
    
    for doc in OPERATOR_KB:
        # Count matching keywords
        matches = len(query_words.intersection(set(doc["keywords"])))
        # Normalize score
        score = matches / max(1, len(doc["keywords"]))
        # Boost score slightly if direct word overlaps are found in content
        content_words = set(doc["content"].lower().split())
        content_matches = len(query_words.intersection(content_words))
        score += (content_matches / len(content_words)) * 0.3
        
        # Keep highest score
        if score > best_score:
            best_score = min(1.0, score)
            best_doc = doc
            
    # Normalize score range (e.g. if any keywords overlap, give it a base relevance)
    if best_score > 0.0:
        best_score = round(0.4 + (best_score * 0.6), 2)
        
    return best_doc, best_score

@st.cache_resource
def get_asr_router():
    return ASRRouter()

@st.cache_resource
def get_agent_graph():
    return build_graph()

# ----------------- BACKGROUND MODEL PRELOADING -----------------
if "models_preloading" not in st.session_state:
    st.session_state.models_preloading = True
    
    import threading
    import subprocess
    def preload_models():
        print("[Background Preload] Starting to load ASRRouter and LangGraph...")
        get_asr_router()
        get_agent_graph()
        print("[Background Preload] Finished loading models!")
        
        # Seed customer database if needed
        print("[Background Preload] Ensuring customer DB is seeded...")
        try:
            from vay.tools.db_queries import init_db
            init_db()
            print("[Background Preload] Customer DB ready!")
        except Exception as e:
            print(f"[Background Preload] DB seed error: {e}")
        
        print("[Background Preload] Triggering KB build in background...")
        try:
            subprocess.Popen(["uv", "run", "python", "scripts/build_kb.py"])
        except Exception as e:
            print(f"[Background Preload] Error starting KB build: {e}")
            
    threading.Thread(target=preload_models, daemon=True).start()

def process_real_audio(base64_audio):
    """
    Decodes the browser's audio, runs it through ASRRouter -> LangGraph -> TTS.
    """
    print(f"\n--- [process_real_audio] Started! Base64 string length: {len(base64_audio)} ---")
    try:
        # 1. Decode base64 to WAV tensor
        wav_bytes = decode_audio(base64_audio)
        wav_io = io.BytesIO(wav_bytes)
        audio_np, sr = sf.read(wav_io)
        
        if len(audio_np.shape) > 1:
            audio_np = audio_np.mean(axis=1)
        tensor_chunk = torch.from_numpy(audio_np).float()
        
        # 2. Route and Transcribe (ASR)
        t_asr_start = time.time()
        router = get_asr_router()
        asr_result = router.route_and_transcribe(tensor_chunk)
            
        user_text = asr_result.raw_text
        detected_lang = asr_result.detected_language
        t_asr_end = time.time()
        
        print(f"[process_real_audio] STT: '{user_text}' (lang: {detected_lang}) [Took {t_asr_end - t_asr_start:.2f}s]")
        
        if not user_text.strip():
            st.session_state.status = "idle"
            st.rerun()
            return
            
        # 3. LangGraph Orchestrator (RAG + Agent)
        t_graph_start = time.time()
        graph = get_agent_graph()
        
        # Ensure session exists
        if "agent_session" not in st.session_state:
            st.session_state.agent_session = SessionContext(
                phone_number=st.session_state.phone_number,
                verified=True,
                language=detected_lang
            )
        if "agent_history" not in st.session_state:
            st.session_state.agent_history = []
            
        st.session_state.agent_session.language = detected_lang
            
        state: GraphState = {
            "phone_number": st.session_state.phone_number,
            "language": detected_lang,
            "transcript": user_text,
            "conversation_history": st.session_state.agent_history,
            "show_debug": False,
            "min_similarity": 0.3,
            "session": st.session_state.agent_session,
        }
        
        print("[process_real_audio] Invoking LangGraph...")
        result_state = graph.invoke(state)
        t_graph_end = time.time()
        
        # Extract reply
        reply = result_state.get("final_reply") or "Transferring you to an agent."
        print(f"[process_real_audio] Agent Reply: {reply} [Took {t_graph_end - t_graph_start:.2f}s]")
        
        # 4. Update History
        st.session_state.agent_history.append(HumanMessage(content=user_text))
        st.session_state.agent_history.append(AIMessage(content=reply))
        st.session_state.agent_history = trim_history(st.session_state.agent_history, 6)
        
        # 5. Update UI Pipeline Data
        st.session_state.current_pipeline_data = {
            "language": detected_lang,
            "route": asr_result.model_used,
            "intent": result_state.get("routed_agent", "Orchestrator"),
            "confidence": round(asr_result.confidence, 2),
            "raw_text": user_text,
            "clean_text": user_text,
            "action": "Handoff" if result_state.get("handoff") else "Completed",
            "entities": {}
        }
        
        # 6. Chat History UI
        st.session_state.chat_history.append({"speaker": "user", "text": user_text, "lang": detected_lang})
        st.session_state.chat_history.append({"speaker": "assistant", "text": reply, "lang": detected_lang})
        
        # 7. Check for call termination
        if result_state.get("call_end_requested") or result_state.get("handoff"):
            st.session_state.status = "handoff"
            st.rerun()
            return
            
        # 8. Generate TTS
        t_tts_start = time.time()
        print("[process_real_audio] Generating TTS...")
        tts_bytes = generate_text_to_speech(reply, lang_code=detected_lang)
        t_tts_end = time.time()
        print(f"[process_real_audio] TTS Generated [Took {t_tts_end - t_tts_start:.2f}s]")
        
        if tts_bytes:
            st.session_state.audio_to_play = encode_audio_bytes(tts_bytes)
            st.session_state.status = "speaking"
        else:
            st.session_state.status = "idle"
            
        print("[process_real_audio] Done! Triggering rerun.")
        st.rerun()
        
    except Exception as e:
        print(f"[process_real_audio] ERROR: {e}")
        import traceback
        traceback.print_exc()
        st.session_state.status = "idle"
        st.rerun()

def escalate_session(reason):
    """
    Triggers human agent escalation logic (Gotcha §7 - shared escalation route).
    Preserves active user session while recording escalation in Live Agent Dashboard.
    """
    st.session_state.status = "handoff"
    
    escalation_entry = {
        "timestamp": time.strftime("%H:%M:%S"),
        "raw_text": st.session_state.current_pipeline_data.get("raw_text", "Customer requested escalation"),
        "language": st.session_state.current_pipeline_data.get("language", "English"),
        "intent": st.session_state.current_pipeline_data.get("intent", "Supervisor Escalation"),
        "confidence": st.session_state.current_pipeline_data.get("confidence", 0.0),
        "reason": reason
    }
    
    st.session_state.escalation_queue.insert(0, escalation_entry)
    
    # Record end/handoff notification in conversation transcript
    st.session_state.chat_history.append({
        "speaker": "system",
        "text": f"ℹ️ [CALL ENDED] Reason: {reason}",
        "lang": "en"
    })
    
    # Keep session active and ready for live operator handling
    st.session_state.audio_to_play = None
    st.session_state.current_pipeline_data["action"] = f"Escalated ({reason})"
    st.rerun()

# ----------------- MAIN LAYOUT ROUTING -----------------
if not st.session_state.session_started:
    # Render pure white Galaxy WebGL background from React Bits
    galaxy_component(
        mouseRepulsion=False,
        mouseInteraction=False,
        density=1.5,
        glowIntensity=0.2,
        saturation=0.0,
        hueShift=0.0,
        starSpeed=0.5,
        speed=1.0,
        transparent=True,
        key="galaxy_frontpage_bg"
    )

    # Load VAY logo as base64 for clean landing header rendering
    logo_b64 = ""
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "vay_logo.png")
    if os.path.exists(logo_path):
        import base64
        with open(logo_path, "rb") as img_f:
            logo_b64 = base64.b64encode(img_f.read()).decode("utf-8")

    # Render premium landing screen with frosted glassmorphism over Galaxy
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, login_col, _ = st.columns([1, 1.4, 1])
    with login_col:
        img_html = f'<img src="data:image/png;base64,{logo_b64}" style="width: 145px; height: auto; border-radius: 12px; margin-bottom: 12px; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.5); border: 1px solid rgba(255,255,255,0.15); display: inline-block;">' if logo_b64 else '<div style="font-size: 38px; margin-bottom: 8px;">🌌</div>'
        st.markdown(f"""
        <div class="premium-card" style="text-align: center; padding: 35px 30px; margin-bottom: 0; background: rgba(14, 14, 20,0); border:0px solid rgba(255, 255, 255, 0.12); backdrop-filter: blur(0px); -webkit-backdrop-filter: blur(20px); box-shadow: 0 20px 50px rgba(0, 0, 0, 0);">
            {img_html}
            <h1 style="margin-top: 0; font-weight: 800; font-size: 28px; background: linear-gradient(135deg, #f97316, #a855f7, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: 1.5px;">VAY ASSISTANT</h1>
            <p style="color: #cbd5e1; font-size: 14px; margin-top: 6px; margin-bottom: 0; letter-spacing: 0.3px;">Next-Gen Multilingual Voice Operator Portal</p>
            <p style="color: #64748b; font-size: 12px; margin-top: 4px; margin-bottom: 20px;">Powered by Real-Time Neural Speech & Hybrid RAG</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown("<div style='padding: 24px; background: rgba(0, 0, 0, 0); border: 0px solid rgba(255,255,255,0); border-top: none; border-radius: 0 0 16px 16px; backdrop-filter: blur(0px); -webkit-backdrop-filter: blur(20px); box-shadow: 0 0px 50px rgba(0, 0, 0, 0);'>", unsafe_allow_html=True)
            
            country_code = st.selectbox(
                "Country Code",
                ["+91 (India)", "+1 (USA)", "+44 (UK)", "+61 (Australia)", "+65 (Singapore)"],
                index=0
            )
            
            phone_input = st.text_input(
                "Phone Number",
                max_chars=10,
                placeholder="Enter 10-digit mobile number"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("Start Session", type="primary", use_container_width=True):
                # Validation rules
                if not phone_input:
                    st.warning("⚠️ Please enter your phone number.")
                elif not phone_input.isdigit():
                    st.error("❌ Phone number must contain only numeric digits.")
                elif len(phone_input) != 10:
                    st.error(f"❌ Phone number must be exactly 10 digits (current length: {len(phone_input)}).")
                else:
                    selected_code = country_code.split(" ")[0]
                    st.session_state.phone_number = f"{selected_code} {phone_input}"
                    st.session_state.component_key += 1 # Force fresh component instance
                    st.session_state.session_started = True
                    st.success("✅ Session started successfully!")
                    time.sleep(0.6)
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
else:
    # Hide sidebar and side tab controls completely on the session page
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"],
            [data-testid="collapsedControl"],
            button[data-testid="stSidebarCollapseButton"] {
                display: none !important;
                visibility: hidden !important;
                width: 0 !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Render Aurora WebGL background with authentic Northern Lights colors
    aurora_component(
        colorStops=["#00FF87", "#60EFFF", "#7C3AED"],
        amplitude=1.0,
        blend=0.55,
        speed=0.45,
        key="aurora_session_bg"
    )

    # Fixed permanent visualizer configuration
    strand_colors = ["#FF4242", "#7C3AED", "#06B6D4", "#EAB308"]
    strand_count = 4
    strand_amp = 1.0
    strand_thickness = 0.8
    strand_glow = 1.20
    glass_mode = False
    glass_refraction = 1.0
    glass_size = 1.0

    # ---- Status-aware strand speed: slow idle, gently faster when active ----
    _status = st.session_state.status
    if _status in ("listening", "speaking"):
        strand_speed = 0.1  # slightly faster, but still smooth and steady
    elif _status == "thinking":
        strand_speed = 0.1  # medium while processing
    else:
        strand_speed = 0.1 # slow, calm idle flow

    # Simulated accent data (used internally by audio_recorded pipeline)
    current_accent_data = {
        "lang": "en",
        "queries": [
            "How much is my late payment fee?",
            "Tell me about the 599 unlimited data plan",
            "I would like to cancel my mobile subscription",
            "Is there an option to recharge for international travel?"
        ]
    }

    # ----------------- PHONE NUMBER BOX — TOP RIGHT (Streamlit-compatible) -----------------
    # Inject a global CSS rule + place the box in a full-width row floated right
    _phone = st.session_state.get("phone_number", "")
    st.markdown(
        f"""
        <style>
        /* Force phone pill to stay fixed in viewport top-right, above Streamlit's toolbar */
        #vay-phone-pill {{
            position: fixed;
            top: 10px;
            right: 20px;
            z-index: 999999;
            background: rgba(12, 12, 18, 0.92);
            border: 1.5px solid #f97316;
            border-radius: 32px;
            padding: 7px 20px 7px 14px;
            display: flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 0 18px rgba(249,115,22,0.25), 0 2px 8px rgba(0,0,0,0.6);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            font-family: 'Inter', sans-serif;
            min-width: 160px;
            pointer-events: none;
        }}
        #vay-phone-pill .icon {{
            font-size: 15px;
            animation: pulse-icon 2.5s ease-in-out infinite;
        }}
        #vay-phone-pill .num {{
            color: #f97316;
            font-weight: 700;
            font-size: 13px;
            letter-spacing: 0.6px;
        }}
        #vay-phone-pill .label {{
            color: #6b7280;
            font-size: 10px;
            font-weight: 500;
            display: block;
            line-height: 1.1;
        }}
        @keyframes pulse-icon {{
            0%, 100% {{ opacity: 1; }}
            50%       {{ opacity: 0.5; }}
        }}
        </style>
        <div id="vay-phone-pill">
            <span class="icon">📞</span>
            <div>
                <span class="label">ACTIVE SESSION</span>
                <span class="num">{_phone}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ----------------- MAIN VOICE INTERFACE (full-width) -----------------
    # Load VAY logo
    _logo_b64 = ""
    _logo_path = os.path.join(os.path.dirname(__file__), "assets", "vay_logo.png")
    if os.path.exists(_logo_path):
        import base64 as _b64
        with open(_logo_path, "rb") as _lf:
            _logo_b64 = _b64.b64encode(_lf.read()).decode("utf-8")

    if _logo_b64:
        st.markdown(
            f"<div style='text-align:center; margin-bottom:4px;'>"
            f"<img src='data:image/png;base64,{_logo_b64}' "
            f"style='width:160px; height:auto; border-radius:12px; "
            f"box-shadow:0 6px 24px rgba(0,0,0,0.5); display:inline-block;'>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<h2 style='text-align:center; margin-bottom:0;'>🎙️ VAY - Voice Assistant for you</h2>",
            unsafe_allow_html=True
        )

    # Render Strands visualizer
    event_data = strands_component(
        colors=strand_colors,
        count=strand_count,
        speed=strand_speed,
        amplitude=strand_amp,
        waviness=1.1,
        thickness=strand_thickness,
        glow=strand_glow,
        taper=3.0,
        spread=1.1,
        hueShift=0.0,
        intensity=0.7,
        saturation=1.5,
        opacity=1.0,
        scale=1.4,
        glass=glass_mode,
        refraction=glass_refraction,
        dispersion=1.0,
        glassSize=glass_size,
        status=st.session_state.status,
        audio_data=st.session_state.audio_to_play,
        key=f"strands_element_{st.session_state.component_key}"
    )

    # Process events from the WebGL component safely with event ID deduplication
    if event_data is not None:
        event_id = event_data.get("id")
        event_name = event_data.get("event")

        # Only process if this is a newly received event that has not been handled
        if event_id and event_id != st.session_state.get("last_processed_event_id"):
            st.session_state.last_processed_event_id = event_id

            if event_name == "mic_start":
                if st.session_state.status != "listening":
                    st.session_state.status = "listening"
                    st.session_state.audio_to_play = None
                    st.rerun()

            elif event_name in ("mic_pause", "mic_stop"):
                if st.session_state.status != "idle":
                    st.session_state.status = "idle"
                    st.session_state.audio_to_play = None
                    st.rerun()

            elif event_name == "audio_recorded":
                st.session_state.status = "thinking"
                base64_audio = event_data.get("audio")
                if base64_audio:
                    process_real_audio(base64_audio)

            elif event_name == "audio_finished":
                if st.session_state.status != "listening":
                    st.session_state.status = "listening"
                    st.session_state.audio_to_play = None
                    st.rerun()

            elif event_name in ("end_session", "escalate_click"):
                st.session_state.session_started = False
                st.session_state.phone_number = ""
                st.session_state.chat_history = []
                st.session_state.status = "idle"
                st.session_state.audio_to_play = None
                st.session_state.component_key += 1
                st.rerun()


