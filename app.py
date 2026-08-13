import streamlit as st
import time
import json
import os
from audio_handler import decode_audio, encode_audio_bytes
from component_strands import strands_component

# Ensure page config is set first
st.set_page_config(
    page_title="VAY Voice Assistant - Operator Portal & Live Agent Dashboard",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
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

# Track toggle changes or button actions
if "component_key" not in st.session_state:
    st.session_state.component_key = 0

# ----------------- HELPER FUNCTIONS -----------------
def generate_text_to_speech(text, lang_code="en"):
    """
    Generates a TTS audio response using gTTS (or mocks if offline)
    """
    if not HAS_GTTS:
        # Return a silent / mock response byte array
        return b""
    
    # Map input ISO lang_codes to gtts options
    gtts_lang = "en"
    if lang_code == "ta":
        gtts_lang = "ta"
    elif lang_code == "hi":
        gtts_lang = "hi"
        
    try:
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        # Write to memory file
        from io import BytesIO
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception as e:
        st.sidebar.error(f"TTS Generation Error: {e}")
        return b""

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

def run_multilingual_pipeline(user_query, detected_lang="en"):
    """
    Executes the logical pipeline defined in project_context.md
    """
    # 1. State: Thinking (ASR complete, now normalizing/cleanup)
    time.sleep(0.5) # Simulate small latency
    
    # Normalize transcription (simulate LLM-based code-switch normalization)
    normalized_query = user_query
    # Simple mockup cleanup for Tanglish/Hinglish
    if detected_lang == "ta" and "bill" in user_query.lower():
        normalized_query = "எனது பில் விவரங்கள் என்ன?"
    elif detected_lang == "hi" and "bill" in user_query.lower():
        normalized_query = "मेरा बिल का विवरण क्या है?"
        
    # 2. Intent + Entity classification
    intent = "General Query"
    entities = {}
    
    query_lower = user_query.lower()
    if any(k in query_lower for k in ["close", "cancel", "terminate", "deactivate", "delete", "மூடு", "बंद"]):
        intent = "Restricted: Connection Termination"
        action = "ESCALATE (Sensitive Intent)"
    elif any(k in query_lower for k in ["bill", "charge", "dispute", "pay", "பில்", "बिल"]):
        intent = "Billing Inquiry"
        action = "RAG Search"
    elif any(k in query_lower for k in ["plan", "unlimited", "roaming", "5g", "pack", "திட்டம்", "प्लान"]):
        intent = "Plan Upgrade / Inquiry"
        action = "RAG Search"
    else:
        intent = "General Help"
        action = "RAG Search"
        
    # Route sensitive intents directly to human handoff (Gotcha §8)
    if intent.startswith("Restricted"):
        st.session_state.current_pipeline_data = {
            "language": "Tamil" if detected_lang == "ta" else ("Hindi" if detected_lang == "hi" else "English"),
            "route": "IndicConformer" if detected_lang in ["ta", "hi"] else "Whisper-v3-turbo",
            "intent": intent,
            "confidence": 0.99,
            "raw_text": user_query,
            "clean_text": normalized_query,
            "action": "Escalate to Human Agent",
            "entities": entities
        }
        escalate_session("Sensitive billing/account cancellation requested.")
        return
        
    # 3. RAG Module: Query Operator Knowledge Base
    best_doc, score = query_hybrid_rag(normalized_query)
    
    st.session_state.current_pipeline_data = {
        "language": "Tamil" if detected_lang == "ta" else ("Hindi" if detected_lang == "hi" else "English"),
        "route": "IndicConformer" if detected_lang in ["ta", "hi"] else "Whisper-v3-turbo",
        "intent": intent,
        "confidence": score,
        "raw_text": user_query,
        "clean_text": normalized_query,
        "action": "RAG Retrieval",
        "entities": entities
    }
    
    # 4. Handoff check (Threshold τ ≈ 0.75 check)
    retrieval_threshold = 0.65 # Empirically relaxed for mock keywords
    if score < retrieval_threshold or best_doc is None:
        escalate_session(f"Low retrieval confidence ({score} < {retrieval_threshold})")
        return
        
    # 5. LLM Response Generation (grounded in retrieved context, in user's language)
    response_text = ""
    if detected_lang == "ta":
        if best_doc["id"] == "KB001":
            response_text = "சூப்பர் 5ஜி அன்லிமிட்டெட் திட்டம் மாதத்திற்கு 599 ரூபாய். இதில் அன்லிமிட்டெட் 5ஜி டேட்டா மற்றும் டிஸ்னி+ ஹாட்ஸ்டார் சந்தா உள்ளது."
        elif best_doc["id"] == "KB002":
            response_text = "சர்வதேச ரோமிங் திட்டம் 1499 ரூபாய் ஆகும். இது 28 நாட்களுக்கு செல்லுபடியாகும்."
        else:
            response_text = "உங்கள் பில் கட்டண முறையீடு 15 நாட்களுக்குள் பதிவு செய்யப்பட வேண்டும். கூடுதல் விவரங்களுக்கு எங்கள் ஏஜெண்டை தொடர்பு கொள்ளவும்."
    elif detected_lang == "hi":
        if best_doc["id"] == "KB001":
            response_text = "सुपर 5G अनलिमिटेड प्लान की कीमत 599 रुपये प्रति माह है। इसमें अनलिमिटेड 5G डेटा और डिज़नी+ हॉटस्टार शामिल है।"
        elif best_doc["id"] == "KB002":
            response_text = "इंटरनेशनल रोमिंग पैक की कीमत 1499 रुपये है। यह 28 दिनों के लिए वैध है।"
        else:
            response_text = "आपके बिल विवाद को 15 दिनों के भीतर दर्ज किया जाना चाहिए। 3 दिनों की अतिरिक्त छूट अवधि मिलती है।"
    else: # English
        if best_doc["id"] == "KB001":
            response_text = "The Super 5G Unlimited Plan costs 599 rupees per month and includes unlimited 5G data plus Disney+ Hotstar."
        elif best_doc["id"] == "KB002":
            response_text = "The International Roaming Pack costs 1499 rupees, is valid for 28 days, and has 5GB roaming data."
        else:
            response_text = "Invoice disputes must be registered within 15 days. We provide a 3-day payment grace period."

    # Record history
    st.session_state.chat_history.append({"speaker": "user", "text": user_query, "lang": detected_lang})
    st.session_state.chat_history.append({"speaker": "assistant", "text": response_text, "lang": detected_lang})
    
    # 6. Generate TTS Response Audio
    tts_bytes = generate_text_to_speech(response_text, lang_code=detected_lang)
    
    if tts_bytes:
        st.session_state.audio_to_play = encode_audio_bytes(tts_bytes)
        st.session_state.status = "speaking"
    else:
        st.session_state.status = "idle"
        
    st.rerun()

def escalate_session(reason):
    """
    Triggers human agent escalation logic (Gotcha §7 - shared escalation route)
    """
    st.session_state.status = "handoff"
    
    escalation_entry = {
        "timestamp": time.strftime("%H:%M:%S"),
        "raw_text": st.session_state.current_pipeline_data["raw_text"],
        "language": st.session_state.current_pipeline_data["language"],
        "intent": st.session_state.current_pipeline_data["intent"],
        "confidence": st.session_state.current_pipeline_data["confidence"],
        "reason": reason
    }
    
    st.session_state.escalation_queue.insert(0, escalation_entry)
    st.session_state.chat_history.append({
        "speaker": "system",
        "text": f"⚠️ Session Escalated to Live Agent. Reason: {reason}",
        "lang": "en"
    })
    
    st.session_state.current_pipeline_data["action"] = "ESCALATED TO LIVE EXECUTIVE"
    st.rerun()

# ----------------- SIDEBAR: SYSTEM DASHBOARD & SETTINGS -----------------
with st.sidebar:
    st.markdown("## ⚙️ Demo Accent Controller")
    st.markdown("Control what language accent the simulator transcribes when you speak or click.")
    
    accent_option = st.radio(
        "Voice Input Simulation Mode",
        ("Tamil Speaker Accent", "Hindi Speaker Accent", "English Speaker Accent")
    )
    
    simulated_queries = {
        "Tamil Speaker Accent": {
            "lang": "ta",
            "queries": [
                "எனது பில் கட்டணம் எவ்வளவு?",
                "599 அன்லிமிட்டெட் திட்டம் விவரம்",
                "எனது கணக்கை மூட வேண்டும்",  # Sensitive intent trigger
                "சர்வதேச ரோமிங் என்ன?"
            ]
        },
        "Hindi Speaker Accent": {
            "lang": "hi",
            "queries": [
                "मेरा बिल कितना आया है?",
                "599 वाला प्लान क्या है?",
                "कनेक्शन बंद करवाना है",  # Sensitive intent trigger
                "इंटरनेशनल रोमिंग पैक"
            ]
        },
        "English Speaker Accent": {
            "lang": "en",
            "queries": [
                "How much is my late payment fee?",
                "Tell me about the 599 unlimited data plan",
                "I would like to cancel my mobile subscription",  # Sensitive intent trigger
                "Is there an option to recharge for international travel?"
            ]
        }
    }
    
    current_accent_data = simulated_queries[accent_option]
    
    st.markdown("---")
    st.markdown("## 🛠️ Strands Customization")
    st.markdown("Tweak physical parameters of the React Bits `<Strands />` visualizer.")
    
    color_presets = {
        "Vaporwave Glow": ["#FF4242", "#7C3AED", "#06B6D4", "#EAB308"],
        "Sunset Fire": ["#F97316", "#EA580C", "#F43F5E", "#EAB308"],
        "Ocean Matrix": ["#06B6D4", "#0D9488", "#059669", "#7C3AED"],
        "Rainbow Spectrum": []
    }
    preset = st.selectbox("Color Palette", list(color_presets.keys()))
    strand_colors = color_presets[preset]
    
    strand_count = st.slider("Strand Count", 1, 12, 4)
    strand_speed = st.slider("Flow Speed", 0.1, 2.0, 0.6, step=0.1)
    strand_amp = st.slider("Wave Amplitude", 0.2, 3.0, 1.2, step=0.1)
    strand_thickness = st.slider("Strand Thickness", 0.1, 2.0, 0.8, step=0.1)
    strand_glow = st.slider("Glow Density", 0.5, 5.0, 2.8, step=0.1)
    
    glass_mode = st.toggle("Glass Orb Lens", value=False)
    glass_refraction = st.slider("Refraction Index", 0.1, 2.0, 1.0) if glass_mode else 1.0
    glass_size = st.slider("Glass Sphere Size", 0.5, 1.5, 1.0) if glass_mode else 1.0

# ----------------- MAIN LAYOUT: MULTI-COLUMN DESIGN -----------------
col_main, col_dashboard = st.columns([5, 4])

with col_main:
    st.markdown("<h2 style='text-align: center; margin-bottom: 0;'>🎙️ VAY Voice Interface</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color:#6b7280; font-size:13px;'>Interact using the microphone visualizer below. Select presets in the sidebar to simulate different accents.</p>", unsafe_allow_html=True)
    
    # Renders the Unified Custom Component
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
    
    # Process return events from custom WebGL component
    if event_data is not None:
        event_name = event_data.get("event")
        
        if event_name == "mic_start":
            st.session_state.status = "listening"
            st.session_state.audio_to_play = None
            st.rerun()
            
        elif event_name == "audio_recorded":
            st.session_state.status = "thinking"
            # Get simulated transcript based on the selected speaker accent
            # To showcase full intelligence, we select a query randomly or cyclically from list
            import random
            simulated_query = random.choice(current_accent_data["queries"])
            
            # Execute pipeline
            run_multilingual_pipeline(simulated_query, detected_lang=current_accent_data["lang"])
            
        elif event_name == "audio_finished":
            st.session_state.status = "idle"
            st.session_state.audio_to_play = None
            st.rerun()
            
        elif event_name == "escalate_click":
            escalate_session("User requested immediate human handoff.")

    # In-App Text Fallback Query Box
    st.markdown("---")
    st.markdown("### 💬 Text Query Fallback")
    text_query = st.chat_input("Type your customer service query here (Hindi, Tamil, or English)...")
    if text_query:
        st.session_state.status = "thinking"
        # Determine language simply
        lang_id = "en"
        # Simple character checks to identify script
        tamil_chars = set(range(0x0B80, 0x0BFF))
        hindi_chars = set(range(0x0900, 0x097F))
        for char in text_query:
            val = ord(char)
            if val in tamil_chars:
                lang_id = "ta"
                break
            elif val in hindi_chars:
                lang_id = "hi"
                break
        
        # Execute pipeline
        run_multilingual_pipeline(text_query, detected_lang=lang_id)

    # State Machine Visualizer
    st.markdown("### 🛠️ State Machine Architecture")
    
    # Dynamic class markers based on status
    is_vad = "active" if st.session_state.status == "listening" else ""
    is_asr = "active" if st.session_state.status == "thinking" else ""
    is_rag = "active-llm" if st.session_state.status == "thinking" and st.session_state.current_pipeline_data["action"] == "RAG Search" else ""
    is_tts = "active-tts" if st.session_state.status == "speaking" else ""
    is_handoff = "active" if st.session_state.status == "handoff" else ""
    
    st.markdown(f"""
    <div class="pipeline-container">
        <div class="pipeline-node {is_vad}">VAD Detection</div>
        <div style="color:#4b5563">→</div>
        <div class="pipeline-node {is_asr}">Language ID / ASR</div>
        <div style="color:#4b5563">→</div>
        <div class="pipeline-node {is_rag}">Hybrid RAG Gate</div>
        <div style="color:#4b5563">→</div>
        <div class="pipeline-node {is_tts}">Language TTS</div>
        <div style="color:#4b5563">→</div>
        <div class="pipeline-node {is_handoff}">Human Handoff</div>
    </div>
    """, unsafe_allow_html=True)

with col_dashboard:
    st.markdown("<h2 style='text-align: center; margin-bottom: 0;'>🛡️ Live Agent Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color:#6b7280; font-size:13px;'>Real-time orchestration logs and human agent queue escalation analytics.</p>", unsafe_allow_html=True)
    
    # Pipeline Metrics Panel
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.markdown("<h3>🎯 Real-time Call Intelligence</h3>", unsafe_allow_html=True)
    
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("Detected Language")
        st.markdown(f"<p class='metric-value'>{st.session_state.current_pipeline_data['language']}</p>", unsafe_allow_html=True)
        st.markdown("ASR Model Route")
        st.write(f"`{st.session_state.current_pipeline_data['route']}`")
    with mc2:
        st.markdown("Confidence Score")
        st.markdown(f"<p class='metric-value'>{st.session_state.current_pipeline_data['confidence'] * 100}%</p>", unsafe_allow_html=True)
        st.markdown("Intent Classification")
        st.write(f"`{st.session_state.current_pipeline_data['intent']}`")
        
    st.markdown("<div style='margin-top: 15px;'><strong>Current Pipeline Node:</strong></div>", unsafe_allow_html=True)
    st.write(f"`{st.session_state.current_pipeline_data['action']}`")
    
    st.markdown("<strong>Transcription (Raw):</strong>")
    st.write(f"*{st.session_state.current_pipeline_data['raw_text']}*")
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Escalated Human Handoff Queue
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.markdown("<h3>🚨 Escalated Sessions Queue</h3>", unsafe_allow_html=True)
    
    if not st.session_state.escalation_queue:
        st.info("No active escalations. System is operating within safety thresholds.")
    else:
        for idx, item in enumerate(st.session_state.escalation_queue):
            st.markdown(f"""
            <div class="escalated-item">
                <div style="display:flex; justify-content:space-between; font-weight:600; margin-bottom: 4px;">
                    <span style="color:#ef4444;">Session #{len(st.session_state.escalation_queue)-idx}</span>
                    <span style="color:#6b7280; font-size:11px;">{item['timestamp']}</span>
                </div>
                <div><strong>Query:</strong> "{item['raw_text']}"</div>
                <div style="font-size:11px; margin-top:4px; color:#9ca3af;">
                    Lang: {item['language']} | Intent: {item['intent']} | Reason: <span style="color:#f87171;">{item['reason']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    if st.button("Clear Escalation Queue", type="secondary"):
        st.session_state.escalation_queue = []
        st.rerun()
        
    st.markdown("</div>", unsafe_allow_html=True)

    # Chat Transcript Log
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.markdown("<h3>📝 Transcript History</h3>", unsafe_allow_html=True)
    
    if not st.session_state.chat_history:
        st.caption("No dialog active yet. Start speaking or typing a query to begin.")
    else:
        for msg in st.session_state.chat_history:
            if msg["speaker"] == "user":
                st.write(f"🗣️ **User ({msg['lang']}):** {msg['text']}")
            elif msg["speaker"] == "assistant":
                st.write(f"🤖 **Assistant ({msg['lang']}):** {msg['text']}")
            else:
                st.write(msg["text"])
                
    if st.button("Reset Conversation", type="primary"):
        st.session_state.chat_history = []
        st.session_state.status = "idle"
        st.session_state.audio_to_play = None
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
        st.session_state.component_key += 1 # Force component reset
        st.rerun()
        
    st.markdown("</div>", unsafe_allow_html=True)
