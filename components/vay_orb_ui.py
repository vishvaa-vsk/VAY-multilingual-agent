import streamlit as st
import streamlit.components.v1 as components
import json

def render_vay_ui(state: str, caption_text: str, current_lang: str = "en", key: str = "vay_component"):
    """
    Renders the VAY Multilingual Voice Assistant UI component inside Streamlit.
    
    Parameters:
    - state: "IDLE", "LISTENING", "PROCESSING", "RESPONDING", "ENDED"
    - caption_text: The live transcript/subtitle text shown at the bottom
    - current_lang: Selected language code ('en', 'es', 'fr', 'de', 'hi', 'ja')
    """
    
    # Map state to HUD label
    status_labels = {
        "IDLE": "TAP MIC TO SPEAK",
        "LISTENING": "LISTENING…",
        "PROCESSING": "PROCESSING…",
        "RESPONDING": "RESPONDING…",
        "ENDED": "SESSION ENDED"
    }
    
    status_display = status_labels.get(state, "LISTENING…")
    
    # Create HTML/CSS/JS payload
    html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=Montserrat:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                user-select: none;
            }}

            body, html {{
                width: 100%;
                height: 100%;
                background-color: #070312;
                font-family: 'Montserrat', sans-serif;
                overflow: hidden;
                color: #ffffff;
            }}

            /* Container & Cosmic Background */
            .vay-stage {{
                position: relative;
                width: 100%;
                height: 640px;
                background: radial-gradient(circle at 50% 45%, #230f3f 0%, #0d051a 60%, #05020a 100%);
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: space-between;
                padding: 30px 24px;
                border-radius: 16px;
                box-shadow: inset 0 0 80px rgba(0, 0, 0, 0.8), 0 10px 40px rgba(0, 0, 0, 0.6);
                border: 1px solid rgba(255, 255, 255, 0.08);
                overflow: hidden;
            }}

            /* Starfield & Nebula Canvas Background */
            #spaceCanvas {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: 1;
                pointer-events: none;
                opacity: 0.85;
            }}

            /* Header Section */
            .header-container {{
                position: relative;
                z-index: 10;
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
                margin-top: 10px;
            }}

            /* VAY Brand Logo with Metallic Chrome Effect */
            .brand-logo-wrap {{
                position: relative;
                display: inline-block;
                padding: 0 10px;
            }}

            .brand-title {{
                font-family: 'Cinzel', serif;
                font-size: 52px;
                font-weight: 700;
                letter-spacing: 12px;
                background: linear-gradient(180deg, #ffffff 0%, #e2d2f7 40%, #bca1e6 70%, #764eb8 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-shadow: 0 0 25px rgba(188, 161, 230, 0.4);
                line-height: 1;
            }}

            /* Circular Soundwave Motif over Brand */
            .brand-wave-svg {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 170px;
                height: 60px;
                pointer-events: none;
                opacity: 0.85;
                filter: drop-shadow(0 0 6px rgba(255, 110, 199, 0.6));
            }}

            .brand-subtitle {{
                font-family: 'Space Grotesk', sans-serif;
                font-size: 11px;
                font-weight: 500;
                letter-spacing: 5px;
                color: #a491ca;
                text-transform: uppercase;
                margin-top: 8px;
                text-shadow: 0 0 10px rgba(164, 145, 202, 0.3);
            }}

            .status-label {{
                font-family: 'Space Grotesk', sans-serif;
                font-size: 13px;
                font-weight: 600;
                letter-spacing: 4px;
                color: #e0d5ff;
                text-transform: uppercase;
                margin-top: 22px;
                text-shadow: 0 0 12px rgba(255, 255, 255, 0.6);
                transition: all 0.3s ease;
            }}

            .status-label.listening {{
                color: #ff6ec7;
                text-shadow: 0 0 16px rgba(255, 110, 199, 0.9);
                animation: statusPulse 1.8s infinite ease-in-out;
            }}

            .status-label.processing {{
                color: #4ef0d0;
                text-shadow: 0 0 16px rgba(78, 240, 208, 0.9);
            }}

            .status-label.responding {{
                color: #b18eff;
                text-shadow: 0 0 16px rgba(177, 142, 255, 0.9);
            }}

            @keyframes statusPulse {{
                0%, 100% {{ opacity: 1; transform: scale(1); }}
                50% {{ opacity: 0.7; transform: scale(0.98); }}
            }}

            /* Center Stage & Iridescent Orb Visualizer */
            .orb-stage {{
                position: relative;
                z-index: 10;
                display: flex;
                align-items: center;
                justify-content: center;
                width: 100%;
                height: 270px;
            }}

            /* Ambient Horizontal Waveforms Layer */
            #waveCanvas {
    position: absolute;
    top: 50%;
    left: 0;
    width: 100%;
    height: 180px;
    transform: translateY(-50%);
    z-index: 2;
}

            /* Glowing Iridescent 3D Orb Sphere */
            .iridescent-orb {{
                position: relative;
                z-index: 5;
                width: 210px;
                height: 210px;
                border-radius: 50%;
                background: radial-gradient(circle at 35% 30%, 
                            #ffffff 0%, 
                            #ff9ee2 22%, 
                            #ad6bff 45%, 
                            #4ee1f0 75%, 
                            #1a0933 100%);
                box-shadow: 
                    0 0 45px rgba(255, 110, 199, 0.6),
                    0 0 90px rgba(78, 225, 240, 0.4),
                    inset 0 0 35px rgba(255, 255, 255, 0.7),
                    inset -15px -15px 40px rgba(18, 5, 38, 0.85);
                transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
                cursor: pointer;
            }}

            /* Glass Highlight Layer */
            .iridescent-orb::before {{
                content: '';
                position: absolute;
                top: 8%;
                left: 15%;
                width: 55%;
                height: 40%;
                border-radius: 50%;
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.75) 0%, rgba(255, 255, 255, 0) 100%);
                transform: rotate(-25deg);
                pointer-events: none;
                filter: blur(1px);
            }}

            /* Outer Aura Ring */
            .iridescent-orb::after {{
                content: '';
                position: absolute;
                top: -12px;
                left: -12px;
                right: -12px;
                bottom: -12px;
                border-radius: 50%;
                background: radial-gradient(circle, rgba(255,110,199,0.3) 0%, rgba(78,225,240,0.15) 50%, rgba(0,0,0,0) 70%);
                z-index: -1;
                pointer-events: none;
                transition: all 0.5s ease;
            }}

            /* State-specific Orb Behaviors */
            
            /* IDLE: Normal size, soft resting pulse */
            .iridescent-orb.idle {{
                transform: scale(1);
                animation: orbFloat 4s ease-in-out infinite;
            }}

            /* LISTENING: Shrinks slightly, dramatic halo glow, breathing rhythm */
            .iridescent-orb.listening {{
                transform: scale(0.88);
                box-shadow: 
                    0 0 65px rgba(255, 75, 185, 0.9),
                    0 0 130px rgba(78, 225, 240, 0.75),
                    inset 0 0 45px rgba(255, 255, 255, 0.9),
                    inset -10px -10px 30px rgba(255, 45, 85, 0.4);
                animation: orbListenPulse 1.4s infinite ease-in-out;
            }}

            /* PROCESSING: Rapid iridescent color swirl */
            .iridescent-orb.processing {{
                transform: scale(0.95);
                box-shadow: 
                    0 0 70px rgba(78, 240, 208, 0.9),
                    0 0 120px rgba(177, 142, 255, 0.8),
                    inset 0 0 50px rgba(255, 255, 255, 0.9);
                animation: orbProcessSwirl 1.2s infinite linear;
            }}

            /* RESPONDING: Expands back with voice amplitude pulse */
            .iridescent-orb.responding {{
                transform: scale(1.06);
                box-shadow: 
                    0 0 80px rgba(188, 140, 255, 0.85),
                    0 0 140px rgba(255, 110, 199, 0.65),
                    inset 0 0 40px rgba(255, 255, 255, 0.85);
                animation: orbRespondBreath 1.8s infinite ease-in-out;
            }}

            @keyframes orbFloat {{
                0%, 100% {{ transform: translateY(0px) scale(1); }}
                50% {{ transform: translateY(-8px) scale(1.02); }}
            }}

            @keyframes orbListenPulse {{
                0%, 100% {{ 
                    transform: scale(0.86); 
                    box-shadow: 0 0 60px rgba(255, 60, 175, 0.95), 0 0 120px rgba(78, 225, 240, 0.8), inset 0 0 45px rgba(255, 255, 255, 0.9);
                }}
                50% {{ 
                    transform: scale(0.92); 
                    box-shadow: 0 0 85px rgba(255, 110, 199, 1), 0 0 150px rgba(140, 80, 255, 0.9), inset 0 0 55px rgba(255, 255, 255, 1);
                }}
            }}

            @keyframes orbProcessSwirl {{
                0% {{ filter: hue-rotate(0deg) contrast(1.1); transform: scale(0.94) rotate(0deg); }}
                50% {{ filter: hue-rotate(180deg) contrast(1.3); transform: scale(0.97) rotate(180deg); }}
                100% {{ filter: hue-rotate(360deg) contrast(1.1); transform: scale(0.94) rotate(360deg); }}
            }}

            @keyframes orbRespondBreath {{
                0%, 100% {{ transform: scale(1.02); filter: brightness(1); }}
                50% {{ transform: scale(1.10); filter: brightness(1.2); }}
            }}

            /* Bottom Caption & Response Subtitles */
            .caption-container {{
                position: relative;
                z-index: 10;
                width: 90%;
                max-width: 680px;
                text-align: center;
                min-height: 65px;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 0 10px;
            }}

            .response-caption {{
                font-family: 'Cormorant Garamond', serif;
                font-size: 23px;
                font-style: italic;
                font-weight: 500;
                line-height: 1.4;
                color: #f3e6cd;
                text-shadow: 0 2px 14px rgba(243, 230, 205, 0.3);
                transition: opacity 0.4s ease;
            }}

            /* Bottom Controls Section */
            .bottom-controls {{
                position: relative;
                z-index: 10;
                width: 100%;
                display: flex;
                align-items: center;
                justify-content: flex-end;
                gap: 16px;
                padding-right: 12px;
                padding-bottom: 5px;
            }}

            /* Action Buttons (Mic & X Close) */
            .action-btn {{
                width: 50px;
                height: 50px;
                border-radius: 50%;
                border: 1px solid rgba(255, 255, 255, 0.2);
                background: rgba(255, 255, 255, 0.06);
                backdrop-filter: blur(12px);
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                color: #e2d9f3;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            }}

            .action-btn:hover {{
                transform: translateY(-2px) scale(1.05);
                border-color: rgba(255, 255, 255, 0.4);
                background: rgba(255, 255, 255, 0.12);
            }}

            /* Mic Icon Button Active State: Turns RED */
            .mic-btn.active {{
                background: linear-gradient(135deg, #ff2d55 0%, #cc0033 100%) !important;
                border-color: #ff6b8b !important;
                color: #ffffff !important;
                box-shadow: 0 0 25px rgba(255, 45, 85, 0.9), 0 0 45px rgba(255, 45, 85, 0.5) !important;
                animation: micActivePulse 1.2s infinite ease-in-out;
            }}

            @keyframes micActivePulse {{
                0%, 100% {{ transform: scale(1); box-shadow: 0 0 20px rgba(255, 45, 85, 0.8); }}
                50% {{ transform: scale(1.08); box-shadow: 0 0 35px rgba(255, 45, 85, 1); }}
            }}

            /* X Close Button */
            .close-btn:hover {{
                background: rgba(255, 60, 60, 0.2);
                border-color: rgba(255, 100, 100, 0.5);
                color: #ff6b6b;
                box-shadow: 0 0 20px rgba(255, 80, 80, 0.4);
            }}

            .action-btn svg {{
                width: 22px;
                height: 22px;
                fill: currentColor;
            }}
        </style>
    </head>
    <body>
        <div class="vay-stage">
            <!-- Starfield Background -->
            <canvas id="spaceCanvas"></canvas>

            <!-- Top Header & Brand -->
            <div class="header-container">
                <div class="brand-logo-wrap">
                    <h1 class="brand-title">VAY</h1>
                    <!-- Circular Soundwave Motif -->
                    <svg class="brand-wave-svg" viewBox="0 0 200 80" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <ellipse cx="100" cy="40" rx="65" ry="24" stroke="url(#waveGrad)" stroke-width="2.5" stroke-dasharray="4 3"/>
                        <path d="M 40 40 Q 60 20 80 40 T 120 40 T 160 40" stroke="url(#lineGrad)" stroke-width="2" fill="none"/>
                        <defs>
                            <linearGradient id="waveGrad" x1="0" y1="0" x2="200" y2="0" gradientUnits="userSpaceOnUse">
                                <stop stop-color="#ff6ec7"/>
                                <stop offset="0.5" stop-color="#b18eff"/>
                                <stop offset="1" stop-color="#4ef0d0"/>
                            </linearGradient>
                            <linearGradient id="lineGrad" x1="0" y1="0" x2="200" y2="0" gradientUnits="userSpaceOnUse">
                                <stop stop-color="#4ef0d0"/>
                                <stop offset="1" stop-color="#ff6ec7"/>
                            </linearGradient>
                        </defs>
                    </svg>
                </div>
                <div class="brand-subtitle">MULTILINGUAL VOICE</div>
                <div id="statusLabel" class="status-label {state.lower()}">{status_display}</div>
            </div>

            <!-- Central Orb & Waveform Stage -->
            <div class="orb-stage">
                <canvas id="waveCanvas"></canvas>
                <div id="vayOrb" class="iridescent-orb {state.lower()}" onclick="toggleMicSession()"></div>
            </div>

            <!-- Live Response Transcript Subtitle -->
            <div class="caption-container">
                <p id="captionText" class="response-caption">“{caption_text}”</p>
            </div>

            <!-- Bottom Action Controls -->
            <div class="bottom-controls">
                <!-- Mic Button: Turns Red when active -->
                <button id="micBtn" class="action-btn mic-btn {'active' if state == 'LISTENING' else ''}" title="Toggle Microphone Listening" onclick="toggleMicSession()">
                    <svg viewBox="0 0 24 24">
                        <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
                        <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
                    </svg>
                </button>
                <!-- Close X Button: Hard stop voice session -->
                <button id="closeBtn" class="action-btn close-btn" title="End Session" onclick="hardStopSession()">
                    <svg viewBox="0 0 24 24">
                        <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
                    </svg>
                </button>
            </div>
        </div>

        <script>
            // State variable passed from Streamlit
            let currentAppState = "{state}";
            let currentLang = "{current_lang}";
            let captionContent = `{caption_text}`;
            let recognition = null;
            let isListening = (currentAppState === "LISTENING");

            // Web Speech API Language Code Mapping
            const langMap = {{
                'en': 'en-US',
                'es': 'es-ES',
                'fr': 'fr-FR',
                'de': 'de-DE',
                'hi': 'hi-IN',
                'ja': 'ja-JP'
            }};

            // Setup Starfield Canvas Animation
            const spaceCanvas = document.getElementById('spaceCanvas');
            const sCtx = spaceCanvas.getContext('2d');
            let stars = [];

            function resizeCanvas() {{
                spaceCanvas.width = spaceCanvas.offsetWidth;
                spaceCanvas.height = spaceCanvas.offsetHeight;
                waveCanvas.width = waveCanvas.offsetWidth;
                waveCanvas.height = waveCanvas.offsetHeight;
                initStars();
                initStrands();
            }}

            function initStars() {{
                stars = [];
                const numStars = 110;
                for (let i = 0; i < numStars; i++) {{
                    stars.push({{
                        x: Math.random() * spaceCanvas.width,
                        y: Math.random() * spaceCanvas.height,
                        radius: Math.random() * 1.4 + 0.3,
                        alpha: Math.random(),
                        speed: Math.random() * 0.015 + 0.005
                    }});
                }}
            }}

            function drawStars() {{
                sCtx.clearRect(0, 0, spaceCanvas.width, spaceCanvas.height);
                for (let star of stars) {{
                    star.alpha += star.speed;
                    if (star.alpha > 1 || star.alpha < 0) star.speed = -star.speed;
                    sCtx.beginPath();
                    sCtx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
                    sCtx.fillStyle = `rgba(255, 255, 255, ${{Math.abs(star.alpha)}})`;
                    sCtx.shadowBlur = 4;
                    sCtx.shadowColor = '#d9b3ff';
                    sCtx.fill();
                }}
                requestAnimationFrame(drawStars);
            }}

            // Setup React Bits "Strands" Animated Wave Engine
            const waveCanvas = document.getElementById('waveCanvas');
            const wCtx = waveCanvas.getContext('2d');
            let strands = [];
            let globalPhase = 0;

            const strandPalette = [
                '#ff6ec7', // Neon Pink
                '#4ef0d0', // Glowing Cyan
                '#b18eff', // Soft Violet
                '#3b82f6', // Electric Blue
                '#ff758c', // Coral Pink
                '#e0d5ff'  // Luminous Cream
            ];

            function initStrands() {{
                strands = [];
                const numStrands = 22; // Layered strand filaments
                for (let i = 0; i < numStrands; i++) {{
                    strands.push({{
                        color: strandPalette[i % strandPalette.length],
                        baseYOffset: (Math.random() - 0.5) * 40,
                        freq1: 0.006 + Math.random() * 0.014,
                        freq2: 0.012 + Math.random() * 0.018,
                        amp1: 12 + Math.random() * 26,
                        amp2: 6 + Math.random() * 14,
                        phase1: Math.random() * Math.PI * 2,
                        phase2: Math.random() * Math.PI * 2,
                        speed: 0.015 + Math.random() * 0.025,
                        width: 1.2 + Math.random() * 2.2,
                        alpha: 0.25 + Math.random() * 0.55,
                        particlePos: Math.random() * 0.8 + 0.1
                    }});
                }}
            }}

            function drawWaveforms() {{
                wCtx.clearRect(0, 0, waveCanvas.width, waveCanvas.height);
                const w = waveCanvas.width;
                const h = waveCanvas.height;
                const centerY = h / 2;

                const isListen = (currentAppState === "LISTENING");
                const isProcess = (currentAppState === "PROCESSING");
                const isResp = (currentAppState === "RESPONDING");

                // Dynamic speed & amplitude multiplier based on voice assistant state
                const speedMulti = isListen ? 2.6 : (isProcess ? 3.2 : (isResp ? 1.6 : 0.8));
                const ampMulti = isListen ? 2.2 : (isProcess ? 1.8 : (isResp ? 1.6 : 0.7));

                globalPhase += 0.02 * speedMulti;

                for (let i = 0; i < strands.length; i++) {{
                    const st = strands[i];
                    st.phase1 += st.speed * 0.02 * speedMulti;
                    st.phase2 += st.speed * 0.015 * speedMulti;

                    wCtx.beginPath();
                    wCtx.lineWidth = isListen ? st.width * 1.3 : st.width;

                    // Create smooth gradient stroke for each strand
                    const grad = wCtx.createLinearGradient(0, 0, w, 0);
                    grad.addColorStop(0, 'rgba(0,0,0,0)');
                    grad.addColorStop(0.2, st.color);
                    grad.addColorStop(0.5, '#ffffff');
                    grad.addColorStop(0.8, st.color);
                    grad.addColorStop(1, 'rgba(0,0,0,0)');

                    wCtx.strokeStyle = grad;
                    wCtx.globalAlpha = isListen ? Math.min(1.0, st.alpha * 1.4) : st.alpha;
                    wCtx.shadowBlur = isListen ? 16 : 8;
                    wCtx.shadowColor = st.color;

                    let prevX = 0;
                    let prevY = centerY;

                    for (let x = 0; x <= w; x += 4) {{
                        const normX = x / w;
                        // Envelope shape: tapers at screen edges
                        const env = Math.sin(normX * Math.PI);
                        
                        // Pinch funnel near central orb (normX ~ 0.5)
                        const distFromCenter = Math.abs(normX - 0.5);
                        const pinch = 1.0 - 0.35 * Math.exp(-Math.pow(distFromCenter / 0.18, 2));

                        // Multi-harmonic sinusoidal wave equation
                        const y1 = Math.sin(x * st.freq1 + st.phase1 + globalPhase) * st.amp1;
                        const y2 = Math.cos(x * st.freq2 - st.phase2) * st.amp2;
                        const y = centerY + (y1 + y2) * ampMulti * env * pinch + st.baseYOffset * env;

                        if (x === 0) {{
                            wCtx.moveTo(x, y);
                        }} else {{
                            wCtx.lineTo(x, y);
                        }}

                        prevX = x;
                        prevY = y;
                    }}
                    wCtx.stroke();

                    // Draw glowing particle pulses drifting along strands
                    st.particlePos = (st.particlePos + 0.003 * speedMulti) % 0.85 + 0.05;
                    const px = st.particlePos * w;
                    const pNormX = st.particlePos;
                    const pEnv = Math.sin(pNormX * Math.PI);
                    const pPinch = 1.0 - 0.35 * Math.exp(-Math.pow(Math.abs(pNormX - 0.5) / 0.18, 2));
                    const py1 = Math.sin(px * st.freq1 + st.phase1 + globalPhase) * st.amp1;
                    const py2 = Math.cos(px * st.freq2 - st.phase2) * st.amp2;
                    const py = centerY + (py1 + py2) * ampMulti * pEnv * pPinch + st.baseYOffset * pEnv;

                    wCtx.beginPath();
                    wCtx.arc(px, py, isListen ? 2.5 : 1.8, 0, Math.PI * 2);
                    wCtx.fillStyle = '#ffffff';
                    wCtx.shadowBlur = 12;
                    wCtx.shadowColor = st.color;
                    wCtx.fill();
                }}

                wCtx.globalAlpha = 1.0;
                requestAnimationFrame(drawWaveforms);
            }}


            // Web Speech API Initialization
            function initSpeech() {{
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                if (SpeechRecognition) {{
                    recognition = new SpeechRecognition();
                    recognition.continuous = false;
                    recognition.interimResults = true;
                    recognition.lang = langMap[currentLang] || 'en-US';

                    recognition.onstart = function() {{
                        updateUIState('LISTENING', 'LISTENING…', 'Listening for your query…');
                        sendStateToStreamlit('LISTENING', '');
                    }};

                    recognition.onresult = function(event) {{
                        let transcript = '';
                        for (let i = event.resultIndex; i < event.results.length; i++) {{
                            transcript += event.results[i][0].transcript;
                        }}
                        document.getElementById('captionText').innerText = `“${{transcript}}”`;
                        
                        if (event.results[0].isFinal) {{
                            updateUIState('PROCESSING', 'PROCESSING…', `Analyzing: "${{transcript}}"`);
                            sendStateToStreamlit('USER_SPEECH', transcript);
                        }}
                    }};

                    recognition.onerror = function(event) {{
                        console.warn('Speech recognition error:', event.error);
                        if (event.error !== 'no-speech') {{
                            updateUIState('IDLE', 'TAP MIC TO SPEAK', 'Mic access interrupted. Tap mic to retry.');
                        }}
                    }};

                    recognition.onend = function() {{
                        if (currentAppState === 'LISTENING') {{
                            updateUIState('PROCESSING', 'PROCESSING…', 'Processing transcript…');
                        }}
                    }};
                }}
            }}

            // Speech Synthesis (TTS Response Feedback)
            function speakText(text, lang) {{
                if ('speechSynthesis' in window) {{
                    window.speechSynthesis.cancel(); // Stop ongoing
                    const utterance = new SpeechSynthesisUtterance(text);
                    utterance.lang = langMap[lang] || 'en-US';
                    utterance.rate = 0.95;
                    utterance.pitch = 1.05;

                    utterance.onstart = function() {{
                        updateUIState('RESPONDING', 'RESPONDING…', text);
                    }};

                    utterance.onend = function() {{
                        updateUIState('IDLE', 'TAP MIC TO SPEAK', text);
                        sendStateToStreamlit('TTS_FINISHED', text);
                    }};

                    window.speechSynthesis.speak(utterance);
                }}
            }}

            // UI State Manager
            function updateUIState(state, statusText, captionTextVal) {{
                currentAppState = state;
                const orb = document.getElementById('vayOrb');
                const statusLabel = document.getElementById('statusLabel');
                const captionText = document.getElementById('captionText');
                const micBtn = document.getElementById('micBtn');

                orb.className = `iridescent-orb ${{state.toLowerCase()}}`;
                statusLabel.className = `status-label ${{state.toLowerCase()}}`;
                statusLabel.innerText = statusText;

                if (captionTextVal) {{
                    captionText.innerText = `“${{captionTextVal}}”`;
                }}

                if (state === 'LISTENING') {{
                    micBtn.classList.add('active');
                }} else {{
                    micBtn.classList.remove('active');
                }}
            }}

            // User Toggle Mic Button
            function toggleMicSession() {{
                if (currentAppState === 'LISTENING') {{
                    if (recognition) recognition.stop();
                    updateUIState('IDLE', 'TAP MIC TO SPEAK', captionContent);
                    sendStateToStreamlit('CANCEL', '');
                }} else {{
                    if (recognition) {{
                        try {{
                            recognition.lang = langMap[currentLang] || 'en-US';
                            recognition.start();
                        }} catch (e) {{
                            console.log('Recognition start retry:', e);
                            updateUIState('LISTENING', 'LISTENING…', 'Listening…');
                            sendStateToStreamlit('TRIGGER_LISTEN', '');
                        }}
                    }} else {{
                        // Fallback if browser SpeechRecognition disabled
                        updateUIState('LISTENING', 'LISTENING…', 'Listening… (Speak now)');
                        sendStateToStreamlit('TRIGGER_LISTEN', '');
                    }}
                }}
            }}

            // Hard Stop Session via X Button
            function hardStopSession() {{
                if (recognition) recognition.abort();
                if ('speechSynthesis' in window) window.speechSynthesis.cancel();
                updateUIState('IDLE', 'TAP MIC TO SPEAK', 'Session ended. Tap mic to begin.');
                sendStateToStreamlit('HARD_STOP', '');
            }}

            // Send messages back to parent Streamlit container
            function sendStateToStreamlit(actionType, payload) {{
                window.parent.postMessage({{
                    type: 'VAY_EVENT',
                    action: actionType,
                    payload: payload
                }}, '*');
            }}

            // Auto-trigger TTS if state is RESPONDING
            window.addEventListener('load', function() {{
                resizeCanvas();
                window.addEventListener('resize', resizeCanvas);
                drawStars();
                drawWaveforms();
                initSpeech();

                if (currentAppState === 'RESPONDING' && captionContent) {{
                    speakText(captionContent, currentLang);
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    # Render component with height 660px
    return components.html(html_code, height=660, scrolling=False)
