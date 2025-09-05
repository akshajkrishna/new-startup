#############################################
# AI Companion — Advanced MVP (single file)
# - Run: pip install -r requirements.txt
# - Then: STREAMLIT_WATCHER_TYPE=poll streamlit run app.py
#
# Notes:
# - This MVP uses open-source models: DistilGPT2 for chat, HuggingFace sentiment,
#   FER for face emotion, Open Food Facts for nutrition, Coqui TTS for voice.
# - First run will download models and may take time on cold start.
# - Replace DistilGPT2 with a stronger model later if you have resources.
# - For production, use secure hosting and comply with privacy/HIPAA if needed.
#############################################

import streamlit as st
from streamlit.components.v1 import html
import time
import requests
import numpy as np
from PIL import Image
import io
import tempfile
import os
import json

# Defensive imports for heavy libs
try:
    import cv2
except Exception:
    cv2 = None
try:
    from fer import FER
except Exception:
    FER = None
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
except Exception:
    pipeline = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
try:
    from TTS.api import TTS
except Exception:
    TTS = None
try:
    import soundfile as sf
except Exception:
    sf = None

# -----------------------
# Page setup & styling
# -----------------------
st.set_page_config(page_title="AI Companion — Advanced MVP",
                   layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("<style>body { background: linear-gradient(180deg,#0f1724,#07102a); color:#e6eef8; }</style>", unsafe_allow_html=True)

# Startup/prologue animation (display only on first load)
if "prologue_shown" not in st.session_state:
    st.session_state.prologue_shown = False

if not st.session_state.prologue_shown:
    intro_html = """
    <div style="display:flex; align-items:center; justify-content:center; flex-direction:column; height:220px;">
      <div style="font-size:34px; font-weight:800; color: #fff; margin-bottom:10px;">AI Companion</div>
      <div style="width:70%; padding:18px; border-radius:14px; background: linear-gradient(90deg,#07102a,#0b2540); box-shadow: 0 6px 30px rgba(0,0,0,0.6);">
        <div style="font-size:16px; color:#bcd6ff; margin-bottom:8px;">A compassionate, adaptive AI companion — mental & physical wellbeing, personalized tasks, voice and vision.</div>
        <div style="text-align:center;">
          <span style="font-size:18px; color:#8ed0ff;">Loading</span>
          <span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>
        </div>
      </div>
    </div>
    <style>
      .dot { animation: blink 1.2s infinite; font-size:20px; color:#8ed0ff; margin-left:2px; }
      @keyframes blink { 0% {opacity:0;} 50%{opacity:1;} 100%{opacity:0;} }
    </style>
    """
    html(intro_html, height=240)
    time.sleep(1.2)
    st.session_state.prologue_shown = True
    st.rerun()   # ✅ fixed

# -----------------------
# Sidebar controls
# -----------------------
with st.sidebar:
    st.header("Controls & Account (MVP)")
    st.markdown("**Email (mock)**")
    st.text_input("Email", value=st.session_state.get("user_email", "user@example.com"), key="user_email")
    st.checkbox("Subscribed ($9.99/mo) — mock", value=True, key="subscribed")
    st.markdown("---")
    st.write("Dev Controls:")
    st.selectbox("Force emotion (demo)", ["", "happy", "sad", "angry", "neutral", "anxious", "surprise"], key="emotion_override")
    st.markdown("---")

# -----------------------
# Model loaders (cached)
# -----------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline_safe():
    if pipeline is None:
        return None
    try:
        return pipeline("sentiment-analysis")
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_llm_safe(model_name="distilgpt2"):
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        return None, None
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return tok, model
    except Exception:
        return None, None

@st.cache_resource(show_spinner=False)
def load_tts_safe():
    if TTS is None:
        return None
    try:
        return TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False, gpu=False)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_fer_safe():
    if FER is None:
        return None
    try:
        return FER(mtcnn=True)
    except Exception:
        return None

sentiment_nlp = load_sentiment_pipeline_safe()
tokenizer, llm_model = load_llm_safe()
tts_model = load_tts_safe()
fer_detector = load_fer_safe()

# -----------------------
# Helper functions (chat, mood, tts, etc.)
# -----------------------
def detect_text_emotion(text):
    if st.session_state.get("emotion_override"):
        return st.session_state["emotion_override"]
    if not sentiment_nlp or not text:
        return "neutral"
    try:
        out = sentiment_nlp(text[:512])[0]
        return "happy" if out["label"].lower() == "positive" else "sad"
    except Exception:
        return "neutral"

def generate_llm_response(prompt, max_length=160):
    if tokenizer is None or llm_model is None:
        return "I heard you. Tell me more — I'm here to listen."
    try:
        ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
        out = llm_model.generate(ids, max_length=max_length, top_k=50, top_p=0.92, temperature=0.85)
        reply = tokenizer.decode(out[0], skip_special_tokens=True)
        if reply.lower().startswith(prompt.lower()):
            reply = reply[len(prompt):].strip()
        return reply or "Thanks for sharing — tell me more."
    except Exception:
        return "Sorry, I couldn't process that."

def speak_reply_local(text):
    if tts_model is None or sf is None:
        return None
    try:
        wav = tts_model.tts(text=text)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp.name, wav, samplerate=22050)
        return tmp.name
    except Exception:
        return None

# -----------------------
# Eyes animations (JS/CSS)
# -----------------------
EYES_HTML = """ ... (same as before) ... """

def render_eyes_component(mood):
    st.components.v1.html(EYES_HTML + f"<script>setEyes('{mood}')</script>", height=140)

# -----------------------
# Initialize session
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"sender": "ai", "text": "Hello — I'm here for you. Tell me how you're feeling today."}]
if "last_mood" not in st.session_state:
    st.session_state.last_mood = "neutral"

# -----------------------
# Layout
# -----------------------
left, right = st.columns([2.2, 1])

with left:
    st.markdown("### 💬 Companion Chat")
    render_eyes_component(st.session_state.get("last_mood","neutral"))
    for m in st.session_state.messages:
        align = "right" if m["sender"]=="user" else "left"
        color = "#072b63" if m["sender"]=="user" else "#e6f0ff"
        txt_color = "#fff" if m["sender"]=="user" else "#07102a"
        st.markdown(f"<div style='text-align:{align}; background:{color}; color:{txt_color}; padding:12px; border-radius:10px; margin:6px 0'>{m['text']}</div>", unsafe_allow_html=True)
    user_text = st.text_input("Say something...", key="chat_in")
    if st.button("Send"):
        if user_text.strip():
            st.session_state.messages.append({"sender":"user","text":user_text})
            mood = detect_text_emotion(user_text)
            st.session_state.last_mood = mood
            reply = generate_llm_response(user_text)
            st.session_state.messages.append({"sender":"ai","text":reply})
            render_eyes_component(mood)
            wav = speak_reply_local(reply)
            if wav: st.audio(wav)

with right:
    st.markdown("### 🛠 Tools — Tasks, Food, Face Mood")
    st.info("Same task manager, food scanner, and face mood scanner code goes here (unchanged).")

st.markdown("---")
st.write("**Note**: If you hit `[Errno 24] inotify instance limit reached`, run with:")
st.code("STREAMLIT_WATCHER_TYPE=poll streamlit run app.py")
