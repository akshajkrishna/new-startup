#############################################
# AI Companion — Advanced MVP (single file)
# - Run: pip install -r requirements.txt
# - Then: streamlit run app.py
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

# Defensive imports for heavy libs (show clear error message if missing)
try:
    import cv2
except Exception as e:
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
st.set_page_config(page_title="AI Companion — Advanced MVP", layout="wide", initial_sidebar_state="expanded")
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
    # wait a moment to simulate startup
    time.sleep(1.2)
    st.session_state.prologue_shown = True
    st.experimental_rerun()

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
    st.write("Model choices and debug info below.")
    st.markdown("---")
    # show simple debug
    if st.checkbox("Show debug info"):
        st.write("Session keys:", list(st.session_state.keys()))

# -----------------------
# Model loaders (cached)
# -----------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline_safe():
    if pipeline is None:
        return None
    try:
        return pipeline("sentiment-analysis")
    except Exception as e:
        st.error(f"Failed to load sentiment pipeline: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_llm_safe(model_name="distilgpt2"):
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load LLM '{model_name}': {e}")
        return None, None

@st.cache_resource(show_spinner=False)
def load_tts_safe():
    if TTS is None:
        return None
    try:
        # small Coqui model (cpu)
        tts = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False, gpu=False)
        return tts
    except Exception as e:
        st.error(f"Failed to load TTS: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_fer_safe():
    if FER is None:
        return None
    try:
        detector = FER(mtcnn=True)
        return detector
    except Exception as e:
        st.error(f"Failed to load FER detector: {e}")
        return None

# instantiate models (may take time)
sentiment_nlp = load_sentiment_pipeline_safe()
tokenizer, llm_model = load_llm_safe()
tts_model = load_tts_safe()
fer_detector = load_fer_safe()

# -----------------------
# Helper functions
# -----------------------
def detect_text_emotion(text):
    """Estimate mood from text (fallback)."""
    override = st.session_state.get("emotion_override", "")
    if override:
        return override
    if not sentiment_nlp or not text:
        return "neutral"
    try:
        out = sentiment_nlp(text[:512])[0]
        label = out["label"].lower()
        # map HF sentiment to our moods
        if label == "positive":
            return "happy"
        return "sad"
    except Exception:
        return "neutral"

def detect_face_emotion_from_bytes(img_bytes):
    """Use FER to detect emotions from camera bytes."""
    if fer_detector is None or cv2 is None:
        return "neutral", {}
    try:
        arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = fer_detector.detect_emotions(rgb)
        if detections:
            emotions = detections[0]["emotions"]
            mood = max(emotions, key=emotions.get)
            return mood, emotions
        return "neutral", {}
    except Exception:
        return "neutral", {}

def generate_llm_response(prompt, max_length=160):
    """Generate a reply using the loaded LLM (distilgpt2 by default)."""
    # safety: short-circuit if model not loaded
    if tokenizer is None or llm_model is None:
        # fallback canned replies
        return "I heard you. Tell me more — I'm here to listen."
    try:
        input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
        out = llm_model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.92,
            temperature=0.85
        )
        reply = tokenizer.decode(out[0], skip_special_tokens=True)
        # strip possible echoed prompt
        if reply.lower().startswith(prompt.lower()):
            reply = reply[len(prompt):].strip()
        return reply or "Thanks for sharing — tell me more."
    except Exception as e:
        return "Sorry, I couldn't process that. Could you rephrase?"

def speak_reply_local(text):
    """Synthesize speech using Coqui TTS (local). Returns path to temp wav file or None."""
    if tts_model is None or sf is None:
        return None
    try:
        wav = tts_model.tts(text=text)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp.name, wav, samplerate=22050)
        return tmp.name
    except Exception as e:
        st.warning(f"TTS failed: {e}")
        return None

# -----------------------
# Eye animation HTML/CSS (SVG + CSS)
# -----------------------
EYES_HTML = """
<style>
:root { --bg: #07102a; --eye-bg: linear-gradient(90deg,#2b6cff,#00c7a7); }
.eyes-wrap { display:flex; gap:16px; justify-content:center; align-items:center; margin:6px 0 20px; }
.eye { width:84px; height:84px; border-radius:50%; background:linear-gradient(180deg,#111 0%, #222 100%); display:flex; align-items:center; justify-content:center; box-shadow:0 10px 30px rgba(0,0,0,0.6); transition: all .24s ease; }
.pupil { width:30px; height:30px; border-radius:50%; background:white; transform: translateY(0); transition: all .18s ease; box-shadow: inset 0 -6px 12px rgba(0,0,0,0.12); }
.eye.happy { transform: translateY(-6px) scale(1.02); background: linear-gradient(90deg,#12c2e9,#c471ed); }
.eye.happy .pupil { transform: translateY(-6px) translateX(6px); }
.eye.sad { transform: translateY(6px) scaleY(.92); background: linear-gradient(90deg,#223344,#334455); }
.eye.sad .pupil { transform: translateY(4px) translateX(-4px); width:22px; height:22px; opacity:0.95; }
.eye.angry { transform: rotate(-8deg) translateY(-4px); background: linear-gradient(90deg,#ff416c,#ff4b2b); }
.eye.angry .pupil { transform: translateY(-2px) translateX(-6px); }
.eye.anxious { animation: shake 0.6s infinite; background: linear-gradient(90deg,#f6d365,#fda085); }
.eye.surprise { transform: scale(1.12); background: linear-gradient(90deg,#66ccff,#66ffcc); }
@keyframes shake { 0%{transform:translateX(0)}25%{transform:translateX(-3px)}50%{transform:translateX(3px)}75%{transform:translateX(-2px)}100%{transform:translateX(0)} }
.small-note { text-align:center; color:#bcd6ff; font-size:13px; margin-top:6px; }
</style>
<div class="eyes-wrap">
  <div id="eyeLeft" class="eye neutral"><div class="pupil"></div></div>
  <div id="eyeRight" class="eye neutral"><div class="pupil"></div></div>
</div>
<script>
function setEyes(state){
  const left = document.getElementById('eyeLeft');
  const right = document.getElementById('eyeRight');
  const classes = ['happy','sad','angry','neutral','anxious','surprise'];
  classes.forEach(c => { left.classList.remove(c); right.classList.remove(c); });
  left.classList.add(state);
  right.classList.add(state);
}
// expose setter for streamlit
window.setEyes = setEyes;
</script>
"""

# Utility to call JS setEyes via html component
def render_eyes_component(mood):
    st.components.v1.html(EYES_HTML + f"<script>setEyes('{mood}')</script>", height=140)

# -----------------------
# Food scanner: Open Food Facts + robust parsing
# -----------------------
def query_openfoodfacts(query):
    # simple sanitizer
    q = requests.utils.requote_uri(query)
    url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={q}&search_simple=1&action=process&json=1&page_size=3"
    try:
        r = requests.get(url, timeout=8)
        data = r.json()
    except Exception:
        return []
    products = data.get("products", [])
    results = []
    for p in products:
        nutr = p.get("nutriments", {})
        results.append({
            "product_name": p.get("product_name") or p.get("brands") or "Unnamed",
            "cal_per_100g": nutr.get("energy-kcal_100g") or nutr.get("energy-kcal"),
            "protein_g": nutr.get("proteins_100g"),
            "carb_g": nutr.get("carbohydrates_100g"),
            "fat_g": nutr.get("fat_100g"),
            "raw": p
        })
    return results

# -----------------------
# Task manager & prioritizer (adaptive)
# -----------------------
def init_tasks():
    if "tasks" not in st.session_state:
        st.session_state.tasks = [
            {"id":1,"title":"Drink a glass of water","done":False,"base_priority":2,"history":0},
            {"id":2,"title":"Short 8-minute walk","done":False,"base_priority":3,"history":0},
            {"id":3,"title":"Journal: 5 minutes","done":False,"base_priority":1,"history":0},
        ]

def compute_priority(task, mood):
    score = task.get("base_priority", 3)
    # lower score => higher priority visually (we'll sort ascending)
    if mood in ["sad","anxious"]:
        # boost small wellness tasks
        if "journal" in task["title"].lower() or "walk" in task["title"].lower():
            score -= 2
    if mood in ["happy"]:
        # push social/productivity tasks up
        if "walk" in task["title"].lower():
            score -= 1
    # reduce priority if user repeatedly fails it (history)
    score += max(0, task.get("history",0)-1)*0.5
    return score

# -----------------------
# Safety / crisis detection
# -----------------------
CRISIS_KEYWORDS = ["suicid","kill myself","want to die","end my life","cant go on"]
def check_crisis(text):
    low = text.lower()
    for k in CRISIS_KEYWORDS:
        if k in low:
            return True
    return False

# -----------------------
# Initialize session
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"sender":"ai","text":"Hello — I'm here for you. Tell me how you're feeling today."}]
if "last_mood" not in st.session_state:
    st.session_state.last_mood = "neutral"
init_tasks()

# -----------------------
# Layout: 2 columns — left (chat), right (tools)
# -----------------------
left, right = st.columns([2.2, 1])

with left:
    st.markdown("### 💬 Companion Chat")
    # Render eyes (JS-based fancy eyes)
    render_eyes_component(st.session_state.get("last_mood","neutral"))

    # Display chat history
    for m in st.session_state.messages:
        if m["sender"] == "user":
            st.markdown(f"<div style='text-align:right; background:#072b63; color:#fff; padding:12px; border-radius:10px; margin:6px 0'>{m['text']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align:left; background:#e6f0ff; color:#07102a; padding:12px; border-radius:10px; margin:6px 0'>{m['text']}</div>", unsafe_allow_html=True)

    # Input area
    user_text = st.text_input("Say something to your companion...", key="chat_in")
    col_send, col_voice = st.columns([1,1])
    with col_send:
        if st.button("Send"):
            if user_text.strip():
                # save user message
                st.session_state.messages.append({"sender":"user","text":user_text})
                # check crisis
                if check_crisis(user_text):
                    st.error("Crisis language detected. Please contact local emergency services or a hotline immediately.")
                    # add safety instruction message
                    st.session_state.messages.append({"sender":"ai","text":"I detect distress — please contact local emergency services or a crisis hotline now. If you are in immediate danger, call your local emergency number."})
                else:
                    # determine mood: priority -> face (if set this session) > override > text sentiment
                    forced = st.session_state.get("emotion_override", "")
                    if forced:
                        mood = forced
                    elif st.session_state.get("face_mood", None):
                        mood = st.session_state["face_mood"]
                    else:
                        mood = detect_text_emotion(user_text)
                    st.session_state.last_mood = mood

                    # generate reply (LLM)
                    with st.spinner("Thinking..."):
                        reply = generate_llm_response(user_text)
                    st.session_state.messages.append({"sender":"ai","text":reply})
                    # update eyes via JS
                    render_eyes_component(mood)
                    # speak reply
                    wav = speak_reply_local(reply)
                    if wav:
                        st.audio(wav)
                    # increase task learning: if user completes tasks or mentions wins, adjust history
                    if any(win_word in user_text.lower() for win_word in ["done","completed","finished","yes","achieved"]):
                        # promote tasks completion signal
                        for t in st.session_state.tasks:
                            if not t["done"]:
                                t["history"] = max(0, t.get("history",0)-1)

    with col_voice:
        if st.button("Voice Input (disabled demo)"):
            st.info("Voice input feature is a planned enhancement — integrate WebRTC or browser speech recognition for full voice input.")

with right:
    st.markdown("### 🛠 Tools — Tasks, Food, Face Mood")
    st.markdown("#### Adaptive Task Manager")
    mood_now = st.session_state.get("last_mood","neutral")
    st.markdown(f"**Detected mood:** `{mood_now}`")
    # Render small eyes text
    st.markdown("<div style='font-size:12px;color:#bcd6ff;margin-bottom:8px'>Eyes reflect mood in the chat header.</div>", unsafe_allow_html=True)

    # Show tasks sorted by adaptive priority
    tasks_sorted = sorted(st.session_state.tasks, key=lambda t: compute_priority(t, mood_now))
    for t in tasks_sorted:
        cols = st.columns([6,1])
        with cols[0]:
            st.write(f"{'✅' if t['done'] else '🔲'} **{t['title']}** (priority {compute_priority(t, mood_now):.1f})")
        with cols[1]:
            if st.button("Done", key=f"task_done_{t['id']}"):
                t['done'] = True
                t['history'] = max(0, t.get("history",0)-1)
                st.success(f"Marked done: {t['title']}")
    st.markdown("Add a quick task:")
    new_task = st.text_input("Task title", key="new_task_box")
    if st.button("Add Task"):
        nid = max([t["id"] for t in st.session_state.tasks]) + 1 if st.session_state.tasks else 1
        st.session_state.tasks.append({"id":nid,"title":new_task,"done":False,"base_priority":3,"history":0})
        st.success("Task added")

    st.markdown("---")
    st.markdown("#### 🍎 Food Scanner")
    food_q = st.text_input("Search food / product (e.g., 'banana', 'coke 330ml')", key="food_query_box")
    if st.button("Lookup Nutrition"):
        if food_q.strip():
            results = query_openfoodfacts(food_q)
            if not results:
                st.info("No product found. Try a packaged product or more specific term.")
            else:
                for r in results[:3]:
                    st.markdown(f"**{r['product_name']}** — Calories/100g: `{r['cal_per_100g']}` | Protein: `{r['protein_g']}`g | Carbs: `{r['carb_g']}`g | Fat: `{r['fat_g']}`g")
                    # optional: show raw product image if available
                    raw = r.get("raw", {})
                    img_url = raw.get("image_small_url") or raw.get("image_url")
                    if img_url:
                        try:
                            st.image(img_url, width=120)
                        except Exception:
                            pass

    st.markdown("---")
    st.markdown("#### 📷 Face Mood Scanner")
    st.markdown("Use your webcam to take a selfie (Streamlit camera input).")
    img_file = st.camera_input("Take a selfie", key="face_camera")
    if img_file:
        img_bytes = img_file.getvalue()
        mood, details = detect_face_emotion_from_bytes(img_bytes)
        st.session_state["face_mood"] = mood
        st.session_state.last_mood = mood
        st.write(f"Detected mood: **{mood}**")
        st.json(details)

    st.markdown("---")
    st.markdown("#### Safety & Therapist")
    st.write("If the app detects crisis language it will prompt resources. For true human escalation integrate a therapist backend or scheduling flow (placeholder).")

# -----------------------
# Footer / developer notes
# -----------------------
st.markdown("---")
st.markdown("""
**Developer notes**  
- To improve chat quality, swap `distilgpt2` with a stronger, instruction-tuned LLM (be mindful of resources).  
- Lottie: replace eye HTML/CSS with Lottie web player for richer motion (drop in Lottie JSON and update JS).  
- Voice: Coqui TTS uses local models — you can pick different voices from the TTS model zoo.  
- Face scanner: FER uses MTCNN internally; on resource-constrained hosts it may be slow. Use smaller detectors or server-side GPU for production.  
- Privacy: This demo runs models locally in the container — no third-party text is sent to commercial APIs. Do not deploy for live medical use without compliance and clinician oversight.
""")
