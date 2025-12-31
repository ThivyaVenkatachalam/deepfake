import streamlit as st
from PIL import Image
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Deepfake and Misinformation Detection Tool",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🛡️ AI Verification System")
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Analyze Content",
        "🎥 Video Deepfake Detection",
        "🎙️ Audio Deepfake Detection",
        "🤖 Bot & Social Media Monitoring",
        "🌐 Multilingual Bot Detection",
        "ℹ️ About & Integration"
    ]
)

# ---------------- FUNCTIONS ----------------
def image_score(image):
    score = 0
    if not image.info:
        score += 30
    score += 30  # simulated AI artifacts
    return score

def url_score(url):
    score = 0
    if len(url) > 75:
        score += 20
    if not url.startswith("https"):
        score += 20
    if re.search(r"login|verify|otp|free|update|bank", url.lower()):
        score += 30
    return min(score, 100)

def bot_score(comments):
    if len(comments) < 2:
        return 0
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(comments)
    sim = cosine_similarity(X)
    similar = np.sum(sim > 0.8) - len(comments)
    return min(similar * 10, 100)

# ---------------- HOME ----------------
if page == "🏠 Home":
    st.markdown("## 🛡️ AI Deepfake and Misinformation Detection Tool")

    st.markdown("""
    A unified platform to detect **AI-generated media, fake links, bot-driven misinformation,
    and election manipulation**, designed especially for **elderly and non-technical users**.
    """)

    col1, col2, col3 = st.columns(3)
    col1.success("✔ Image & URL Verification")
    col2.warning("✔ Bot & Comment Analysis")
    col3.info("✔ Election & Misinformation Context")

    st.markdown("---")
    st.markdown("### 🔎 Supported Media")
    st.write("Images • Videos • Audio • URLs • Social Media Comments")

# ---------------- ANALYZE CONTENT ----------------
elif page == "🔍 Analyze Content":
    st.header("🔍 Analyze Media Content")

    media_type = st.selectbox("Select Media Type", ["Image", "URL", "Video", "Audio"])

    uploaded_file = None
    url_input = ""

    if media_type == "Image":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])
    elif media_type == "URL":
        url_input = st.text_input("Paste URL")
    elif media_type in ["Video", "Audio"]:
        uploaded_file = st.file_uploader(f"Upload {media_type}", type=["mp4", "mp3", "wav"])

    comments_text = st.text_area(
        "Paste Comments (optional – one per line)",
        height=150
    )

    if st.button("Analyze"):
        score = 0

        if media_type == "Image" and uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            score += image_score(image)

        elif media_type == "URL" and url_input:
            score += url_score(url_input)

        elif media_type in ["Video", "Audio"]:
            st.info("Advanced deepfake analysis under development.")
            score += 40

        comments = [c for c in comments_text.split("\n") if c.strip()]
        score += bot_score(comments)

        score = min(score, 100)

        st.markdown("### 📊 Result")
        st.metric("Confidence Score", f"{score}%")

        if score < 40:
            st.success("✅ Real Content")
        elif score < 70:
            st.warning("⚠️ Suspicious Content")
        else:
            st.error("❌ Likely Fake / Misinformation")

# ---------------- VIDEO ----------------
elif page == "🎥 Video Deepfake Detection":
    st.header("🎥 Video Deepfake Detection")

    st.markdown("""
    **Planned Detection Techniques:**
    - Frame-by-frame analysis
    - Face landmark inconsistencies
    - Lip-sync mismatch detection
    - Temporal artifact detection
    """)

    st.info("This module will use CNN + temporal models (future phase).")

# ---------------- AUDIO ----------------
elif page == "🎙️ Audio Deepfake Detection":
    st.header("🎙️ Audio & Voice Cloning Detection")

    st.markdown("""
    **Detection Approach:**
    - Voiceprint comparison
    - Spectrogram anomaly detection
    - AI voice cloning markers
    """)

    st.info("Integration with pretrained audio deepfake models planned.")

# ---------------- BOT & SOCIAL ----------------
elif page == "🤖 Bot & Social Media Monitoring":
    st.header("🤖 Bot & Real-Time Social Media Monitoring")

    st.markdown("""
    **Capabilities:**
    - Detect coordinated comments
    - Identify bot-like repetition
    - Election misinformation alerts
    """)

    demo = st.text_area("Try sample comments:", height=200)

    if st.button("Check Bot Activity"):
        comments = [c for c in demo.split("\n") if c.strip()]
        score = bot_score(comments)
        st.metric("Bot Probability", f"{score}%")

        if score > 60:
            st.error("🤖 High Bot Activity Detected")
        else:
            st.success("✅ Normal User Activity")

# ---------------- MULTILINGUAL ----------------
elif page == "🌐 Multilingual Bot Detection":
    st.header("🌐 Multilingual Bot Detection")

    st.markdown("""
    **Supported Languages (Planned):**
    - English
    - Tamil
    - Hindi

    **Approach:**
    - Language detection
    - Keyword & sentiment analysis
    - Cross-language bot similarity
    """)

    st.info("Multilingual NLP models will be integrated in future phases.")

# ---------------- ABOUT ----------------
elif page == "ℹ️ About & Integration":
    st.header("ℹ️ About & Government Integration")

    st.markdown("""
    **Target Users:**
    - Elderly citizens
    - General public
    - Cyber Crime Units
    - Election Commission

    **Planned Integrations:**
    - Cyber Crime Portal
    - Election Commission monitoring systems
    - Mobile app for elderly users (simple UI, voice alerts)
    """)

    st.markdown("🔗 https://cybercrime.gov.in")
