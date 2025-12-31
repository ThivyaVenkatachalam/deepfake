import streamlit as st
from PIL import Image
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="AI Deepfake and Misinformation Detection Tool",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #f8f9fb;
}
.main-title {
    font-size: 46px;
    font-weight: 800;
}
.subtitle {
    font-size: 18px;
    color: #555;
}
.card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    transition: 0.3s ease-in-out;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
}
.section-title {
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 10px;
}
.badge {
    padding: 6px 14px;
    border-radius: 20px;
    background: #eef2ff;
    display: inline-block;
    margin-right: 10px;
}
.footer {
    text-align: center;
    color: gray;
    margin-top: 50px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='main-title'>AI Deepfake and Misinformation Detection Tool</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Detect fake media, misinformation, bot activity & election manipulation using AI</div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- NAVIGATION ----------------
tabs = st.tabs([
    "🔍 Analyze",
    "🎥 Video Deepfake",
    "🎙️ Audio Deepfake",
    "🤖 Social Media & Bots",
    "🌐 Multilingual",
    "🏛️ Integration"
])

# ---------------- HELPER FUNCTIONS ----------------
def bot_score(comments):
    if len(comments) < 2:
        return 0
    vec = TfidfVectorizer()
    X = vec.fit_transform(comments)
    sim = cosine_similarity(X)
    return min((np.sum(sim > 0.8) - len(comments)) * 10, 100)

def url_score(url):
    score = 0
    if len(url) > 75: score += 20
    if not url.startswith("https"): score += 20
    if re.search("login|otp|bank|verify|free|update", url.lower()): score += 30
    return min(score, 100)

# ---------------- TAB 1 ----------------
with tabs[0]:
    st.markdown("<div class='section-title'>Verify Media Content</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        media = st.selectbox("Select Media Type", ["Image", "URL", "Video", "Audio"])

        if media == "Image":
            img = st.file_uploader("Upload Image", type=["jpg","png"])
        elif media == "URL":
            url = st.text_input("Paste Website / Link")
        else:
            st.file_uploader(f"Upload {media}", disabled=True)

        comments = st.text_area("Paste Comments (one per line)", height=120)

        if st.button("Analyze Content"):
            score = 0
            if media == "URL":
                score += url_score(url)
            if comments:
                score += bot_score([c for c in comments.split("\n") if c])

            st.markdown("### Result")
            st.progress(score/100)
            st.write(f"Confidence Score: **{score}%**")

            if score < 40:
                st.success("✅ Real Content")
            elif score < 70:
                st.warning("⚠️ Suspicious Content")
            else:
                st.error("❌ Likely Fake / Misinformation")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- TAB 2 ----------------
with tabs[1]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🎥 Video Deepfake Detection")
    st.markdown("""
    **Planned AI Techniques**
    - Frame-level artifact detection  
    - Facial landmark inconsistencies  
    - Lip-sync mismatch analysis  
    - Temporal coherence checks  
    """)
    st.info("Will use CNN + LSTM models for real-time detection.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- TAB 3 ----------------
with tabs[2]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🎙️ Audio & Voice Cloning Detection")
    st.markdown("""
    **Detection Pipeline**
    - Voiceprint matching  
    - Spectrogram anomaly detection  
    - AI-generated voice markers  
    """)
    st.info("Planned integration with pretrained audio deepfake models.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- TAB 4 ----------------
with tabs[3]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🤖 Social Media & Bot Monitoring")
    st.markdown("""
    - Detect coordinated comment attacks  
    - Election manipulation alerts  
    - Fake trend identification  
    """)
    demo = st.text_area("Try sample comments")
    if st.button("Check Bot Activity"):
        score = bot_score([c for c in demo.split("\n") if c])
        st.metric("Bot Probability", f"{score}%")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- TAB 5 ----------------
with tabs[4]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🌐 Multilingual Bot Detection")
    st.markdown("""
    **Languages**
    - English  
    - Tamil  
    - Hindi  

    Uses multilingual NLP & cross-language similarity.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- TAB 6 ----------------
with tabs[5]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🏛️ Government & Mobile Integration")
    st.markdown("""
    - Cyber Crime Portal integration  
    - Election Commission monitoring  
    - Mobile app for elderly users  
    - Voice-based alerts & warnings  
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("<div class='footer'>© 2025 AI Deepfake & Misinformation Detection Tool</div>", unsafe_allow_html=True)
