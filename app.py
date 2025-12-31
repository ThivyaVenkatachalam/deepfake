import streamlit as st
from PIL import Image
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Deepfake and Misinformation Detection Tool",
    layout="centered"
)

# ---------------- TITLE ----------------
st.title("AI Deepfake and Misinformation Detection Tool")

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Analyze Content",
    "🤖 Bot Detection",
    "ℹ️ About",
    "🚀 Future Scope"
])

# ---------------- FUNCTIONS ----------------

def image_score(image):
    score = 0
    # Metadata check
    if not image.info:
        score += 30
    # Simulated AI artifact detection
    score += 30
    return score


def url_score(url):
    score = 0
    if len(url) > 75:
        score += 20
    if not url.startswith("https"):
        score += 20
    if re.search(r"login|verify|otp|free|update|bank", url.lower()):
        score += 30
    if re.match(r"https?://\d+\.\d+\.\d+\.\d+", url):
        score += 30
    return min(score, 100)


def bot_score(comments):
    if len(comments) < 2:
        return 0
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(comments)
    similarity = cosine_similarity(X)
    similar_pairs = np.sum(similarity > 0.8) - len(comments)
    return min(similar_pairs * 10, 100)


def election_context(text):
    keywords = [
        "vote", "election", "party", "pm", "government",
        "bjp", "congress", "dmk", "aiadmk"
    ]
    return any(word in text.lower() for word in keywords)

# ---------------- TAB 1: ANALYZE CONTENT ----------------
with tab1:
    st.subheader("Verify Media Content")

    media_type = st.selectbox(
        "Select Media Type",
        ["Image", "Video", "Audio", "URL"]
    )

    uploaded_file = None
    url_input = ""

    if media_type == "Image":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])
    elif media_type == "Video":
        uploaded_file = st.file_uploader("Upload Video", type=["mp4"])
    elif media_type == "Audio":
        uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])
    elif media_type == "URL":
        url_input = st.text_input("Paste Website / Link")

    comments_text = st.text_area(
        "Paste Comments (one per line)",
        height=150,
        placeholder="Vote for X now!\nVote for X now!\nThis leader will save the country"
    )

    if st.button("Analyze Content"):
        total_score = 0

        if media_type == "Image" and uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            total_score += image_score(image)

        elif media_type == "URL" and url_input:
            total_score += url_score(url_input)

        elif media_type in ["Video", "Audio"]:
            st.info("Audio / Video analysis is under development (simulated result).")
            total_score += 40

        comments = [c for c in comments_text.split("\n") if c.strip()]
        total_score += bot_score(comments)

        if election_context(comments_text):
            total_score += 20

        total_score = min(total_score, 100)

        st.subheader("Final Result")
        st.write("Confidence Score:", total_score, "%")

        if total_score < 40:
            st.success("✅ REAL CONTENT")
        elif total_score < 70:
            st.warning("⚠️ SUSPICIOUS CONTENT")
        else:
            st.error("❌ LIKELY FAKE / MISINFORMATION")

# ---------------- TAB 2: BOT DETECTION ----------------
with tab2:
    st.subheader("Bot & Coordinated Comment Detection")

    st.write("""
    This module detects **automated or coordinated bot activity**, commonly used in
    misinformation and election manipulation campaigns.
    
    **Indicators used:**
    - Repeated comments
    - High text similarity
    - Coordinated messaging patterns
    """)

    demo_comments = st.text_area(
        "Enter comments to check bot activity (one per line):",
        height=200
    )

    if st.button("Check Bot Activity"):
        comments = [c for c in demo_comments.split("\n") if c.strip()]
        score = bot_score(comments)

        st.write("Bot Probability:", score, "%")

        if score > 60:
            st.error("🤖 High Bot Activity Detected")
        else:
            st.success("✅ No Significant Bot Activity Detected")

# ---------------- TAB 3: ABOUT ----------------
with tab3:
    st.subheader("About This Project")

    st.write("""
    ### Problem
    AI-generated deepfakes, fake voice calls, and malicious links are increasingly used
    for fraud, impersonation, and election misinformation. Elderly and non-technical users
    are especially vulnerable.

    ### Our Solution
    The **AI Deepfake and Misinformation Detection Tool** verifies:
    - Images
    - URLs
    - Behavioral patterns in comments

    It provides a **clear verdict** with a **confidence score**, making risk assessment
    simple and understandable.

    ### Target Users
    - Elderly users
    - General public
    - Cybercrime investigators
    - Election monitoring authorities
    """)

# ---------------- TAB 4: FUTURE SCOPE ----------------
with tab4:
    st.subheader("Future Scope & Enhancements")

    st.markdown("""
    🔹 Video deepfake detection (frame & lip-sync analysis)  
    🔹 Audio deepfake & AI voice cloning detection  
    🔹 Real-time social media monitoring  
    🔹 Multilingual bot detection (Tamil, Hindi, English)  
    🔹 Mobile app for elderly users  
    🔹 Integration with Cyber Crime & Election Commission systems
    """)

    st.info("This prototype demonstrates core detection logic. Advanced AI models will be integrated in future phases.")
