import streamlit as st
from PIL import Image
import cv2
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Deepfake Detection", layout="centered")

st.title("AI Deepfake & Election Misinformation Detection Tool")

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
    uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])
elif media_type == "URL":
    url_input = st.text_input("Paste URL")

comments_text = st.text_area(
    "Paste Comments (one per line)",
    height=150
)

# ---------- FUNCTIONS ----------

def image_score(image):
    score = 0
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur < 100:
        score += 25
    if not image.info:
        score += 30
    return score

def url_score(url):
    score = 0
    if len(url) > 75:
        score += 20
    if not url.startswith("https"):
        score += 20
    if re.search(r"login|verify|otp|free|update", url.lower()):
        score += 30
    if re.match(r"https?://\d+\.\d+\.\d+\.\d+", url):
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

def election_context(text):
    keywords = ["vote", "election", "party", "pm", "bjp", "congress", "dmk"]
    return any(word in text.lower() for word in keywords)

# ---------- ANALYZE ----------

if st.button("Analyze"):
    total_score = 0

    if media_type == "Image" and uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        total_score += image_score(img)

    elif media_type == "URL" and url_input:
        total_score += url_score(url_input)

    elif media_type in ["Video", "Audio"]:
        st.info("This media analysis is under development (simulated output).")
        total_score += 40

    comments = [c for c in comments_text.split("\n") if c.strip()]
    total_score += bot_score(comments)

    if election_context(comments_text):
        total_score += 20

    total_score = min(total_score, 100)

    st.subheader("Final Result")
    st.write("Confidence Score:", total_score, "%")

    if total_score < 40:
        st.success("REAL CONTENT")
    elif total_score < 70:
        st.warning("SUSPICIOUS CONTENT")
    else:
        st.error("LIKELY FAKE / MISINFORMATION")
