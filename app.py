import streamlit as st
import pickle
import string
import nltk
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
import time
from dotenv import load_dotenv

load_dotenv()


@st.cache_resource
def setup_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)


setup_nltk()

MODELS_DIR = "models/"
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

st.set_page_config(
    page_title="Cyberbullying Detector",
    page_icon="🛡️",
    layout="centered"
)

st.title("Cyberbullying Detection AI")
st.markdown("""
This application uses Machine Learning to detect if a piece of text contains cyberbullying or toxic content.
Enter your text below or use the YouTube Analysis tool to check video comments.
""")


@st.cache_resource
def load_resources():
    if not os.path.exists(VECTORIZER_PATH):
        return None, None

    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)

    models = {}
    if os.path.exists(MODELS_DIR):
        for file in os.listdir(MODELS_DIR):
            if file.endswith(".pkl") and file != "tfidf_vectorizer.pkl":
                model_name = file.replace(".pkl", "")
                models[model_name] = os.path.join(MODELS_DIR, file)

    return vectorizer, models


vectorizer, available_models = load_resources()

if not vectorizer:
    st.error(
        f"Vectorizer not found at {VECTORIZER_PATH}. Please run the 'model_training.ipynb' notebook first to generate the models.")
    st.stop()

if not available_models:
    st.error("No trained models found in 'models/' directory. Please run the 'model_training.ipynb' notebook first.")
    st.stop()

st.sidebar.title("Settings")
st.sidebar.markdown("Configure the detection engine.")

default_index = 0
model_names = list(available_models.keys())
if "LogisticRegression" in model_names:
    default_index = model_names.index("LogisticRegression")

selected_model_name = st.sidebar.selectbox("Select Model", model_names, index=default_index)
model_path = available_models[selected_model_name]


@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


classifier = load_model(model_path)


def advanced_clean_text(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


def preprocess_input(text):
    tokens = advanced_clean_text(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)


st.subheader("Text Analysis")
user_text = st.text_area("Enter message or comment:", height=150,
                         placeholder="Type something here to check for toxicity...",
                         help="The text you enter here will be processed locally.")

col1, col2 = st.columns([1, 4])
with col1:
    analyze_btn = st.button("Analyze Text", type="primary")

if analyze_btn:
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing content..."):
            try:
                processed_text = preprocess_input(user_text)

                if not processed_text:
                    st.warning(
                        "Input text contains only stopwords or punctuation. Please provide more meaningful text.")
                else:
                    input_vec = vectorizer.transform([processed_text])

                    prediction = classifier.predict(input_vec)[0]

                    confidence = 0.0
                    if hasattr(classifier, "predict_proba"):
                        probs = classifier.predict_proba(input_vec)[0]
                        confidence = max(probs)

                    st.markdown("---")

                    if prediction == 1:
                        st.error("Cyberbullying Detected")
                        st.markdown(f"**Verdict:** This text is classified as **TOXIC**.")
                        if confidence > 0:
                            st.metric("Confidence Level", f"{confidence:.2%}")
                    else:
                        st.success("Non-Toxic Content")
                        st.markdown(f"**Verdict:** This text appears to be **SAFE**.")
                        if confidence > 0:
                            st.metric("Confidence Level", f"{confidence:.2%}")

                    # Additional Details
                    with st.expander("View Technical Details"):
                        st.write("**Original Text:**", user_text)
                        st.write("**Processed Tokens:**", processed_text)
                        st.write("**Model Used:**", selected_model_name)
                        st.write("**Vector Dimensions:**", input_vec.shape)

            except Exception as e:
                st.error(f"An error occurred: {e}")

st.markdown("---")
st.subheader("YouTube Comment Analysis")
st.markdown("Analyze top comments from a YouTube video.")

yt_api_key = os.getenv("YOUTUBE_API_KEY")

yt_url = st.text_input("Enter YouTube Video URL")
if st.button("Fetch & Analyze Comments"):
    if not yt_url:
        st.error("Please provide a YouTube Video URL.")
    else:
        try:
            parsed_url = urlparse(yt_url)
            video_id = None
            if parsed_url.hostname == 'youtu.be':
                video_id = parsed_url.path[1:]
            elif parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
                if parsed_url.path == '/watch':
                    p = parse_qs(parsed_url.query)
                    video_id = p['v'][0]
                if parsed_url.path[:7] == '/embed/':
                    video_id = parsed_url.path.split('/')[2]
                if parsed_url.path[:3] == '/v/':
                    video_id = parsed_url.path.split('/')[2]

            if not video_id:
                st.error("Could not parse Video ID from URL.")
                st.stop()

            # Fetch Comments
            youtube = build('youtube', 'v3', developerKey=yt_api_key)

            comments_data = []
            next_page_token = None

            status_text = st.empty()
            status_text.text("Fetching comments...")

            while True:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=next_page_token,
                    textFormat="plainText"
                )
                response = request.execute()

                for item in response['items']:
                    top_comment = item['snippet']['topLevelComment']['snippet']
                    text = top_comment['textDisplay']
                    author = top_comment['authorDisplayName']
                    comments_data.append({'Author': author, 'Comment': text})

                status_text.text(f"Fetched {len(comments_data)} comments...")

                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break

            status_text.text(f"Completed! Total comments fetched: {len(comments_data)}")
            time.sleep(1)
            status_text.empty()

            results = []
            toxic_count = 0

            progress_bar = st.progress(0)

            for i, item in enumerate(comments_data):
                text = item['Comment']
                processed = preprocess_input(text)

                if processed:
                    vec = vectorizer.transform([processed])
                    pred = classifier.predict(vec)[0]
                    prob = 0.0
                    if hasattr(classifier, "predict_proba"):
                        prob = max(classifier.predict_proba(vec)[0])

                    label = "Toxic" if pred == 1 else "Non-Toxic"
                    if pred == 1:
                        toxic_count += 1

                    results.append({
                        'Author': item['Author'],
                        'Comment': text,
                        'Prediction': label,
                        'Confidence': f"{prob:.2f}"
                    })
                else:
                    results.append({
                        'Author': item['Author'],
                        'Comment': text,
                        'Prediction': "Skipped (Empty/Stopwords)",
                        'Confidence': "N/A"
                    })
                progress_bar.progress((i + 1) / len(comments_data))

            st.success(f"Analyzed {len(comments_data)} comments.")

            col1, col2 = st.columns(2)
            col1.metric("Total Comments Processed", len(comments_data))
            col2.metric("Toxic Comments Detected", toxic_count, delta_color="inverse")

            df_results = pd.DataFrame(results)

            st.subheader("Preview (Top 20 Comments)")
            st.caption("The table below shows a sample. Download the CSV for the full dataset.")
            st.dataframe(df_results.head(20))

            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Results as CSV",
                csv,
                "youtube_analysis_results.csv",
                "text/csv",
                key='download-csv'
            )

        except Exception as e:
            st.error(f"Error: {e}")
