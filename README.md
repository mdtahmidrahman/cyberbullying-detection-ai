# Cyberbullying Detection AI

A Streamlit web application that detects cyberbullying and toxic content in text and YouTube video comments using machine‑learning models.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Environment Variables](#environment-variables)
- [Model Training](#model-training)
- [Running the Application](#running-the-application)
- [YouTube Comment Analysis](#youtube-comment-analysis)
- [Dependencies](#dependencies)

---

## Overview

The **Cyberbullying Detection AI** project provides a simple UI for users to:
1. Paste any text and get an instant toxicity prediction.
2. Enter a YouTube video URL and analyse the top comments for toxic content.

The backend uses a TF‑IDF vectoriser and several pre‑trained classification models (e.g., Logistic Regression, SVM). Models are trained in the `model_training.ipynb` notebook and stored as pickle files under the `models/` directory.

---

## Features

- **Text analysis** – Real‑time prediction with confidence scores.
- **YouTube analysis** – Fetches all comments, processes them, and provides a downloadable CSV of results.
- **Model selection** – Choose from any trained model via a sidebar dropdown.
- **Technical details** – Expandable view shows original text, processed tokens, model used, and vector dimensions.

---

## Project Structure

```
cyberbullying-detection/
├───.venv/
├───models/
│───data/
│───app.py
│───model_training.ipynb
│───.env
│───requirements.txt
└───README.md
```

---

## Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mdtahmidrahman/cyberbullying-detection.git
   cd cyberbullying-detection
   ```
2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # on Windows
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download NLTK resources** (the app does this automatically on first run, but you can pre‑download)
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
   ```
5. **Set up environment variables** – create a `.env` file in the project root containing:
   ```
   YOUTUBE_API_KEY=YOUR_YOUTUBE_DATA_API_KEY
   ```
   The file is ignored by Git (`.gitignore`).

---

## Environment Variables

- `YOUTUBE_API_KEY` – Required for the YouTube comment analysis feature. Obtain a key from the Google Cloud Console (YouTube Data API v3).

---

## Model Training

The models are not shipped with the repository. To generate them:
1. Open `model_training.ipynb` in Jupyter or VS Code.
2. Run all cells – the notebook will:
   - Load the dataset.
   - Pre‑process text.
   - Train several classifiers.
   - Save the TF‑IDF vectoriser to `models/tfidf_vectorizer.pkl`.
   - Save each trained model as `<model_name>.pkl` in the `models/` folder.
3. After training, you should see files like `LogisticRegression.pkl`, `DecisionTreeClassifier.pkl`, etc.

> **Note:** The Streamlit app will refuse to start if the vectoriser or at least one model is missing, displaying an informative error.

---

## Running the Application

```bash
streamlit run app.py
```

The app will launch in your default browser at `http://localhost:8501`. Use the sidebar to select a model, then:
- **Text Analysis** – Paste text and click **Analyze Text**.
- **YouTube Comment Analysis** – Enter a YouTube URL and click **Fetch & Analyze Comments**.

---

## YouTube Comment Analysis

The workflow:
1. Parse the video ID from the provided URL.
2. Use the YouTube Data API to fetch comment threads
3. Pre‑process each comment and predict toxicity.
4. Show a summary (total comments, toxic count) and a preview table of the first 20 comments.
5. Offer a **Download Results as CSV** button for the full dataset.

---

## Dependencies

Key Python packages (see `requirements.txt`):
- `streamlit`
- `pandas`
- `scikit-learn`
- `nltk`
- `python-dotenv`
- `google-api-python-client`
