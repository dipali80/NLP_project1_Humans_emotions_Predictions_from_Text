


import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🧠",
    layout="centered"
)

# Download stopwords (only first time)
nltk.download('stopwords')

# -------------------------------
# Load Dataset
# -------------------------------
data = pd.read_csv(
    "TEXT_DATA/train.txt",
    sep=";",
    header=None,
    names=["text", "emotion"]
)

# -------------------------------
# Label Encoding
# -------------------------------
unique_emotions = data["emotion"].unique()
emotion_numbers = {emo: i for i, emo in enumerate(unique_emotions)}
data["emotion"] = data["emotion"].map(emotion_numbers)

# Reverse mapping (number → emotion)
reverse_emotion_map = {v: k for k, v in emotion_numbers.items()}

# -------------------------------
# Text Cleaning Function
# -------------------------------
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([i for i in text if not i.isdigit()])
    text = ''.join([i for i in text if i.isascii()])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

data["text"] = data["text"].apply(clean_text)

# -------------------------------
# TF-IDF Vectorization
# -------------------------------
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data["text"])
y = data["emotion"]

# -------------------------------
# Train Logistic Regression
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# -------------------------------
# Streamlit UI
# -------------------------------

st.markdown("<h1 style='text-align: center;'>Human Emotion Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict emotions from text using NLP and Machine Learning</p>", unsafe_allow_html=True)

st.divider()

user_input = st.text_area("Enter your sentence below:", height=150)

if st.button("Analyze Emotion"):
    if user_input.strip() != "":
        cleaned_input = clean_text(user_input)
        vector_input = tfidf.transform([cleaned_input])

        prediction = model.predict(vector_input)[0]
        probabilities = model.predict_proba(vector_input)[0]

        emotion_label = reverse_emotion_map[prediction]
        confidence = max(probabilities) * 100

        st.success(f"Predicted Emotion: {emotion_label}")
        st.info(f"Confidence Score: {confidence:.2f}%")

        st.subheader("Emotion Probability Distribution")

        prob_df = pd.DataFrame({
            "Emotion": list(reverse_emotion_map.values()),
            "Probability": probabilities
        })

        st.bar_chart(prob_df.set_index("Emotion"))

    else:
        st.warning("Please enter some text.")































# import streamlit as st
# import pandas as pd
# import string
# import nltk
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression

# # Download stopwords if not already downloaded
# nltk.download('stopwords')

# # -------------------------------
# # Load Dataset
# # -------------------------------
# data = pd.read_csv(
#     "TEXT_DATA/train.txt",
#     sep=";",
#     header=None,
#     names=["text", "emotion"]
# )

# # -------------------------------
# # Label Encoding
# # -------------------------------
# unique_emotions = data["emotion"].unique()
# emotion_numbers = {emo: i for i, emo in enumerate(unique_emotions)}
# data["emotion"] = data["emotion"].map(emotion_numbers)

# # Reverse mapping
# reverse_emotion_map = {v: k for k, v in emotion_numbers.items()}

# # -------------------------------
# # Text Cleaning Function
# # -------------------------------
# stop_words = set(stopwords.words("english"))

# def clean_text(text):
#     text = text.lower()
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     text = ''.join([i for i in text if not i.isdigit()])
#     text = ''.join([i for i in text if i.isascii()])
#     words = text.split()
#     words = [word for word in words if word not in stop_words]
#     return " ".join(words)

# data["text"] = data["text"].apply(clean_text)

# # -------------------------------
# # TF-IDF Vectorization
# # -------------------------------
# tfidf = TfidfVectorizer()
# X = tfidf.fit_transform(data["text"])
# y = data["emotion"]

# # -------------------------------
# # Train Logistic Regression
# # -------------------------------
# model = LogisticRegression(max_iter=1000)
# model.fit(X, y)

# # -------------------------------
# # Streamlit UI
# # -------------------------------
# st.title("Human Emotion Detection using NLP")

# st.write("Enter a sentence and the model will predict the emotion.")

# user_input = st.text_area("Enter Text Here")

# if st.button("Predict Emotion"):
#     if user_input.strip() != "":
#         cleaned_input = clean_text(user_input)
#         vector_input = tfidf.transform([cleaned_input])
#         prediction = model.predict(vector_input)[0]
#         emotion_label = reverse_emotion_map[prediction]

#         st.subheader("Predicted Emotion:")
#         st.success(emotion_label)
#     else:
#         st.warning("Please enter some text.")
