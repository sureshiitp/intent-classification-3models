import streamlit as st
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.title("Intent Classification (TF-IDF, BiGRU, TinyBERT)")

# ---------------- TF-IDF ----------------
@st.cache_resource
def load_tfidf():
    clf = joblib.load("tfidf/tfidf_model.joblib")
    vec = joblib.load("tfidf/tfidf_vectorizer.joblib")
    le  = joblib.load("tfidf/label_encoder.joblib")
    return clf, vec, le

# ---------------- BiGRU ----------------
@st.cache_resource
def load_bigru():
    model = tf.keras.models.load_model("bilstm/bilstm_model.h5")
    tok   = joblib.load("bilstm/tokenizer_bilstm.joblib")
    le    = joblib.load("bilstm/label_encoder.joblib")
    return model, tok, le

# ---------------- TinyBERT ----------------
@st.cache_resource
def load_tinybert():
    tok   = AutoTokenizer.from_pretrained("tinybert")
    model = AutoModelForSequenceClassification.from_pretrained("tinybert")
    le    = joblib.load("tinybert/label_encoder.joblib")
    return tok, model, le


model_choice = st.selectbox("Choose Model", ["TF-IDF", "BiGRU", "TinyBERT"])
user_text = st.text_input("Enter your message:")


# ---------------- Predict ----------------
if st.button("Predict") and user_text.strip() != "":

    # ----- TF-IDF -----
    if model_choice == "TF-IDF":
        clf, vec, le = load_tfidf()
        X = vec.transform([user_text])
        pred = clf.predict(X)[0]
        label = le.inverse_transform([pred])[0]
        st.success(f"Prediction: {label}")

    # ----- BiGRU -----
    elif model_choice == "BiGRU":
        model, tok, le = load_bigru()
        seq = tok.texts_to_sequences([user_text])
        seq = pad_sequences(seq, maxlen=64)
        probs = model.predict(seq)[0]
        pred = np.argmax(probs)
        label = le.inverse_transform([pred])[0]
        st.success(f"Prediction: {label}")

    # ----- TinyBERT -----
    else:
        tok, model, le = load_tinybert()
        inputs = tok(user_text, return_tensors="pt")
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits).item()
        label = le.inverse_transform([pred])[0]
        st.success(f"Prediction: {label}")


