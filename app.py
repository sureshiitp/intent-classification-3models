
import streamlit as st
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

st.title("Intent Classification (TF-IDF, BiGRU, TinyBERT)")

#########################################
# TF-IDF LOAD
#########################################
def load_tfidf():
    clf = joblib.load("models/tfidf/tfidf_model.joblib")
    vec = joblib.load("models/tfidf/tfidf_vectorizer.joblib")
    le = joblib.load("models/tfidf/label_encoder.joblib")
    return clf, vec, le

#########################################
# BiGRU LOAD
#########################################
def load_bigru():
    model = tf.keras.models.load_model("models/bilstm/bilstm_model.h5")
    tok = joblib.load("models/bilstm/tokenizer_bilstm.joblib")
    le = joblib.load("models/bilstm/label_encoder.joblib")
    return model, tok, le

#########################################
# TinyBERT LOAD
#########################################
def load_tinybert():
    tok = AutoTokenizer.from_pretrained("models/tinybert")
    model = AutoModelForSequenceClassification.from_pretrained("models/tinybert")
    le = joblib.load("models/tinybert/label_encoder.joblib")
    return tok, model, le

model_name = st.selectbox("Choose Model", ["TF-IDF", "BiGRU", "TinyBERT"])
user_input = st.text_input("Enter your message:")

if st.button("Predict") and user_input:
    
    #########################################
    # TF-IDF PREDICTION
    #########################################
    if model_name == "TF-IDF":
        clf, vec, le = load_tfidf()
        x = vec.transform([user_input])
        pred = clf.predict(x)[0]
        st.success(le.inverse_transform([pred])[0])

    #########################################
    # BiGRU PREDICTION
    #########################################
    elif model_name == "BiGRU":
        model, tok, le = load_bigru()
        seq = tok.texts_to_sequences([user_input])
        seq = pad_sequences(seq, maxlen=64)
        probs = model.predict(seq)[0]
        pred = np.argmax(probs)
        st.success(le.inverse_transform([pred])[0])

    #########################################
    # TinyBERT Prediction
    #########################################
    else:
        tok, model, le = load_tinybert()
        tokens = tok(user_input, return_tensors="pt")
        out = model(**tokens)
        pred = int(torch.argmax(out.logits))
        st.success(le.inverse_transform([pred])[0])
