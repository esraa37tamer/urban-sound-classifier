import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import joblib
import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE

TARGET_SR = 16000

# ------------------- Load models -------------------
@st.cache_resource
def load_yamnet():
    return tf.saved_model.load("yamnet")

@st.cache_resource
def load_custom_model():
    clf = joblib.load("classifier.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return clf, scaler, label_encoder

# ------------------- Preprocessing -------------------
def preprocess_audio(audio, sr, target_sr=TARGET_SR, target_duration=4, noise_reduction=False):
    # Resample
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=20)

    # Normalize
    if np.abs(audio).max() > 0:
        audio = audio / np.abs(audio).max()

    # Optional simple noise reduction
    if noise_reduction:
        audio = audio * (np.abs(audio) > 0.02)

    # Pad or truncate to target duration
    target_len = int(target_sr * target_duration)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
    else:
        audio = audio[:target_len]

    return audio.astype(np.float32)

# ------------------- Features -------------------
def extract_yamnet_embedding(yamnet_model, audio):
    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet_model(waveform)
    emb_mean = tf.reduce_mean(embeddings, axis=0, keepdims=True)
    return emb_mean.numpy()

def extract_mfcc(audio, sr, n_mfcc=40):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0).reshape(1, -1)

# ------------------- Plots -------------------
def plot_waveform(audio, sr):
    fig, ax = plt.subplots()
    times = np.arange(len(audio)) / float(sr)
    ax.plot(times, audio)
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    return fig

def plot_spectrogram(audio, sr):
    fig, ax = plt.subplots()
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title("Mel-Spectrogram")
    return fig

def plot_probabilities(prob_df):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x="Probability", y="Class", data=prob_df, palette="viridis", ax=ax)
    ax.set_xlabel("Probability")
    ax.set_ylabel("Class")
    ax.set_title("Class Probability Distribution")
    return fig

def plot_pie(prob_df):
    fig, ax = plt.subplots()
    ax.pie(prob_df['Probability'], labels=prob_df['Class'], autopct='%1.1f%%', startangle=140)
    ax.set_title("Prediction Confidence Pie")
    return fig

# ------------------- Streamlit UI -------------------
st.title("🎵 Enhanced Urban Sound Classification")

# Sidebar options
st.sidebar.header("Preprocessing Options")
noise_reduction = st.sidebar.checkbox("Apply Simple Noise Reduction", False)
target_duration = st.sidebar.slider("Audio Duration (seconds)", min_value=2, max_value=10, value=4)

uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

if uploaded_file:
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Load audio and preprocess
        audio_raw, sr = librosa.load(tmp_file_path, sr=None)
        audio_processed = preprocess_audio(audio_raw, sr, target_duration=target_duration,
                                           noise_reduction=noise_reduction)

        # Audio playback and visualizations
        st.audio(tmp_file_path, format="audio/wav")
        st.pyplot(plot_waveform(audio_processed, TARGET_SR))
        st.pyplot(plot_spectrogram(audio_processed, TARGET_SR))

        # Load models
        yamnet_model = load_yamnet()
        clf, scaler, label_encoder = load_custom_model()

        # Prediction (MFCC only)
        mfcc_feat = extract_mfcc(audio_processed, TARGET_SR)
        mfcc_scaled = scaler.transform(mfcc_feat)
        prediction = clf.predict(mfcc_scaled)
        pred_label = label_encoder.inverse_transform(prediction)[0]
        probas = clf.predict_proba(mfcc_scaled)[0]
        pred_confidence = np.max(probas)

        st.subheader(f"🎯 Predicted Class: {pred_label} ({pred_confidence*100:.2f}% confidence)")

        # Probability DataFrame aligned with label_encoder
        prob_df = pd.DataFrame({
            "Class": label_encoder.inverse_transform(np.arange(len(probas))),
            "Probability": probas
        }).sort_values("Probability", ascending=False).reset_index(drop=True)

        st.subheader("📊 Class Probabilities")
        st.dataframe(prob_df.style.format({"Probability": "{:.3f}"}))
        st.pyplot(plot_probabilities(prob_df))
        st.pyplot(plot_pie(prob_df))

        os.remove(tmp_file_path)

    except Exception as e:
        st.error(f"Error processing audio file: {e}")

else:
    st.info("Please upload a WAV file to classify.")