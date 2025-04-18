import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = load_model(r'C:\Users\baded\Audio Project\cnn_audio_model.h5')

# Define the class names and emojis
emotion_emojis = {
    'sadness': '😢',
    'fear': '😨',
    'disgust': '🤢',
    'joy': '😊',
    'surprise': '😲',
    'neutral': '😐',
    'anger': '😡'
}

# Load the label encoder classes
try:
    label_encoder_classes = np.load(r'C:\Users\baded\Audio Project\label_encoder_classes.npy', allow_pickle=True)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = label_encoder_classes
except FileNotFoundError:
    st.error("Label encoder file not found. Make sure 'label_encoder_classes.npy' is present.")
    st.stop()

# Extract class names
class_names = label_encoder.classes_

# Function to extract audio features using Librosa
def extract_audio_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_processed = np.mean(mfccs.T, axis=0)  # Taking the mean across time
        return mfccs_processed
    except Exception as e:
        st.error(f"Error extracting features from {audio_path}: {str(e)}")
        return None

# Streamlit app
st.title("Emotion Prediction from Audio")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Extract audio features
    features = extract_audio_features(uploaded_file)
    
    if features is not None:
        # Prepare the features for prediction
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        features = np.expand_dims(features, axis=-1)  # Add channel dimension if required by your model
        
        # Predict emotion
        predictions = model.predict(features)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_emotion = label_encoder.inverse_transform([predicted_class_index])[0]
        emoji = emotion_emojis.get(predicted_emotion, "❓")
        
        st.success(f"**Predicted Emotion:** {predicted_emotion} {emoji}")
    else:
        st.warning("Could not extract features from the audio file.")
