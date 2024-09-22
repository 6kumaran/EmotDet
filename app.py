import streamlit as st
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load pre-trained models and data
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotion_model = load_model('model.h5')
    Music_Player = pd.read_csv("data_moods.csv")[['name', 'artist', 'mood', 'popularity']]
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_emotions(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    predicted_emotion = ""
    
    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))  # Resize to the size expected by the model
        roi_gray = roi_gray.astype('float32') / 255  # Normalize the image
        
        roi_gray = roi_gray.reshape(1, 48, 48, 1)
        # Predict the emotion
        predictions = emotion_model.predict(roi_gray)
        max_index = predictions[0].argmax()
        predicted_emotion = emotion_labels[max_index]
        
        # Draw rectangle around the face and put text
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return frame, predicted_emotion

def recommend_songs(pred_class):
    Play = pd.DataFrame()  # Initialize Play as an empty DataFrame
    
    if pred_class == 'Disgust':
        Play = Music_Player[Music_Player['mood'] == 'Sad']
    elif pred_class in ['Happy', 'Sad']:
        Play = Music_Player[Music_Player['mood'] == 'Happy']
    elif pred_class in ['Fear', 'Angry']:
        Play = Music_Player[Music_Player['mood'] == 'Calm']
    elif pred_class in ['Surprise', 'Neutral']:
        Play = Music_Player[Music_Player['mood'] == 'Energetic']
    
    # If no songs were found, return an empty DataFrame
    if Play.empty:
        return pd.DataFrame(columns=['name', 'artist', 'mood', 'popularity'])  # Return empty DataFrame with same columns
    
    Play = Play.sort_values(by="popularity", ascending=False).head(5).reset_index(drop=True)
    return Play

def main():
    st.title("Real-Time Emotion Detection and Music Recommendation")

    # Initialize session state for storing last detected emotion and recommendations
    if 'last_emotion' not in st.session_state:
        st.session_state.last_emotion = ""
    
    if 'last_recommendations' not in st.session_state:
        st.session_state.last_recommendations = pd.DataFrame(columns=['name', 'artist', 'mood', 'popularity'])

    # Start video capture
    run = st.checkbox('Run Emotion Detection')
    
    # Placeholder for video feed and recommendations
    video_placeholder = st.empty()
    
    if run:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Error opening video stream or file")
            return
        
        while run:  # Continue capturing frames while run is True
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break
            
            frame, emot = detect_emotions(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Streamlit
            
            # Display video frame
            video_placeholder.image(frame_rgb)

            # Update recommendations only if a new emotion is detected
            if emot != st.session_state.last_emotion:
                st.session_state.last_emotion = emot
                st.session_state.last_recommendations = recommend_songs(emot)  # Store last recommendations

            # Display last recommendations
            if not st.session_state.last_recommendations.empty:
                recommendations_placeholder = st.empty()
                recommendations_placeholder.subheader("Recommended Songs Based on Last Detected Emotion:")
                recommendations_placeholder.dataframe(st.session_state.last_recommendations)

            # Allow Streamlit to rerun and check if run is still True
            if not run:
                break

        cap.release()

    # Display last recommendations even after unchecking the checkbox
    if not st.session_state.last_recommendations.empty:
        st.subheader("Recommended Songs Based on Last Detected Emotion:")
        st.dataframe(st.session_state.last_recommendations)

if __name__ == "__main__":
    main()