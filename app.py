import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Load pre-trained models and data
try:
    emotion_model = load_model('model.h5')
    Music_Player = pd.read_csv("data_moods.csv")[['name', 'artist', 'mood', 'popularity']]
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_emotion = ""
        self.last_recommendations = pd.DataFrame(columns=['name', 'artist', 'mood', 'popularity'])

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Detect emotions in the image
        processed_frame, predicted_emotion = detect_emotions(img_rgb)

        # Update recommendations if a new emotion is detected
        if predicted_emotion != self.last_emotion:
            self.last_emotion = predicted_emotion
            self.last_recommendations = recommend_songs(predicted_emotion)

        return processed_frame  # Return processed frame for display

def detect_emotions(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
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
        
        # Draw rectangle around the face and put text (optional)
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

    # Start video capture using streamlit-webrtc
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    # Display last recommendations even after unchecking the checkbox
    if st.session_state.get('last_recommendations') is not None and not st.session_state.last_recommendations.empty:
        st.subheader("Recommended Songs Based on Last Detected Emotion:")
        st.dataframe(st.session_state.last_recommendations)

if __name__ == "__main__":
    main()
