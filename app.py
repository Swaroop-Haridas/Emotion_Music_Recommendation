import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
import time

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = cv2.face.LBPHFaceRecognizer_create()
model.read("model.yml")
label_dict = np.load("labels.npy", allow_pickle=True).item()
reverse_label_dict = {v: k for k, v in label_dict.items()}

def get_songs_for_emotion(emotion):
    path = os.path.join("songs", f"{emotion.lower()}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if df.columns[0].lower().startswith("unnamed"):
            df = df.iloc[:, 1:]
        df.columns = ["Track", "Artists", "Album", "SpotifyURL"]
        return df
    return pd.DataFrame()

st.set_page_config(page_title="Emotion Music Recommender", layout="centered")
st.title("🎭 Emotion-Based Music Recommender")

if "detected" not in st.session_state:
    st.session_state.detected = False

frame_placeholder = st.empty()
status_placeholder = st.empty()

if st.button("Start Camera") and not st.session_state.detected:
    cap = cv2.VideoCapture(0)
    detected_emotion = None
    last_detection_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Could not access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi_gray, (200, 200))
            label_id, confidence = model.predict(roi_resized)
            detected_emotion = label_dict[label_id]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, detected_emotion.upper(), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            if time.time() - last_detection_time > 2:
                cap.release()
                st.session_state.detected = True
                st.session_state.emotion = detected_emotion
                break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        if st.session_state.detected:
            break

    cap.release()

if st.session_state.get("emotion"):
    status_placeholder.success(f"🎯 Emotion Detected: **{st.session_state['emotion'].upper()}**")
    songs_df = get_songs_for_emotion(st.session_state['emotion'])

    if not songs_df.empty:
        st.subheader("🎵 Recommended Songs")
        st.dataframe(songs_df)
    else:
        st.warning("No songs found for this emotion.")
