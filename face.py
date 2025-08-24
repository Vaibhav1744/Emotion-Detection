import cv2
import numpy as np
import streamlit as st
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Emotion Dictionary
emotion_dict = {
    0: "Angry", 
    1: "Disgusted", 
    2: "Fearful", 
    3: "Happy", 
    4: "Neutral", 
    5: "Sad", 
    6: "Surprised"
}

# Model Definition
def create_emotion_model():
    """Create and load the CNN model for emotion detection."""
    try:
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')
        ])
        model.load_weights('model.h5')  # Pre-trained weights file
        logger.info("✅ Model weights loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"❌ Error loading model.h5: {e}")
        return None

def main():
    st.set_page_config(page_title="Emotion Detection", page_icon="😊", layout="wide")
    st.title("🎥 Real-Time Emotion Detection")
    st.write("This app detects your **face** and predicts your **emotion** using a CNN model.")

    run_button = st.button("Start Emotion Detection")

    if run_button:
        model = create_emotion_model()
        if model is None:
            st.error("Model could not be loaded. Please ensure `model.h5` is in this folder.")
            return

        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        if face_cascade.empty():
            st.error("Failed to load Haar Cascade XML file.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("No camera found. Please connect a webcam.")
            return

        frame_window = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Camera feed interrupted.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                try:
                    cropped_img = cv2.resize(roi_gray, (48, 48))
                    cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)
                    cropped_img = cropped_img.astype('float32') / 255.0
                    prediction = model.predict(cropped_img, verbose=0)
                    max_index = int(np.argmax(prediction))
                    emotion = emotion_dict[max_index]
                    cv2.putText(frame, emotion, (x+20, y-60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2, cv2.LINE_AA)
                except Exception as e:
                    logger.error(f"Prediction error: {e}")
                    continue

            # Convert frame to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame_rgb, channels="RGB")

    else:
        st.info("Click **Start Emotion Detection** to begin.")

if __name__ == "__main__":
    main()
