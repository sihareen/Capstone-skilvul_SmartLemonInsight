import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Load your trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'D:/PROJECT/Capstone-skilvul_SmartLemonInsight/modelLemon.hdf5'
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (300, 300))  # Resize to the input size of your model
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make predictions
def predict(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    if predictions[0][0] > 0.5:
        predicted_class = "Lemon with bad quality"
    elif predictions[0][1] > 0.5:
        predicted_class = "Lemon not detected"
    else:
        predicted_class = "Lemon with good quality"
    return predicted_class

# Streamlit app
st.title("Real-Time Lemon Quality Detection with Streamlit")
st.write("Using a pre-trained CNN model")

# Start video capture
video_capture = cv2.VideoCapture(0)

if st.button('Start Detection'):
    stframe = st.empty()
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Make predictions
        predicted_class = predict(frame)
        
        # Display the prediction on the frame
        cv2.putText(frame, f'Prediction: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame with predictions
        stframe.image(frame, channels="BGR")

# Release the video capture
video_capture.release()