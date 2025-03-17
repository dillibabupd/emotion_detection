import os
import tensorflow as tf
from keras.models import load_model
import numpy as np
import streamlit as st

st.header('Emotion Detection CNN Model')

data_cat = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'] # label names 
# Load the trained model
model = load_model(r"D:\guvi\code\emotion_image_model.h5")

def emotion_model(image_path):
    # Load and preprocess the image
    image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    image_array = tf.keras.utils.img_to_array(image)  # Convert to array
    image_batch = tf.expand_dims(image_array, axis=0)  # Add batch dimension
    predict = model.predict(image_batch)  # Get predictions
    result = tf.nn.softmax(predict)  # Assuming a single image
    # Determine the class and confidence
    outcome = f"The image belongs to {data_cat[np.argmax(result)]} with a accuracy of {np.max(result) * 100:.2f}%"
    return outcome

uploaded_file = st.file_uploader('Upload an image')
if uploaded_file is not None:
    # Save the uploaded file to a temporary directory
    upload_path = os.path.join('upload', uploaded_file.name)
    os.makedirs('upload', exist_ok=True)  # Create directory if it doesn't exist
    with open(upload_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(upload_path, width=100) # Display the uploaded image
    st.write(emotion_model(upload_path)) # Get and display the prediction
