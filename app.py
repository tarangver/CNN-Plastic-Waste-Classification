import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model("D:/All-in-one Stuff/Edunet Foundation/Green Skills using AI technologies/CNN-Plastic-Waste-Classification-FINAL/CNN-Plastic-Waste-Classification.h5")

# Title and instructions
st.title('Plastic Waste Classification')
st.write('Upload an image to classify it as Organic Waste or Recyclable Waste')

# File uploader for user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Function to preprocess the image
def preprocess_image(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    img = cv2.resize(img, (224, 224))  # Resize to match model input
    img = np.reshape(img, [-1, 224, 224, 3])  # Add batch dimension
    img = img / 255.0  # Normalize
    return img

# Function to predict the class
def predict_image(img):
    img = preprocess_image(img)
    prediction = model.predict(img)
    if prediction[0][0] > prediction[0][1]:
        return 'Recyclable Waste'
    else:
        return 'Organic Waste'

# If an image is uploaded, display it and make prediction
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    result = predict_image(img)
    st.write(f"Prediction: {result}")

st.markdown(
    """
     <div style="text-align: center; font-family: Arial, sans-serif; font-size: 14px;"> 
        Powered by <b style="font-size: 16px;">TensorFlow</b>, <b style="font-size: 16px;">OpenCV</b>, and <b style="font-size: 16px;">Streamlit</b><br><br>
        <b style="font-size: 16px;">Made by:</b> TARANG VERMA<br>
        <a href="https://www.linkedin.com/in/verma-tarang/" target="_blank" style="color: #333; font-size: 16px;">LinkedIn</a> | 
        <a href="https://github.com/tarangver" target="_blank" style="color: #333; font-size: 16px;">GitHub</a><br><br>
    </div>
    """,
    unsafe_allow_html=True,
)