import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os  # Add this import statement to fix the error
import requests

# Download the model if not present
@st.cache_resource
def download_and_load_model():
    model_path = "model.h5"
    
    if not os.path.exists(model_path):
        url = "https://drive.google.com/file/d/1e90JT3vmdH_usiCykNW-9RbctHzRestA/view?usp=drive_link"  
        r = requests.get(url, allow_redirects=True)
        open(model_path, 'wb').write(r.content)
    
    model = tf.keras.models.load_model(model_path)
    return model

model = download_and_load_model()

# Streamlit app code remains the same
st.title("Gender Classification")
st.write("Upload an image to classify whether it's male or female.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        st.write("Prediction: Female")
    else:
        st.write("Prediction: Male")
