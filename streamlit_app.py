import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Define the Streamlit app
st.title("Gender Classification")

st.write("Upload an image to classify whether it's male or female.")

# Allow user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))  # Resize the image to match the model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image (same as during training)

    # Make a prediction
    prediction = model.predict(img_array)

    # Output the prediction
    if prediction[0] > 0.5:
        st.write("Prediction: Female")
    else:
        st.write("Prediction: Male")
