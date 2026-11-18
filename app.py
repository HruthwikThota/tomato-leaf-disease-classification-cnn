import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Tomato Leaf Disease Classifier",
    page_icon="🍅",
    layout="centered"
)

# Load the trained Scratch CNN Model
@st.cache_resource
def load_model_file():
    model = load_model("tomato_cnn_from_scratch.keras")
    return model

model = load_model_file()

# Define the 10 Class Names
class_names = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# App UI Header
st.title("🍅 Tomato Leaf Disease Classification (CNN)")
st.write("Upload an image of a tomato leaf to detect potential diseases using your trained Convolutional Neural Network.")

# File Uploader
uploaded_file = st.file_uploader("Choose a tomato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Tomato Leaf", use_column_width=True)
    
    # Preprocess image
    image = image.resize((224, 224))
    img_array = np.array(image)
    
    # Handle RGBA or grayscale images
    if img_array.ndim == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
        
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalization matching training pipeline

    # Predict button
    if st.button("Predict Disease"):
        with st.spinner("Analyzing leaf..."):
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            
            predicted_class_name = class_names[predicted_class_idx]

        st.success(f"**Prediction:** {predicted_class_name}")
        st.info(f"**Confidence:** {confidence:.2f}%")