import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

class_names = [None] * len(class_indices)
for class_name, index in class_indices.items():
    class_names[index] = class_name

model = tf.keras.models.load_model("best_model.h5")

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def plot_prediction_bar(predictions):
    confidences = predictions[0]
    top_indices = np.argsort(confidences)[::-1][:3]
    top_classes = [class_names[i] for i in top_indices]
    top_scores = confidences[top_indices]
    fig, ax = plt.subplots()
    bars = ax.barh(top_classes[::-1], top_scores[::-1], color="lightgreen")
    ax.set_xlim([0, 1])
    ax.set_xlabel("Confidence")
    for bar in bars:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width()*100:.1f}%", va='center')
    st.pyplot(fig)

st.set_page_config(page_title=" Tree Species Classifier", layout="centered")
st.title(" Tree Species Classifier")
st.write("Upload an image of a plant or tree leaf to identify its species.")

uploaded_file = st.file_uploader("Drag & drop or browse an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    img = preprocess_image(image)
    with st.spinner("Classifying..."):
        prediction = model.predict(img)
    top_class = np.argmax(prediction)
    confidence = np.max(prediction)
    st.success(f"Prediction: {class_names[top_class]} with {confidence*100:.2f}% confidence")
    st.subheader("Top-3 Prediction Confidence")
    plot_prediction_bar(prediction)
