import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import exifread
import os
import tempfile
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray

# Load the trained model
MODEL_PATH = 'vgg16_casia_model2.h5'
model = load_model(MODEL_PATH)

# Image preprocessing function
def preprocess_image(img):
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    if img_array.shape[-1] != 3:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# # Metadata extraction function
# def extract_metadata(image_file):
#     file_stats = os.fstat(image_file.fileno())
#     metadata = {
#         'File Size': file_stats.st_size,
#         'Creation Time': datetime.fromtimestamp(file_stats.st_ctime),
#         'Last Modified Time': datetime.fromtimestamp(file_stats.st_mtime),
#         'File Format': os.path.splitext(image_file.name)[1]
#     }
#     return metadata

# Define the extract_file_metadata function to get file metadata
def extract_metadata(image_path):
    file_stats = os.stat(image_path)
    metadata = {
        'File Size': file_stats.st_size,
        'Creation Time': datetime.fromtimestamp(file_stats.st_ctime),
        'Last Modified Time': datetime.fromtimestamp(file_stats.st_mtime),
        'File Format': os.path.splitext(image_path)[1]
    }
    return metadata

# Feature extraction functions
def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_lbp(image):
    gray_image = rgb2gray(image)
    lbp = local_binary_pattern(gray_image, P=8, R=1, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_edges(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

def extract_features(image):
    color_hist = extract_color_histogram(image)
    lbp = extract_lbp(image)
    features = np.hstack([color_hist, lbp])
    return features

# Streamlit app layout
st.title('🕵️‍♂️ Fake Image Detection System')
st.write("""
Upload an image to determine if it's **real** or **fake**. 
The system uses a trained CNN model and analyzes image metadata for enhanced accuracy.
""")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image for prediction
    img_preprocessed = preprocess_image(image_pil)
    
    # Make prediction
    prediction = model.predict(img_preprocessed)[0][0]
    prediction_label = "Fake Image" if prediction > 0.5 else "Real Image"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    st.write(f"**Prediction:** {prediction_label} ({confidence * 100:.2f}% confidence)")

    # Extract and display metadata
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp:
        temp.write(uploaded_file.read())
        temp.flush()  # Ensure all data is written to disk
        os.fsync(temp.fileno())  # Synchronize file's state with the disk
        temp_path = temp.name
    
    # Extract metadata using the file path
    metadata = extract_metadata(temp_path)
    
    st.subheader("📄 Metadata Information")
    st.json(metadata)
    
    # Convert PIL image to OpenCV format
    img_cv2 = np.array(image_pil)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
    img_cv2 = cv2.resize(img_cv2, (224, 224))

    # Extract features
    features = extract_features(img_cv2)
    st.subheader("🧩 Extracted Features")
    st.write(f"Color Histogram + LBP Feature Vector Length: {len(features)}")
    
    # Display edges
    edges = extract_edges(img_cv2)
    st.subheader("🖼️ Edge Detection")
    st.image(edges, caption='Detected Edges', use_column_width=True, channels='GRAY')
    
    # Clean up temporary file
    os.remove(temp_path)
