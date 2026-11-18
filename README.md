# Tomato Leaf Disease Classification Using Convolutional Neural Networks (CNN)

## Project Overview
This project advances the tomato leaf disease classification pipeline from a traditional Artificial Neural Network (ANN) to a spatial Deep Learning architecture using a custom Convolutional Neural Network (Scratch CNN) built with TensorFlow/Keras[cite: 6]. By preserving spatial hierarchies and leveraging convolutional feature extraction, the model achieves exceptional diagnostic accuracy. An interactive Streamlit web application is deployed for real-time image inference and disease prediction.

## Objective
To classify tomato leaf images into 10 distinct categories (healthy and diseased) using a custom Convolutional Neural Network architecture optimized for high spatial feature extraction, robust generalization, and production efficiency.

## Dataset
- Dataset Source: New Plant Diseases Dataset (Augmented)
- Selected Classes: 10 Tomato leaf categories (~18,000+ training/validation images)
  - Bacterial Spot
  - Early Blight
  - Late Blight
  - Leaf Mold
  - Septoria Leaf Spot
  - Spider Mites (Two-spotted Spider Mite)
  - Target Spot
  - Tomato Yellow Leaf Curl Virus
  - Tomato Mosaic Virus
  - Healthy

## Technologies Used
- Python
- TensorFlow / Keras
- Scikit-Learn (Classification Report & Confusion Matrix)
- NumPy
- Pandas
- Matplotlib
- Pillow (PIL)
- Streamlit

## Deep Learning Workflow
1. Dataset collection and extraction from Google Drive to `/content/tomato_dataset`
2. Automated directory parsing for train and validation splits across 10 target classes
3. Image resizing (224x224x3) and pixel normalization ([0.0, 1.0]) via rescaling layers
4. TensorFlow `tf.data` pipeline optimization with prefetching (`AUTOTUNE`)
5. Custom 3-block Convolutional Neural Network design (Conv2D + MaxPooling blocks)
6. Model training for 10 epochs using the Adam optimizer and sparse categorical crossentropy loss
7. Rigorous evaluation using training/validation history curves, multi-class Scikit-Learn classification reports, and a Confusion Matrix
8. Model serialization (`.keras` format)
9. Web application development and deployment using Streamlit

## Model Details
- Architecture: Custom 3-Block Convolutional Neural Network (Scratch CNN)
- Input Layer: RGB image tensor ($224 \times 224 \times 3 = 150,528$ inputs)
- Convolutional Blocks: 
  - Block 1: `Conv2D` (32 filters, $3 \times 3$, ReLU, padding='same') $\rightarrow$ `MaxPooling2D`
  - Block 2: `Conv2D` (64 filters, $3 \times 3$, ReLU, padding='same') $\rightarrow$ `MaxPooling2D`
  - Block 3: `Conv2D` (128 filters, $3 \times 3$, ReLU, padding='same') $\rightarrow$ `MaxPooling2D`
- Classification Head: `GlobalAveragePooling2D` $\rightarrow$ `Dense(128, ReLU)` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(10, Softmax)`
- Total Parameters: 111,050 (Ultra-lightweight: 433.79 KB)
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy

## Model Performance & Comparison vs. ANN
- **Scratch CNN Validation Accuracy:** **94.90%** (Validation Loss: 0.1522)[cite: 6]
- **Performance Jump:** Drastically outperforms traditional ANNs (~69.64%) by retaining 2D spatial locality, utilizing weight-sharing via sliding convolutional kernels, and eliminating parameter explosion[cite: 6].
- **Per-Class Precision & Recall:** Verified via a multi-class Confusion Matrix and Classification Report showing strong diagonal dominance (e.g., *Tomato Mosaic Virus* F1-score: 0.9854, *Leaf Mold* F1-score: 0.9796, *Healthy* F1-score: 0.9680)[cite: 6].

## Web Application Features
The Streamlit application provides an intuitive interface supporting:
- Tomato leaf image upload (.jpg, .jpeg, .png)
- Real-time uploaded image preview
- Automated image preprocessing and tensor resizing
- Disease class prediction with confidence percentage
- Clean, responsive UI for agricultural diagnostics

## Project Structure
```text
tomato-leaf-disease-classification-cnn/
│
├── app.py                          # Streamlit web application interface
├── requirements.txt                # Project dependencies and libraries
├── README.md                       # Project documentation
└── tomato_cnn_from_scratch.keras   # Trained custom Scratch CNN model (~94.90% Val Accuracy)
