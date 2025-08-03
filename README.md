# TreeSpecies

# 🌿 Indian Tree Species Classifier

This project is a deep learning-based image classifier that identifies Indian tree species from leaf or plant images. The model is trained using a Convolutional Neural Network (CNN) and deployed through a Streamlit interface.

---

## 🚀 Features

- Image classification using a custom-trained CNN model (`best_model.h5`)
- Image preprocessing (resize, normalize)
- Label mapping using `class_indices.json` to maintain correct prediction order
- Top-3 predictions with confidence bar graph
- Streamlit-based interactive web app

---

## 📁 Files Included

- `app.py` — Streamlit app script  
- `best_model.h5` — Trained Keras model  
- `class_indices.json` — Class-to-index mapping used during training  

---


## Files
- `TreeSpecies.ipynb` – Colab notebook where I did everything from Data Exploration to Model Training.
- `Tree_Species_Dataset/` – Dataset folder (images organized class-wise)
- `tree_species_model.h5` - 1st model
- `improved_cnn_model.h5` - 3rd model 
- `app.py` — Streamlit app script  
- `best_model.h5` — Trained Keras model  
- `class_indices.json` — Class-to-index mapping used during training

## Colab Notebook
[Click here to open the full Colab notebook to access dataset, all trained models and the code](https://colab.research.google.com/drive/1my9sBm2JNZN3LEpeGDSA70kycbZUNTSU?usp=sharing)


## Tasks Performed

###  1. Dataset Preparation
- Dataset loaded from pre-split directories: `train`, `test`, and `validation`.
- Image preprocessing using `ImageDataGenerator` with:
  - Rescaling pixel values.
  - Target image size: (150, 150).
- Data generators set for training, validation, and testing.
### 2. Model Training
- Preprocessed the dataset with `ImageDataGenerator`.
- Trained **3 models**
- Saved models:  
  `tree_species_model.h5` ,  `improved_cnn_model.h5` and `best_model.h5`.
- Plotted training/validation accuracy and loss.


