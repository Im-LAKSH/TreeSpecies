# TreeSpecies

# Tree Species Identification – Week 2 Submission

This repository, for now, contains my Week 2 submission for the Tree Species Identification project. The focus was on initializing the dataset for the tree species classification task and performing preliminary exploration. This focuses on the design and training of convolutional neural network (CNN) models for classifying tree species from images. The notebook includes training of 3 full CNN models.

## Files
- `TreeSpecies.ipynb` – Colab notebook where I did everything from Data Exploration to Model Training.
- `Tree_Species_Dataset/` – Dataset folder (images organized class-wise)
- `tree_species_model.h5` - 1st model.
- `basic_cnn_tree_species.h5` - 2nd model (Can be accessed from the Colab notebook).
- `improved_cnn_model.h5` - 3rd model (Can be accessed from the Colab notebook).

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
  `tree_species_model.h5` , `basic_cnn_tree_species.h5` and `improved_cnn_model.h5`.
- Plotted training/validation accuracy and loss.

## Output Files
- Trained models:
  - `tree_species_model.h5`
  - `basic_cnn_tree_species.h5` - Can't be added due to file size limits. 
  - `improved_cnn_model.h5` - Can't be added due to file size limits. 

## Visualizations
- Training and validation accuracy/loss plots included.
- Used `matplotlib` for visualization.
- Model comparison based on metrics is included.
