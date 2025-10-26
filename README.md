# Bone Fracture Detection Using Machine Learning

## Overview
This project implements a bone fracture classification system using X-ray images. The system extracts features from images and applies multiple machine learning models to classify fractures into 10 categories:

- Comminuted
- Greenstick
- Healthy
- Linear
- Oblique Displaced
- Oblique
- Segmental
- Spiral
- Transverse Displaced
- Transverse

The models implemented include:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Naive Bayes
- Bayesian Network
- Decision Tree
- Random Forest
- XGBoost
- AdaBoost
- Multi-Layer Perceptron (MLP)


## Features Extraction
- HOG (Histogram of Oriented Gradients) features are extracted from images.
- Features are scaled using `StandardScaler` before training.

---

## Bayesian Network
- Features are discretized using `KBinsDiscretizer` before fitting a Bayesian Network.
- Hill Climb Search is used to learn the structure.
- Predictions are made using Variable Elimination.

---

## Model Training
Each model is trained on the training set and evaluated on the validation set. Accuracy metrics collected:

- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- Confusion Matrix (graphical visualization using Seaborn)

