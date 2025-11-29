#  Human Bone Fracture Classification

This repository contains an end-to-end machine learning workflow for **bone fracture classification** using **X-ray and MRI images** from the *Human Bone Fractures Multi-modal Image Dataset (HBFMID)*.
The project includes image loading, preprocessing, HOG feature extraction, model training (Logistic Regression, KNN, SVM), evaluation, and visualization.

---

##  Dataset

**Human Bone Fractures Multi-modal Image Dataset (HBFMID)**

* Includes **X-ray & MRI** scans of multiple bones
* Covers classes such as:
  *Comminuted, Greenstick, Healthy, Linear, Oblique, Segmental, Spiral, Transverse,* etc.
* **641 raw images** (510 X-ray + 131 MRI)
* Published: **2 December 2024**
* Author: **Shahnaj Parvin**

**Citation:**

```
Parvin, Shahnaj (2024), “Human Bone Fractures Multi-modal Image Dataset (HBFMID)”, 
Mendeley Data, V1, doi: 10.17632/xwfs6xbk47.1
```

Folder structure expected:

```
Bone_Fractures_Detection/
    train/
        images/
        labels/
    valid/
        images/
        labels/
    test/
        images/
        labels/
```

---

##  Libraries Used

* OpenCV
* NumPy, Pandas
* scikit-learn
* seaborn, matplotlib
* pgmpy (for Bayesian Networks)
* XGBoost
* scikit-image (HOG features)

---

## Preprocessing & Feature Extraction

### ✔ Image Loading

Images are read in grayscale and paired with label files.

### ✔ Image Size

All images resized to **128 × 128**.

### ✔ HOG Features

Histogram of Oriented Gradients is used to extract structural patterns from images.

Parameters used:

* `pixels_per_cell = (16, 16)`
* `cells_per_block = (2, 2)`
* `feature_vector = True`

### Feature Scaling

`StandardScaler` is applied to all extracted features.

---

##  Models Implemented

| Model                             | Status        | Notes                              |
| --------------------------------- | ------------- | ---------------------------------- |
| **Logistic Regression**           | ✔             | Baseline classifier                |
| **KNN (with PCA)**                | ✔             | GridSearchCV used                  |
| **SVM (with PCA)**                | ✔             | Improves high-dimensional learning |
| **Naive Bayes**                   | (if included) | Works with HOG features            |
| **Decision Tree / Random Forest** | (optional)    | Good for baseline comparison       |
| **XGBoost**                       | (optional)    | Handles non-linear patterns        |
| **MLPClassifier**                 | (optional)    | Feedforward neural network         |


---

##  Key Techniques Explained

###  HOG Feature Extraction

Captures gradient orientations — ideal for fracture edges and bone outlines.

### PCA for Dimensionality Reduction

Used before KNN and SVM to:

* Speed up computation
* Remove noisy features
* Reduce curse of dimensionality

### GridSearchCV

Used for hyperparameter tuning across:

* K values
* Distance metrics
* PCA variance levels

---

## Visualizations Included

* Confusion matrices
* Classification reports
* Accuracy/precision/recall/f1 comparison
* Dataset summaries

---

## Future Improvements

* CNN-based deep learning models
* Transfer learning with ResNet/DenseNet
* MRI/X-ray multimodal fusion
* Explainability (Grad-CAM, SHAP)
* Bayesian Network classification

---

## Citation

If you use the dataset, cite:

```
Parvin, Shahnaj (2024), “Human Bone Fractures Multi-modal Image Dataset (HBFMID)”, 
Mendeley Data, V1, doi: 10.17632/xwfs6xbk47.1

