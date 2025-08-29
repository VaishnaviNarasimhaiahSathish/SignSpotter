**SignSpotter: Traffic Sign Classification with CNN**
Overview:

SignSpotter is an end-to-end deep learning project for traffic sign recognition using Convolutional Neural Networks (CNNs). It classifies traffic signs from the GTSRB dataset with high accuracy and provides tools for visualization and inference.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**FEATURES**

1. Custom PyTorch Dataset class supporting ROI cropping.

2. Preprocessing with image resizing and normalization.

3. CNN model with:

   1. 2 convolutional layers

   2. MaxPooling and dropout layers

   3. Fully connected layers for classification

4. Training with validation tracking and best model saving.

5. Model evaluation using:

   1. Test accuracy

   2. Confusion matrix

   3. Classification report

6. Inference and visualization:

   1. Single image prediction

   2. Random sample predictions

   3. Prediction grids for multiple images

7. Model saving and reloading for deployment-ready inference.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**DATASET**

**Source:** German Traffic Sign Recognition Benchmark (GTSRB)

**Classes:** 43 types of traffic signs

_Samples:_

**Training:** ~39,000 images

**Testing:** ~12,600 images

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**PRE-PROCESSING**

1. Resizing images to 32×32
2. Normalization
3. Data augmentation (rotation, flipping, etc.)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**MODEL ARCHITECTURE**

**1. Framework:** PyTorch

**2. Model:** Convolutional Neural Network (CNN)

**3. Layers:**Convolution → ReLU → MaxPooling → Fully Connected → Softmax

**4. Loss Function:** CrossEntropyLoss

**5. Optimizer:** Adam

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**RESULTS**

**Test Accuracy:** ~77.55%

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**EVALUATION METRICES:**

1. Accuracy
2. Confusion Matrix
3. Sample predictions grid

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**CONFUSION MATRIX**

(Example visualization — from sklearn’s ConfusionMatrixDisplay)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**SAMPLE PREDICTIONS**

Green = correct ✅, Red = incorrect ❌

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## ⚙️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/<VaishnaviNarasimhaiahSathish>/SignSpotter.git
   cd SignSpotter

2. Install dependencies:
   ```bash
   pip install -r requirements.txt


4. Download the GTSRB dataset from Kaggle
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

5. Run the notebook:
 ```bash
 jupyter notebook notebooks/signspotter.ipynb
