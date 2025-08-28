**SignSpotter: AI-Powered Traffic Sign Recognition**

Traffic signs are crucial for road safety, and automatic recognition of signs plays a vital role in autonomous driving systems. SignSpotter is a deep learning project that classifies German traffic signs using the GTSRB dataset.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**DATASET**

**Source:** German Traffic Sign Recognition Benchmark (GTSRB)

**Classes:** 43 types of traffic signs

_Samples:_

**Training:** ~39,000 images

**Testing:** ~12,600 images

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**PRE-PROCESSING**

Resizing images to 32×32

Normalization

Data augmentation (rotation, flipping, etc.)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**MODEL ARCHITECTURE**

**Framework:** PyTorch

**Model:** Convolutional Neural Network (CNN)

**Layers: **Convolution → ReLU → MaxPooling → Fully Connected → Softmax

**Loss Function:** CrossEntropyLoss

**Optimizer:** Adam

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**RESULTS**

**Test Accuracy:** ~77.55%

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**EVALUATION METRICES:**

Accuracy

Confusion Matrix

Sample predictions grid

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
