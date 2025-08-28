**SignSpotter: AI-Powered Traffic Sign Recognition**

Traffic signs are crucial for road safety, and automatic recognition of signs plays a vital role in autonomous driving systems. SignSpotter is a deep learning project that classifies German traffic signs using the GTSRB dataset.

**📂 Dataset**
Source: German Traffic Sign Recognition Benchmark (GTSRB)
Classes: 43 types of traffic signs

_Samples:_
Training: ~39,000 images
Testing: ~12,600 images

_Preprocessing:_
Resizing images to 32×32
Normalization
Data augmentation (rotation, flipping, etc.)

**Model Architecture**
Framework: PyTorch
Model: Convolutional Neural Network (CNN)
Layers: Convolution → ReLU → MaxPooling → Fully Connected → Softmax
Loss Function: CrossEntropyLoss
Optimizer: Adam

**📊 Results**
Test Accuracy: ~77.55%

**Evaluation Metrics:**
Accuracy
Confusion Matrix
Sample predictions grid
