import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from src.model import TrafficSignMobileNet

# Load label names
class_id_to_name = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 tons",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing for vehicles over 3.5 tons",
}

# Basic transform for inference
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# Load model once
@st.cache_resource
def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = TrafficSignMobileNet(num_classes=43).to(device)
    state_dict = torch.load("models/best_signspotter_model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device

model, device = load_model()

# App UI
st.title("SignSpotter — Traffic Sign Classifier")
st.write("Upload any traffic sign image. The model will predict the closest class.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs).item()
        confidence = probs[0][pred_idx].item() * 100

    pred_name = class_id_to_name[pred_idx]

    st.subheader("Prediction")
    st.write(f"**{pred_name}**")
    st.write(f"Confidence: {confidence:.2f}%")
