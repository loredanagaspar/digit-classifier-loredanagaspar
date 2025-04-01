# app.py

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import datetime

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("model.pt", map_location=device)
model.eval()

# Preprocessing transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# App UI
st.title("🧠 MNIST Digit Recognizer")
st.write("Draw a digit (0-9) below:")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict button
if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'))
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            prob = F.softmax(output, dim=1)
            pred = prob.argmax(dim=1).item()
            confidence = prob.max().item()

        st.success(f"Prediction: {pred}  |  Confidence: {confidence:.2f}")

        true_label = st.number_input("Enter the true label (0-9):", min_value=0, max_value=9, step=1)

        if st.button("Submit Label"):
            ts = datetime.datetime.now().isoformat()
            st.write(f"✅ Logged: {ts}, Predicted={pred}, True={true_label}, Confidence={confidence:.2f}")
            # 🔧 Add database logging here
    else:
        st.warning("Please draw a digit first.")