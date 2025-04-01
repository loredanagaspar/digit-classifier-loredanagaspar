# app.py

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import datetime
import psycopg2
import os
import logging
import pandas as pd

# CNN model class (must match training)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

# Preprocessing transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# PostgreSQL connection
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT", 5432)

def log_prediction(ts, pred, true_label, confidence):
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP,
                predicted INTEGER,
                true_label INTEGER,
                confidence FLOAT
            )
        """)
        cur.execute(
            "INSERT INTO predictions (timestamp, predicted, true_label, confidence) VALUES (%s, %s, %s, %s)",
            (ts, pred, true_label, confidence)
        )
        conn.commit()
        cur.close()
        conn.close()
        st.success("✅ Prediction logged successfully.")
    except Exception as e:
        st.error(f"Database error: {e}")

def fetch_recent_predictions(limit=10):
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute("SELECT timestamp, predicted, true_label, confidence FROM predictions ORDER BY id DESC LIMIT %s", (limit,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        st.error(f"❌ Could not fetch predictions: {e}")
        return []
    
def fetch_all_predictions():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute("SELECT timestamp, predicted, true_label, confidence FROM predictions ORDER BY id DESC")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return pd.DataFrame(rows, columns=["timestamp", "predicted", "true_label", "confidence"])
    except Exception as e:
        st.error(f"❌ Could not fetch predictions: {e}")
        return pd.DataFrame()
    
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
         logging.basicConfig(level=logging.INFO)
         logging.info(f"Submitting: {pred}, {true_label}, {confidence}")
         log_prediction(ts, pred, true_label, confidence)
         st.success("✅ Prediction logged to database.")
         st.write(f"{ts} | Prediction: {pred} | True Label: {true_label} | Confidence: {confidence:.2f}")

# View recent predictions
with st.sidebar:
    st.header("📊 Recent Predictions")
    records = fetch_recent_predictions()
    if records:
        for row in records:
            st.write(f"🕓 {row[0]} | Pred: {row[1]} | True: {row[2]} | Conf: {row[3]:.2f}")

    df = fetch_all_predictions()
    if not df.empty:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download All Predictions (CSV)",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
        )
    else:
        st.info("No predictions logged yet.")

