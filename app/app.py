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
from urllib.parse import urlparse
import matplotlib.pyplot as plt

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

# Parse DATABASE_URL
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    st.error("❌ DATABASE_URL is not set.")
    st.stop()

parsed_url = urlparse(DB_URL)
DB_NAME = parsed_url.path[1:]
DB_USER = parsed_url.username
DB_PASSWORD = parsed_url.password
DB_HOST = parsed_url.hostname
DB_PORT = parsed_url.port

def ensure_predictions_table():
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
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception:
        return False

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

def fetch_total_count():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM predictions")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return count
    except:
        return 0

def fetch_accuracy():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM predictions WHERE predicted = true_label")
        correct = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM predictions")
        total = cur.fetchone()[0]
        cur.close()
        conn.close()
        return round((correct / total * 100), 2) if total > 0 else 0.0
    except:
        return 0.0

# App UI
st.title("🔢 MNIST Digit Recognizer")
st.write("Draw a digit (0–9) below:")

# DB Health Check
if ensure_predictions_table():
    st.sidebar.success("🟢 DB connected")
else:
    st.sidebar.error("🔴 DB connection failed")

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

true_label = st.number_input("Enter the true label (0–9):", min_value=0, max_value=9, step=1)
show_preview = st.checkbox("Show preprocessed input", value=True)

if st.button("Predict & Submit"):
    if canvas_result.image_data is not None:
        img_raw = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'))
        img = transform(img_raw).unsqueeze(0).to(device)

        # Apply binary thresholding for stronger clarity
        img = (img > 0.5).float()

        with torch.no_grad():
            output = model(img)
            prob = F.softmax(output, dim=1)
            pred = prob.argmax(dim=1).item()
            confidence = prob.max().item()

        ts = datetime.datetime.now().isoformat()

        if pred == true_label:
            st.success(f"✅ Prediction: {pred}  |  Confidence: {confidence:.2f}")
        else:
            st.error(f"❌ Incorrect Prediction: {pred} | True: {true_label} | Conf: {confidence:.2f}")

        if show_preview:
            st.subheader("🖼️ Preprocessed Input")
            img_np = img.squeeze().cpu().numpy()
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(img_np, cmap="gray")
            ax.axis("off")
            st.pyplot(fig)

        log_prediction(ts, pred, true_label, confidence)
    else:
        st.warning("Please draw a digit first.")

# View recent predictions
with st.sidebar:
    st.header("📊 Recent Predictions")
    total = fetch_total_count()
    accuracy = fetch_accuracy()
    st.markdown(f"**Total Predictions:** {total}")
    st.markdown(f"**Accuracy:** {accuracy:.2f}%")

    records = fetch_recent_predictions()
    if records:
        for row in records:
            is_correct = "✅" if row[1] == row[2] else "❌"
            confidence_tag = "⚠️" if row[3] < 0.6 else ""
            st.write(f"{is_correct} {confidence_tag} {row[0]} | Pred: {row[1]} | True: {row[2]} | Conf: {row[3]:.2f}")

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
