 # app/streamlit_app.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import torch
import torch.nn.functional as F
from datetime import datetime
import psycopg2
import os
from dotenv import load_dotenv


load_dotenv()

# Force Streamlit to use port 8080 for Railway
os.environ["PORT"] = "8080"

# --- Load Model ---
from model.model import MNISTClassifier
model = MNISTClassifier()
model.load_state_dict(torch.load("model/model.pth", map_location=torch.device('cpu')))
model.eval()


# === DB Connection Helper ===
def get_db_connection():
    try:
        url = os.getenv("DATABASE_URL")
        conn = psycopg2.connect(url, sslmode='require')
        return conn
    except Exception as e:
        st.error(f"❌ DB connection failed: {e}")
        return None


# --- UI ---
st.set_page_config(page_title="Digit Recognizer")
st.title("\U0001F522 Digit Recognizer")

st.markdown("### Draw a digit below:")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas",
)

pred_digit = None
confidence = None

if canvas_result.image_data is not None:
    img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype(np.uint8))
    img = img.resize((28, 28)).convert('L')
    tensor = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1)
        pred_digit = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()

if pred_digit is not None:
    st.markdown(f"**Prediction:** {pred_digit}")
    st.markdown(f"**Confidence:** {confidence*100:.0f}%")

    true_label = st.number_input("True label (optional)", min_value=0, max_value=9, step=1)

    if st.button("Submit"):
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO predictions (timestamp, predicted, label)
                VALUES (%s, %s, %s);
            """, (datetime.now(), pred_digit, true_label))
            conn.commit()
            cur.close()
            conn.close()
            st.success("Prediction logged successfully.")
        except Exception as e:
            st.error(f"Database error: {e}")

# --- Display history ---
st.markdown("## History")
try:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT timestamp, predicted, label FROM predictions ORDER BY timestamp DESC LIMIT 10;")
    rows = cur.fetchall()
    st.table(rows)
    cur.close()
    conn.close()
except Exception as e:
    st.warning("Could not load history.")

# === One-time table creation ===
def initialize_db():
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    predicted INTEGER NOT NULL,
                    label INTEGER
                );
            """)
            conn.commit()
            cur.close()
            conn.close()
            st.success("✅ Database table initialized successfully.")
        except Exception as e:
            st.error(f"❌ DB Init Failed: {e}")

# Run once to initialize DB
initialize_db()
