 #  MNIST Digit Classifier 

This project is an end-to-end machine learning application that classifies handwritten digits using a CNN trained on the MNIST dataset.

---

##  Live Demo

🔗 [Try the Live App](https://www.presizely.co.uk/)

---

##  Project Goals

- ✅ Train a CNN using PyTorch on the MNIST dataset
- ✅ Build an interactive UI with Streamlit to draw digits
- ✅ Predict digit, show confidence, and collect feedback
- ✅ Log predictions in a PostgreSQL database
- ✅ Containerize the application with Docker & Docker Compose
- ✅ Deploy to a public server using Railway

---

##  Tech Stack

- **Python 3.10**
- **PyTorch** for model training and inference
- **Streamlit** for the web UI
- **PostgreSQL** for logging predictions
- **Docker** & **Docker Compose** for containerization
- **Railway** for deployment

---

## Features

- Draw digits on an interactive canvas
- Predicts digit with confidence score
- User submits the true label for feedback
- Prediction results logged to PostgreSQL
- Preview the preprocessed 28×28 input image
- Optional binary thresholding for input clarity
- Sidebar shows:
  - DB health
  - Total predictions
  - Accuracy %
  - Recent predictions
- Download all predictions as CSV

---

## Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/loredanagaspar/digit-classifier-loredanagaspar.git
cd digit-classifier-loredanagaspar
```

### 2. Train the model
```bash
python model/train.py
```

### 3. Run Streamlit app
```bash
streamlit run app/app.py
```

> Ensure your `.env` file contains the correct `DATABASE_URL`.

### 4. Docker Compose
```bash
docker-compose up --build
```

---

##  Project Structure
```
├── app/
│   └── app.py             # Streamlit app UI
├── model/
│   ├── train.py           # Model training
│   └── model.pt           # Trained model
├── db/
│   └── init_db.py         # Create predictions table
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env                   # PostgreSQL credentials (not committed)
```

---

## Submit

- ✅ Pushed to GitHub
- ✅ Includes all code, Dockerfile, and documentation
- ✅ Public link to the deployed app

---

## Author

Built by [Loredana Gaspar](https://github.com/loredanagaspar) 

