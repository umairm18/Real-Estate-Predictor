# 🏠 Real Estate Price Predictor

A full-stack machine learning web application to predict real estate sale prices based on property attributes like lot size, year built, beds, baths, and property type.

---

## 📁 Project Structure

real-estate-predictor/
├── backend/
│ ├── main.py # FastAPI app
│ ├── train_model.py # Script to train LightGBM model
│ ├── SALESPRICE_model.pkl # Trained model
│ ├── label_encoder.pkl # Encoded PROPERTYTYPE
│ └── requirements.txt # Backend dependencies
├── frontend/ # Angular frontend
│ ├── src/
│ ├── angular.json
│ └── ...
├── AI Hakathon.csv # Raw dataset used for training
├── README.md
└── .gitignore

---

## 🚀 Getting Started

### ⚙️ Prerequisites

- Python 3.10+  
- Node.js + npm  
- Angular CLI  
- Git

---

## 🔧 Backend Setup (FastAPI + LightGBM)

1. **Navigate to backend folder**:

   ```bash
   cd backend
Create virtual environment and activate:
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

Install dependencies:
pip install -r requirements.txt
Train model (optional — only if not using saved model):
python train_model.py

Run FastAPI server:
uvicorn main:app --reload
API will run at: http://localhost:8000/predict

🖼️ Frontend Setup (Angular)
Navigate to frontend folder:
cd frontend

Install dependencies:
npm install
Run the Angular dev server:

ng serve
App will run at: http://localhost:4200

🧪 Sample Input Fields
The frontend takes these inputs:
Lot Size (in Acres)
Year Built
Beds
Baths
Property Type (Dropdown)

🔁 Model Behavior
Predicts sale price using LightGBM.

If input combination matches existing CSV entries:
Shows average SALESPRICE, ESTIMATEDVALUE, and CONFIDENCESCORE.
Compares errors between predicted and actuals.

If no match:
Shows only predicted price with 0% confidence.
