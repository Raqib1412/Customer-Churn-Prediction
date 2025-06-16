# 📉 Customer Churn Prediction
This project predicts whether a telecom customer is likely to churn based on service usage and demographic data. It involves data preprocessing, feature engineering, and training a Random Forest classifier. The final model is deployed using a Streamlit web app.

## 🚀 Features
- Predict customer churn using key inputs like tenure, contract type, monthly charges, etc.
- Clean and user-friendly web interface built with Streamlit.
- Interactive model predictions with real-time user input.
- Trained model and features stored using `joblib`.

## 📂 Tech Stack
- Python, Pandas, Scikit-learn
- Streamlit (for deployment)
- Joblib (for saving model artifacts)

## 🛠️ How to Run Locally
1. Clone the repo  
2. Install dependencies: `pip install -r requirements.txt`  
3. Run:  
```bash
streamlit run app.py
