import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


df = pd.read_csv("Telco-Customer-Churn.csv")


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'])


df.drop('customerID', axis=1, inplace=True)


df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})


cat_cols = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)


feature_columns = df.drop('Churn', axis=1).columns.tolist()
os.makedirs("model", exist_ok=True)
joblib.dump(feature_columns, 'model/feature_columns.pkl')


X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


joblib.dump(model, 'model/churn_model.pkl')

print("✅ Model and feature columns saved.")
