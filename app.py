import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel(r"C:\Users\amald\OneDrive\Desktop\Stremlit\pima_diabetes.csv.xlsx")
    return df

df = load_data()

# -------------------------------
# Page Title
# -------------------------------
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")
st.title("Diabetes Prediction App")

# -------------------------------
# Sidebar Input Features
# -------------------------------
st.sidebar.header("Input Features")

def user_input_features():
    pregnancies = st.sidebar.number_input("Enter Pregnancies", 0.0, 20.0, 3.0, step=1.0)
    glucose = st.sidebar.number_input("Enter Glucose", 0.0, 200.0, 120.0, step=1.0)
    blood_pressure = st.sidebar.number_input("Enter Blood Pressure", 0.0, 140.0, 70.0, step=1.0)
    skin_thickness = st.sidebar.number_input("Enter Skin Thickness", 0.0, 100.0, 20.0, step=1.0)
    insulin = st.sidebar.number_input("Enter Insulin", 0.0, 900.0, 80.0, step=1.0)
    bmi = st.sidebar.number_input("Enter BMI", 0.0, 70.0, 32.0, step=0.1)
    dpf = st.sidebar.number_input("Enter DiabetesPedigreeFunction", 0.0, 3.0, 0.5, step=0.01)
    age = st.sidebar.number_input("Enter Age", 0.0, 120.0, 33.0, step=1.0)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    features = pd.DataFrame(data, index=[0])

    # ---- Prediction button at bottom ----
    st.sidebar.markdown("---")
    predict_btn = st.sidebar.button("Predict", use_container_width=True)

    return features, predict_btn

input_df, predict_btn = user_input_features()

# -------------------------------
# Dataset and Features
# -------------------------------
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Combine user input with dataset (for proper scaling if needed)
df_input = pd.concat([input_df, X], axis=0)

# -------------------------------
# Train-Test Split & Model
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# Main Area
# -------------------------------
if predict_btn:
    # ✅ Fix column names to match training data
    input_df.rename(columns={
        'BloodPressure': 'Blood pressure',
        'BMI': 'Body mass index',
        'DiabetesPedigreeFunction': 'Diabetes pedigree function',
        'SkinThickness': 'Skin thickness'
    }, inplace=True)

    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction")
    st.write("*Diabetes Prediction:*", "Positive" if prediction[0] == 1 else "Negative")
    st.write("Prediction Probability:", prediction_proba)

# -------------------------------
# Optional: Show dataset
# -------------------------------
if st.checkbox("Show Dataset"):
    st.subheader("Diabetes Dataset")
    st.dataframe(df)

if st.checkbox("Show Model Performance"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write("Accuracy of the model:", acc)

if st.checkbox("Show First Few Rows of Dataset"):
    st.write(df.head())

# -------------------------------
# Feature Importance
# -------------------------------
st.subheader("Feature Importance")
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances = feat_importances.sort_values(ascending=False)
st.dataframe(
    feat_importances.reset_index().rename(
        columns={'index': 'Feature', 0: 'Importance'}
    )
)
