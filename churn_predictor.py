import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
# Title and form decoration
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Mr. Churn Predictor 🕵️‍♀️</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #4CAF50;'>", unsafe_allow_html=True)

# Name
Name = st.text_input("Name", placeholder="Enter your full name")

# Age
Age = st.number_input("Age", min_value=18, max_value=100, help="Your age must be between 18 and 100.")

# Gender
Gender = st.selectbox("Gender", ['Male', 'Female', 'Other'], index=0)

# Country
Country = st.selectbox("Country", ['France', "Germany", "Spain"])

# Credit Score
CreditScore = st.number_input("Credit Score", min_value=300, max_value=850, help="Enter your credit score (300 to 850).")

# Is Active Member
IsActiveMember = st.selectbox("Are you an active member?", ['Yes', 'No'])

# Tenure
Tenure = st.number_input("Tenure (in years)", min_value=0, max_value=30, help="Number of years you've been a member.")

# Balance
Balance = st.number_input("Balance", min_value=0, max_value=100000, help="Enter your current balance.")

# Number of Products
NumofProducts = st.number_input("Number of Products", min_value=1, max_value=10, help="Number of products you have.")

# Has Credit Card
HasCrCard = st.selectbox("Do you have a credit card?", ['Yes', 'No'])

# Load Encoders and Scaler
with open(r'Models_Scalers\gender_encoder.pkl', 'rb') as file:
    gender_encoder = pickle.load(file)
    
with open(r'Models_Scalers\geo_Encoder.pkl', 'rb') as file:
    geo_encoder = pickle.load(file)
    
with open(r'Models_Scalers\scalar.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load Model
model = load_model('Models_Scalers\model.h5')

# Process inputs and make predictions
if st.button('Submit'):
    # Encode Gender and Country
    Gender_encoded = gender_encoder.transform([Gender])
    # Country
    
# Ensure Country is transformed correctly using the one-hot encoder
    try:    
        # OneHotEncoder expects input as a 2D array
        Country_encoded = geo_encoder.transform([[Country]]).toarray() 
        print("Country")
    except ValueError as e:
        st.error(f"Country one-hot encoding failed: {e}")
        st.stop()  # Stop further execution if encoding fails


    # Convert IsActiveMember and HasCrCard to numeric
    IsActiveMember_numeric = 1 if IsActiveMember == 'Yes' else 0
    HasCrCard_numeric = 1 if HasCrCard == 'Yes' else 0

    # Create input DataFrame
    form_data = {
        'CreditScore': CreditScore,
        'Gender': Gender_encoded[0],
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumofProducts,
        'HasCrCard': HasCrCard_numeric,
        'IsActiveMember': IsActiveMember_numeric,
        'EstimatedSalary': 0,  # Placeholder
        'Geography_France': Country_encoded[0][0],
        'Geography_Germany': Country_encoded[0][1],
        'Geography_Spain' :  Country_encoded[0][2]
    }

    input_df = pd.DataFrame([form_data])

    # Scale the data
    scaled_data = scaler.transform(input_df)

    # Make predictions
    prediction = model.predict(scaled_data)

    # Display results
    st.write("Form Submitted Successfully!")
    prediction = prediction[0][0]
    st.subheader(prediction)
    exit_status = st.warning("Exited") if prediction >= 0.5 else st.success("Not Exited")
    
