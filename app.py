import streamlit as st
import pandas as pd
import pickle
import gdown
import pickle
import os

# URL of your file from Google Drive
url = 'https://drive.google.com/file/d/1imvIjMo4xB15Sjo9HRr25R9LAOMXgV1I/view?usp=sharing'
output = 'stacking_regressor.pkl'

# Download the file if it doesn't exist locally
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# Load the downloaded model
with open(output, 'rb') as model_file:
    model = pickle.load(model_file)

# Load the preprocessor
with open('preprocessor.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)

# Load the RFE model if needed
with open('rfe_model.pkl', 'rb') as rfe_file:
    rfe_model = pickle.load(rfe_file)

# Define preprocessing functions
def preprocess_input(user_input):
    # Convert input to DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Apply the preprocessor to the input DataFrame
    processed_input = preprocessor.transform(input_df)
    
    # Apply RFE transformation if applicable
    # Uncomment this if RFE is necessary
    # processed_input = rfe_model.transform(processed_input)
    
    return processed_input

# Streamlit UI
st.title("Caloric Burn Prediction App")

# User input fields
user_id = st.number_input("User ID")
gender = st.selectbox("Gender", ["male", "female", "Other"])
age = st.number_input("Age")
height = st.number_input("Height (cm)")
weight = st.number_input("Weight (kg)")
duration = st.number_input("Duration (minutes)")
heart_rate = st.number_input("Heart Rate")
body_temp = st.number_input("Body Temperature")

# Prepare the input for prediction
if st.button("Predict"):
    user_input = {
        "User_ID": user_id,
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
    }

    # Preprocess input
    processed_input = preprocess_input(user_input)

    # Make prediction
    prediction = model.predict(processed_input)

    # Display result
    st.success(f"Predicted Calories Burned: {prediction[0]:.2f}")
