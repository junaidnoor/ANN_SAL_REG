
import streamlit as st
import pandas as pd
import numpy as np
import pickle as pl
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler,OneHotEncoder

# Load the Trained Model
model = load_model('sal_reg_mdel.h5')

# load the Scaler, pickle, onehot
with open ('onehot_encoder_education.pkl','rb') as file:
    onehot_encoder_education=pl.load(file)

with open('onehot_encoder_job.pkl','rb') as file:
    onehot_encoder_job=pl.load(file)

with open('Sal_Reg_scaler.pkl','rb') as file:
    Sal_Reg_scaler=pl.load(file)

# Load expected columns
with open('expected_columns.pkl', 'rb') as file:
    expected_columns = pl.load(file)


# Streamlit Title
st.write("Salary Prediction")

# Users input
education_level = st.selectbox('Education Level',onehot_encoder_education.categories_[0])
job_role = st.selectbox('Job Role',onehot_encoder_job.categories_[0])
yearof_exp = st.slider('Year of Experience',1,50)

# Input Data
input_data=pd.DataFrame({
    'YearsExperience' : [yearof_exp]
})

# ---- One-Hot Encode 'Education Level' ----
try:
    education_array = onehot_encoder_education.transform([[education_level]]).toarray()
except Exception as e:
    raise ValueError(f"Unknown education level '{education_level}'") from e

education_array_df = pd.DataFrame(
    education_array,
    columns=onehot_encoder_education.get_feature_names_out(['Education Level'])
)

# ---- One-Hot Encode 'Job Role' ----
try:
    job_array = onehot_encoder_job.transform([[job_role]]).toarray()
except Exception as e:
    raise ValueError(f"Unknown job role '{job_role}'") from e

job_array_df = pd.DataFrame(
    job_array,
    columns=onehot_encoder_job.get_feature_names_out(['Job Role'])
)

## concatination of One Hot ecoded data ---- Combine all features ----
input_data = pd.concat([input_data.reset_index(drop=True),education_array_df,job_array_df],axis=1)

# ---- Add Missing Columns with 0 ----
for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0


# Ensure column order matches expected scaler input
input_data = input_data[expected_columns]

## Scale the input data
input_scaled = Sal_Reg_scaler.transform(input_data)

# Prediction the model
prediction = model.predict(input_scaled)

# convert numpy to float and round
prediction_prob = round(float(prediction[0][0]), 2)

st.write(f"predicted Salary: {prediction_prob: .2f}")

# Result
st.write(f"The predicted annual salary for a {job_role} with a {education_level} degree and {yearof_exp} years of experience is ${prediction_prob}.")




