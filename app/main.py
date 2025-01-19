import streamlit as st
import pickle
import os
import numpy as np


def predict_yield(Year, average_rain_fall_mm_per_year,pesticides_tonnes, avg_temp, Area, Item):
    features = np.array([[Year, average_rain_fall_mm_per_year,pesticides_tonnes, avg_temp, Area, Item]])
    transform_features = preprocesser.transform(features)
    predicted_yield = dtr.predict(transform_features).reshape(-1,1)
    return predicted_yield[0][0]

st.set_page_config(page_title="Yield Prediction")
st.header("Yield Prediction")
dataset_url = "https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset?select=yield.csv"
st.write("Model trained on [Crop Yield Prediction Dataset](%s)" % dataset_url)
current_path = os.path.dirname(__file__)
model_path = 'models'
relative_model_path = os.path.join(current_path, model_path)

dtr = pickle.load(open(relative_model_path + '/dtr.pkl', "rb"))
preprocesser = pickle.load(open(relative_model_path + '/preprocesser.pkl', "rb"))
countries = pickle.load(open(relative_model_path + '/country.pkl', "rb"))
crops = pickle.load(open(relative_model_path + '/crops.pkl', "rb"))

col1, col2 = st.columns(2)


with col1:
    year = st.selectbox("Year",range(1990, 2024))
    rainfall = st.number_input("Average rainfall (mm/year)", min_value=0.0, max_value=5000.0, step=0.1)
    pesticides_tonnes = st.number_input("Pesticides (tonnes)", min_value=0.0, max_value=1000.0, step=0.1)
    avg_temp = st.number_input("Average Temperature (°C)", min_value=-50.0, max_value=60.0, step=0.1)

with col2:
    selected_country = st.selectbox("Select Country", countries)
    selected_crop = st.selectbox("Select Crop", crops)

def validate_inputs(rainfall, pesticides_tonnes, avg_temp):
    if rainfall < 0 or rainfall > 5000:
        st.error("Rainfall must be a positive number between 0 and 5000 mm.")
        return False

    if pesticides_tonnes < 0 or pesticides_tonnes > 1000:
        st.error("Pesticides usage must be a positive number between 0 and 1000 tonnes.")
        return False

    if avg_temp < -50 or avg_temp > 60:
        st.error("Average temperature must be between -50°C and 60°C.")
        return False

    return True


if st.button("Predict"):
    if validate_inputs(rainfall, pesticides_tonnes, avg_temp):
        yield_predicted = predict_yield(year, rainfall, pesticides_tonnes, avg_temp, selected_country, selected_crop)
        st.success(f"Inputs are valid! Crop: {selected_crop} " 
                   f"Pesticides: {pesticides_tonnes} tonnes, Avg Temp: {avg_temp}°C. "
                   f"Area: {selected_country} , Predicted Yield: {yield_predicted}")
    else:
        pass




