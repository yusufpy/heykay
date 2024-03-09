import streamlit as st
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define a function to make predictions
def predict_emission(latitude, longitude, year, week_no, SO2_column_density_amf, SO2_slant_column_density, cloud_top_height, Formaldehyde_cloud_fraction, Ozone_cloud_fraction):
    data = pd.DataFrame({
        'latitude': [latitude],
        'longitude': [longitude],
        'year': [year],
        'week_no': [week_no],
        'SulphurDioxide_SO2_column_number_density_amf': [SO2_column_density_amf],
        'SulphurDioxide_SO2_slant_column_number_density': [SO2_slant_column_density],
        'Cloud_cloud_top_height': [cloud_top_height],
        'Formaldehyde_cloud_fraction': [Formaldehyde_cloud_fraction],
        'Ozone_cloud_fraction': [Ozone_cloud_fraction]
    })
    prediction = model.predict(data)
    return prediction[0]

# Streamlit app
def main():
    st.title('Emission Prediction')

    st.write('Enter the required parameters to predict the emission:')
    latitude = st.number_input('Latitude')
    longitude = st.number_input('Longitude')
    year = st.number_input('Year')
    week_no = st.number_input('Week Number')
    SO2_column_density_amf = st.number_input('SulphurDioxide_SO2_column_number_density_amf')
    SO2_slant_column_density = st.number_input('SulphurDioxide_SO2_slant_column_number_density')
    cloud_top_height = st.number_input('Cloud_cloud_top_height')
    Formaldehyde_cloud_fraction = st.number_input('Formaldehyde_cloud_fraction')
    Ozone_cloud_fraction = st.number_input('Ozone_cloud_fraction')

    if st.button('Predict'):
        prediction = predict_emission(latitude, longitude, year, week_no, SO2_column_density_amf, SO2_slant_column_density, cloud_top_height, Formaldehyde_cloud_fraction, Ozone_cloud_fraction)
        st.write(f'Predicted Emission: {prediction}')

if __name__ == '__main__':
    main()
