import streamlit as st
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Initialize or load the DataFrame for storing results with custom column names
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=['Town', 'Flat Type', 'Lease Commencement Date', 'Storey Range', 'Floor Area (SQ FT)', 'Resale Price'])

# Sidebar information
st.sidebar.header('Choose Options')
st.sidebar.markdown("""
Select options from the dropdown menus to display the predictions and data.
""")

# Define model paths
model_paths = {
    "lightgbm": "lightgbm_model.txt"
}

# Define CSV path
csv_path = {
    "final_variable": "userinputvariable.csv"
}

# Load LightGBM model
@st.cache_resource
def load_model(model_path):
    model = lgb.Booster(model_file=model_path)
    return model

# Load models
@st.cache_resource
def load_models(model_paths):
    models = {}
    for model_name, model_path in model_paths.items():
        try:
            models[model_name] = load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model '{model_name}': {e}")
    return models

# Load CSV data
@st.cache_resource
def load_data(csv_path):
    try:
        data = pd.read_csv(csv_path)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load models and historical data
models = load_models(model_paths)
final_combined_data = load_data(csv_path["final_variable"])

# Get feature names from the model
model = models['lightgbm']
model_feature_names = model.feature_name()

# Create LabelEncoders based on the final_combined_data
label_encoders = {}
for feature in ['town', 'flat_type', 'max_floor_lvl']:
    le = LabelEncoder()
    le.fit(final_combined_data[feature])
    label_encoders[feature] = le

# Streamlit App
col1, col2 = st.columns([1, 4])  # Adjust the width ratio as needed

with col1:
    st.image("wow.png", width=100)  # Adjust width as necessary

with col2:
    st.header("🏠 HDB Resale Price Predictor")

st.write("""
This HDB Resale Price Predictor is created by DJ BAB! 🧑🏽‍💻 
Using a LightGBM regression predictive model of history data from 2012-2021
""")

# User input for features
town = st.sidebar.selectbox('Select Town:', final_combined_data['town'].unique())
flat_type = st.sidebar.selectbox('Select Flat Type:', final_combined_data['flat_type'].unique())
lease_commence_date = st.sidebar.selectbox('Select Lease Commencement Date:', final_combined_data['lease_commence_date'].unique())
storey_range = st.sidebar.selectbox('Select Storey Range:', final_combined_data['storey_range'].unique())  # Keep this for user input
floor_area = st.sidebar.slider('Select Floor Area (Sq Ft):', min_value=int(final_combined_data['floor_area_sqft'].min()), 
                               max_value=int(final_combined_data['floor_area_sqft'].max()), value=int(final_combined_data['floor_area_sqft'].mean()))

# Retrieve hdb_age corresponding to the selected lease_commence_date
hdb_age = final_combined_data.loc[final_combined_data['lease_commence_date'] == lease_commence_date, 'hdb_age'].values[0]

# Convert storey_range to max_floor_lvl for prediction purposes
max_floor_lvl = final_combined_data.loc[final_combined_data['storey_range'] == storey_range, 'max_floor_lvl'].values[0]

# Create a DataFrame from user input
input_data = pd.DataFrame({
    'town': [town],
    'flat_type': [flat_type],
    'hdb_age': [hdb_age],  # Use hdb_age instead of lease_commence_date
    'max_floor_lvl': [max_floor_lvl],  # Use max_floor_lvl instead of storey_range
    'floor_area_sqft': [floor_area],
})

# Encode the 'town' variable for prediction
input_data['town_Encoded'] = label_encoders['town'].transform(input_data['town'])
input_data['flat_type_Encoded'] = label_encoders['flat_type'].transform(input_data['flat_type'])
# Transform other features
input_data['max_floor_lvl'] = label_encoders['max_floor_lvl'].transform(input_data['max_floor_lvl'])

# Ensure column order and consistency
input_data = input_data.reindex(columns=model_feature_names, fill_value=0)

# Initialize a variable for prediction result
prediction_result = None

if st.sidebar.button('Predict Resale Price'):
    model = models['lightgbm']
    
    try:
        # Perform prediction
        prediction = model.predict(input_data, raw_score=False, predict_disable_shape_check=True)[0]
        prediction_result = prediction
        
        # Create a DataFrame for the new row with custom column names
        new_row = pd.DataFrame({
            'Town': [town],
            'Flat Type': [flat_type],
            'Lease Commencement Date': [str(lease_commence_date)],  # Display lease_commence_date for user input
            'Storey Range': [storey_range],  # Display storey_range for the user
            'Floor Area (SQ FT)': [floor_area],
            'Resale Price': [f"${prediction_result:,.2f}"],
        })

        # Add the new row to the results DataFrame
        st.session_state.results_df = pd.concat([st.session_state.results_df, new_row], ignore_index=True)
        
        # Display all results DataFrame
        st.subheader('All Predictions Results')
        st.markdown(f"<h2>Resale Price: ${prediction_result:,.2f}</h2>", unsafe_allow_html=True)  # Larger text
        st.dataframe(st.session_state.results_df)
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Download Results as CSV with custom column names
csv = st.session_state.results_df.to_csv(index=False)
st.sidebar.download_button(label="Download Results as CSV", data=csv, file_name="hdb_resale_predictions.csv")
