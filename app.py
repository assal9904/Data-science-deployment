import streamlit as st
import joblib
import pandas as pd

rf = joblib.load('model.pkl')



st.title("Program expenditure prediction random forest model")
states = ['NY', 'HI', 'CA', 'GA', 'AZ', 'FL', 'OR', 'NC', 'OH', 'WI', 
          'OK', 'WA', 'MA', 'DC', 'PA', 'VA', 'NM', 'CO', 'MD', 'TX', 
          'MI', 'MN', 'TN', 'IL', 'ME', 'KY', 'NJ', 'UT', 'MS', 'MO', 
          'AL', 'AK', 'KS', 'CT', 'IN', 'WV', 'IA', 'MT', 'SC', 'NV', 
          'AR', 'RI', 'NE', 'ID', 'SD', 'DE', 'LA', 'VT', 'WY', 'NH', 
          'PR', 'VI', 'ND']


selected_state = st.selectbox('Select a State', states)

total_rev = st.number_input("Total revenue", min_value=0)


state_encoded = {f"state_copy_{state}": 1 if state == selected_state else 0 for state in states}


state_encoded_df = pd.DataFrame([state_encoded])


input_df = pd.DataFrame([[total_rev] + state_encoded_df.values[0].tolist()],
                        columns=['total_rev'] + state_encoded_df.columns.tolist())

input_df = input_df[rf.feature_names_in_.tolist()]

if st.button("Predict"):
    prediction = rf.predict(input_df)[0]
    st.success(f"Prediction: {prediction}")