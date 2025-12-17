

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


def computer_prediction(input_data):
    
  
# daily_usage_hours, charging_cycles, avg_charge_limit_percent,
# battery_age_months, battery_health_percent

   # Changing the input_data to numpy array
  input_data_as_numpy_array = np.asarray(input_data)

  # Reshape the array as we are predicting for one instance
  input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

  prediction = loaded_model.predict(input_data_reshaped)
  print(prediction)
  if prediction[0] == 0:
    return'The laptop has no overheating issue'
  else:
    return'The laptop has an overheating issue'
    
    
def main():
      #    create title
      st.title("Laptop Overheating Prediction App")
      st.header("Enter Laptop Parameters")

      # User inputs
      model_year = st.number_input("Model Year", min_value=2010, max_value=2030, value=2021)
      daily_usage_hours = st.number_input("Daily Usage Hours", min_value=0.0, max_value=24.0, value=6.0)
      charging_cycles = st.number_input("Charging Cycles", min_value=0, value=400)
      avg_charge_limit_percent = st.number_input("Average Charge Limit (%)", min_value=50, max_value=100, value=90)
      battery_health_percent = st.number_input("Battery Health (%)", min_value=0, max_value=100, value=85)
      battery_age_months = st.number_input("Battery Age (Months)", min_value=0, value=36)
   # code for prediction 
      diagnosis=''
      # Prediction button
      if st.button("Predict Overheating Issue"):

         diagnosis = computer_prediction([[model_year,
                             daily_usage_hours,
                             charging_cycles,
                             avg_charge_limit_percent,
                             battery_health_percent,
                             battery_age_months]])

      st.success(diagnosis)
      
    
if __name__=='__main__':
    main()

