import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
st.set_page_config(layout="wide")

scaler = joblib.load("Scaler.pkl")

st.title("Restaurant Rating Prediction App")

st.caption("This app helps you to predict a restaurants review class")
st.divider()

averageCost = st.number_input("Please enter the estimated avg cost for 2 person",min_value=50,max_value=999999,value = 1000,step=200)
tablebooking = st.selectbox("Restaurant has table booking?", ["Yes", "No"])
onlinedelivery = st.selectbox("Restaurant has online delivery?", ["Yes", "No"])
pricerange = st.selectbox("What is our price range (1 cheapest,4 Most Expensive)",[1,2,3,4])

predictbutton = st.button("Predict the review")
st.divider()

model = joblib.load("linear_regression_model.pkl")

bookingstatus = 1 if tablebooking == "Yes" else 0

deliverystatus = 1 if onlinedelivery == "Yes" else 0

values = [[averageCost, bookingstatus, deliverystatus, pricerange]]
my_X_values = np.array(values)

X = scaler.transform(my_X_values)

if predictbutton:
    st.snow()
    prediction = model.predict(X)
    st.subheader("Prediction Result")
    st.success(f"Predicted Rating: {round(float(prediction[0][0]), 1)}")
    if prediction < 2.5:
        st.write("Poor")
    elif prediction < 3.5:
        st.write("Average")
    elif prediction < 4.0:
        st.write("Good")
    elif prediction < 4.5:
        st.write("Very Good")
    else:
        st.write("Excellent")