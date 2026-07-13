import streamlit as st
import pickle
import numpy as np

@st.cache_resource
def load_artifacts():
    with open('housing_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('housing_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

reg_model, scaler = load_artifacts()

st.title("California Housing Price Estimator")
st.write("Fill in block details:")

c1, c2 = st.columns(2)
with c1:
    inc = st.number_input("Median Income (10k USD)", value=4.0)
    age = st.number_input("House Age (yrs)", value=20.0)
    rooms = st.number_input("Avg Rooms", value=6.0)
    bed = st.number_input("Avg Bedrooms", value=1.2)
with c2:
    pop = st.number_input("Population", value=1200.0)
    occ = st.number_input("Avg Occupancy", value=3.2)
    lat = st.number_input("Latitude", value=36.0)
    lon = st.number_input("Longitude", value=-119.0)

if st.button("Estimate Price", type="primary"):
    arr = np.array([[inc, age, rooms, bed, pop, occ, lat, lon]])
    scaled = scaler.transform(arr)
    pred = reg_model.predict(scaled)[0]
    st.success(f"### Estimated Price: ${pred * 100000:,.2f}")
