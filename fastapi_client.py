import requests
import streamlit as st

API_URL = "http://localhost:8000"  # Adjust this if your FastAPI server is running on a different port

st.title("Zillow-like Property Recommender Demo")

# Get recommendations based on a property ID
st.header("Get Recommendations by Property ID")
property_id = st.number_input("Enter a property ID:", min_value=1, max_value=1000, value=1)
if st.button("Get Recommendations by ID"):
    response = requests.get(f"{API_URL}/recommendations/{property_id}")
    if response.status_code == 200:
        recommendations = response.json()["recommendations"]
        st.write("Recommended Properties:")
        st.dataframe(recommendations)
    else:
        st.error("Error fetching recommendations")

# Get recommendations based on features
st.header("Get Recommendations by Features")
price = st.number_input("Price:", min_value=100000, max_value=1000000, value=500000)
bedrooms = st.number_input("Bedrooms:", min_value=1, max_value=5, value=3)
bathrooms = st.number_input("Bathrooms:", min_value=1, max_value=3, value=2)
sq_ft = st.number_input("Square Feet:", min_value=500, max_value=3000, value=1500)
year_built = st.number_input("Year Built:", min_value=1950, max_value=2023, value=2000)
location = st.selectbox("Location:", ["urban", "suburban", "rural"])
lot_size = st.number_input("Lot Size:", min_value=1000, max_value=10000, value=5000)
garage_spaces = st.number_input("Garage Spaces:", min_value=0, max_value=3, value=1)
condition = st.selectbox("Condition:", ["excellent", "good", "fair", "poor"])
school_rating = st.number_input("School Rating:", min_value=1, max_value=10, value=5)

if st.button("Get Recommendations by Features"):
    features = {
        "price": price,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sq_ft": sq_ft,
        "year_built": year_built,
        "location": location,
        "lot_size": lot_size,
        "garage_spaces": garage_spaces,
        "condition": condition,
        "school_rating": school_rating
    }
    response = requests.post(f"{API_URL}/recommendations", json=features)
    if response.status_code == 200:
        recommendations = response.json()["recommendations"]
        st.write("Recommended Properties:")
        st.dataframe(recommendations)
    else:
        st.error("Error fetching recommendations")

# Get property details
st.header("Get Property Details")
property_id = st.number_input("Enter a property ID for details:", min_value=1, max_value=1000, value=1)
if st.button("Get Property Details"):
    response = requests.get(f"{API_URL}/property/{property_id}")
    if response.status_code == 200:
        property_details = response.json()
        st.write("Property Details:")
        st.json(property_details)
    else:
        st.error("Error fetching property details")
