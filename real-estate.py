import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Step 1: Data Preparation
@st.cache_data
def load_data():
    # Simulated dataset of properties
    data = pd.DataFrame({
        'id': range(1, 101),
        'price': np.random.randint(100000, 1000000, 100),
        'bedrooms': np.random.randint(1, 6, 100),
        'bathrooms': np.random.randint(1, 4, 100),
        'sq_ft': np.random.randint(500, 3000, 100),
        'year_built': np.random.randint(1950, 2023, 100),
        'location': np.random.choice(['urban', 'suburban', 'rural'], 100),
    })
    return data

# Step 2: Feature Engineering
def preprocess_data(data):
    # One-hot encode categorical features
    data_encoded = pd.get_dummies(data, columns=['location'])
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    numerical_features = ['price', 'bedrooms', 'bathrooms', 'sq_ft', 'year_built']
    data_encoded[numerical_features] = scaler.fit_transform(data_encoded[numerical_features])
    
    return data_encoded

# Step 3: Similarity Calculation
def get_similar_properties(data, property_id, top_n=5):
    property_features = data.drop('id', axis=1)
    similarity_matrix = cosine_similarity(property_features)
    similar_indices = similarity_matrix[property_id-1].argsort()[::-1][1:top_n+1]
    similar_properties = data.iloc[similar_indices]
    return similar_properties

# Streamlit App
st.title("Zillow-like Property Recommender System Demo with EDA")

# Load data
data = load_data()

# EDA Section
st.header("Exploratory Data Analysis")

# Display sample of the dataset
st.subheader("Sample Property Listings")
st.dataframe(data.head())

# Summary statistics
st.subheader("Summary Statistics")
st.write(data.describe())

# Distribution of numerical features
st.subheader("Distribution of Numerical Features")
numerical_features = ['price', 'bedrooms', 'bathrooms', 'sq_ft', 'year_built']
for feature in numerical_features:
    fig = px.histogram(data, x=feature, title=f"Distribution of {feature}")
    st.plotly_chart(fig)

# Scatter plot: Price vs Square Footage
st.subheader("Price vs Square Footage")
fig = px.scatter(data, x='sq_ft', y='price', color='bedrooms', 
                 title="Price vs Square Footage (colored by number of bedrooms)")
st.plotly_chart(fig)

# Location distribution
st.subheader("Location Distribution")
fig = px.pie(data, names='location', title="Property Location Distribution")
st.plotly_chart(fig)

# Correlation matrix
st.subheader("Correlation Matrix")
corr_matrix = data[numerical_features].corr()
fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                title="Correlation Matrix of Numerical Features")
st.plotly_chart(fig)

# Feature Engineering and Preprocessing
st.header("Feature Engineering and Preprocessing")
st.write("Preprocessing steps:")
st.write("1. One-hot encoding of categorical features (location)")
st.write("2. Normalization of numerical features using MinMaxScaler")

data_processed = preprocess_data(data)
st.subheader("Processed Data Sample")
st.dataframe(data_processed.head())

# Recommender System
st.header("Property Recommendation")
selected_property = st.selectbox("Select a property ID to get recommendations:", data['id'])

if st.button("Get Recommendations"):
    # Get similar properties
    similar_properties = get_similar_properties(data_processed, selected_property)
    
    # Display the selected property
    st.subheader("Selected Property:")
    st.dataframe(data[data['id'] == selected_property])
    
    # Display recommendations
    st.subheader("Recommended Properties:")
    st.dataframe(data[data['id'].isin(similar_properties['id'])])
    
    # Visualize similarities
    st.subheader("Feature Comparison")
    comparison_data = pd.concat([
        data[data['id'] == selected_property],
        data[data['id'].isin(similar_properties['id'])]
    ])
    
    features_to_compare = ['price', 'bedrooms', 'bathrooms', 'sq_ft', 'year_built']
    
    fig = px.bar(comparison_data, x='id', y=features_to_compare, 
                 title="Feature Comparison of Selected and Recommended Properties",
                 labels={'value': 'Value', 'variable': 'Feature'},
                 barmode='group')
    
    fig.update_layout(xaxis_title="Property ID")
    st.plotly_chart(fig)

    # Add a radar chart for a different perspective
    st.subheader("Property Feature Radar Chart")
    
    # Normalize the features for better visualization in the radar chart
    radar_data = comparison_data[features_to_compare]
    radar_data = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())
    radar_data['id'] = comparison_data['id']
    
    fig = px.line_polar(radar_data, r=features_to_compare, theta=features_to_compare, 
                        line_close=True, color='id',
                        title="Radar Chart of Property Features (Normalized)")
    st.plotly_chart(fig)

# Explanation of the recommender system
st.header("How it works")
st.write("""
This demo uses a content-based recommender system to suggest similar properties based on the selected property's features. Here's how it works:

1. Data Preparation: We start with a dataset of properties, including features like price, number of bedrooms, square footage, etc.
2. Exploratory Data Analysis (EDA): We analyze the dataset to understand the distribution of features, relationships between variables, and overall characteristics of the properties.
3. Feature Engineering: We preprocess the data by encoding categorical variables and normalizing numerical features.
4. Similarity Calculation: We use cosine similarity to measure how similar properties are to each other based on their features.
5. Recommendation: When you select a property, the system finds the most similar properties and recommends them.

This is a simplified version of how real estate platforms might implement recommendation systems. In practice, these systems often use more complex algorithms and take into account user behavior and preferences.
""")