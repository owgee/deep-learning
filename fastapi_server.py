from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

# Load and preprocess data
data = pd.read_csv("property_data.csv")

def preprocess_data(df):
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=['location', 'condition'])
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    numerical_features = ['price', 'bedrooms', 'bathrooms', 'sq_ft', 'year_built', 'lot_size', 'garage_spaces', 'school_rating']
    df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])
    
    return df_encoded

preprocessed_data = preprocess_data(data)

class PropertyFeatures(BaseModel):
    price: float
    bedrooms: int
    bathrooms: float
    sq_ft: float
    year_built: int
    location: str
    lot_size: float
    garage_spaces: int
    condition: str
    school_rating: int

def get_similar_properties(data, property_id, top_n=5):
    property_features = data.drop('id', axis=1)
    similarity_matrix = cosine_similarity(property_features)
    similar_indices = similarity_matrix[property_id-1].argsort()[::-1][1:top_n+1]
    similar_properties = data.iloc[similar_indices]
    return similar_properties

@app.get("/recommendations/{property_id}")
async def get_recommendations(property_id: int, num_recommendations: int = 5):
    if property_id not in data['id'].values:
        raise HTTPException(status_code=404, detail="Property not found")
    similar_properties = get_similar_properties(preprocessed_data, property_id, num_recommendations)
    return {"recommendations": similar_properties.to_dict(orient='records')}

@app.post("/recommendations")
async def get_recommendations_by_features(features: PropertyFeatures, num_recommendations: int = 5):
    # Create a DataFrame with the input features
    input_df = pd.DataFrame([features.dict()])
    
    # Preprocess the input data
    input_preprocessed = preprocess_data(input_df)
    
    # Add the preprocessed input to the existing data
    combined_data = pd.concat([preprocessed_data, input_preprocessed], ignore_index=True)
    
    # Get recommendations
    similar_properties = get_similar_properties(combined_data, len(combined_data), num_recommendations)
    
    # Return the original (non-preprocessed) data for these properties
    original_similar_properties = data[data['id'].isin(similar_properties['id'])]
    return {"recommendations": original_similar_properties.to_dict(orient='records')}

@app.get("/property/{property_id}")
async def get_property(property_id: int):
    property_data = data[data['id'] == property_id]
    if property_data.empty:
        raise HTTPException(status_code=404, detail="Property not found")
    return property_data.to_dict(orient='records')[0]
