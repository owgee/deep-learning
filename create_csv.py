import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Generate sample data
n_samples = 1000

data = pd.DataFrame({
    'id': range(1, n_samples + 1),
    'price': np.random.randint(100000, 1000000, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.randint(1, 4, n_samples),
    'sq_ft': np.random.randint(500, 3000, n_samples),
    'year_built': np.random.randint(1950, 2023, n_samples),
    'location': np.random.choice(['urban', 'suburban', 'rural'], n_samples),
    'lot_size': np.random.randint(1000, 10000, n_samples),
    'garage_spaces': np.random.randint(0, 4, n_samples),
    'condition': np.random.choice(['excellent', 'good', 'fair', 'poor'], n_samples),
    'school_rating': np.random.randint(1, 11, n_samples)
})

# Add some correlations
data['price'] = data['price'] + (data['sq_ft'] * 100) + (data['bedrooms'] * 10000) + (data['school_rating'] * 5000)
data['price'] = data['price'] + (data['year_built'] - 1950) * 1000
data['price'] = np.where(data['location'] == 'urban', data['price'] * 1.2, data['price'])
data['price'] = np.where(data['location'] == 'suburban', data['price'] * 1.1, data['price'])
data['price'] = data['price'].astype(int)

# Save to CSV
data.to_csv('property_data.csv', index=False)

print("CSV file 'property_data.csv' has been created successfully.")
