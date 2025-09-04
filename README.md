# House_Pricing_Prediction_Model
Programming Languages : Python Libraries/Frameworks : Data Handling : Numpy , Pandas Data Visualization : Matpplotlib , Seaborn Machine Learning : Scikit Learn,XGBoost
#---Importing All Libraries---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

# --- 1. Generate Mock Dataset ---
np.random.seed(42)

data = {
    'Location': np.random.choice(['Bandra', 'Andheri', 'Thane', 'Chembur', 'Goregaon', 'Dadar', 'Navi Mumbai', 'Powai'], size=2000),
    'Bedrooms': np.random.randint(1, 6, size=2000),
    'SquareFeet': np.random.randint(400, 5000, size=2000),
    'Price': []
}

location_base_price_per_sqft = {
    'Bandra': 45000,
    'Andheri': 30000,
    'Thane': 18000,
    'Chembur': 28000,
    'Goregaon': 25000,
    'Dadar': 38000,
    'Navi Mumbai': 15000,
    'Powai': 35000,
}

for i in range(len(data['Location'])):
    loc = data['Location'][i]
    sqft = data['SquareFeet'][i]
    beds = data['Bedrooms'][i]

    base_price = sqft * location_base_price_per_sqft[loc]

    bedroom_multiplier = 1 + (beds - 1) * 0.08
    price = base_price * bedroom_multiplier

    random_noise = np.random.normal(0, 0.15 * price)
    final_price = price + random_noise

    data['Price'].append(max(1000000, final_price*2))

df = pd.DataFrame(data)

# --- 2. Feature Engineering ---
X = df[['Location', 'Bedrooms', 'SquareFeet']]
y = df['Price']

# --- 3. Data Preprocessing (One-Hot Encoding for Location) ---
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), ['Location'])
    ],
    remainder='passthrough'
)

X_processed = preprocessor.fit_transform(X)

# --- 4. Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# --- 5. Model Training ---
print("\nTraining the RandomForestRegressor model...")
model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Model training complete.")

# --- 6. Model Evaluation ---
print("\nEvaluating the model performance on the test data...")
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R-squared (R2) Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): ₹{mae:,.2f}")

# --- 7. Prediction Function ---
def predict_house_price(location, bedrooms, square_feet, preprocessor_obj, trained_model):
    input_data = pd.DataFrame([[location, bedrooms, square_feet]],
                             columns=['Location', 'Bedrooms', 'SquareFeet'])

    processed_input = preprocessor_obj.transform(input_data)

    predicted_price = trained_model.predict(processed_input)[0]
    return predicted_price

# --- Example Predictions ---
print("\n--- Making a few sample predictions ---")
print("Asking Input from an User :  ")

sample_location_1 = input("Enter the location of an house : ")
sample_bedrooms_1 = int(input("Enter the number of beedrooms: "))
sample_square_feet_1 = float(input("Enter the Area in square feet : "))
predicted_price_1 = predict_house_price(sample_location_1, sample_bedrooms_1, sample_square_feet_1, preprocessor, model)
print(f"Predicted price for a {sample_bedrooms_1} BHK in {sample_location_1} of {sample_square_feet_1} sqft: ₹{predicted_price_1:,.2f}")

