import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




df = pd.read_csv("house_price_dataset_10000 (1).csv")


print("\n========== DATASET ==========")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# Remove missing values
df.dropna(inplace=True)

# Encode Categorical Columns


city_encoder = LabelEncoder()
locality_encoder = LabelEncoder()

df["city"] = city_encoder.fit_transform(df["city"])
df["locality"] = locality_encoder.fit_transform(df["locality"])


# Features and Target


X = df.drop("price", axis=1)
y = df["price"]


# Train Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Build Model


model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Model...")

model.fit(X_train, y_train)

print("Training Completed Successfully!")


# Prediction


predictions = model.predict(X_test)


# Evaluation


mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\n========== MODEL PERFORMANCE ==========")
print(f"MAE        : {mae:.2f}")
print(f"MSE        : {mse:.2f}")
print(f"RMSE       : {rmse:.2f}")
print(f"R2 Score   : {r2:.4f}")


# Feature Importance


importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance)


# Save Model


os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/house_price_model.pkl")
joblib.dump(city_encoder, "model/city_encoder.pkl")
joblib.dump(locality_encoder, "model/locality_encoder.pkl")

print("\n===================================")
print("Model Saved Successfully!")
print("Model Folder : model/")
print("house_price_model.pkl")
print("city_encoder.pkl")
print("locality_encoder.pkl")
print("===================================")


import joblib
import pandas as pd


# Load Model and Encoders


model = joblib.load("model/house_price_model.pkl")
city_encoder = joblib.load("model/city_encoder.pkl")
locality_encoder = joblib.load("model/locality_encoder.pkl")

print("=" * 50)
print("        HOUSE PRICE PREDICTION")
print("=" * 50)


# User Input


area = float(input("Enter Area (sqft): "))
bedrooms = int(input("Enter Bedrooms: "))
bathrooms = int(input("Enter Bathrooms: "))
floors = int(input("Enter Floors: "))
parking = int(input("Enter Parking Spaces: "))
age = int(input("Enter House Age (Years): "))

print("\nAvailable Cities:")
print(list(city_encoder.classes_))
city = input("Enter City: ")

print("\nAvailable Localities:")
print(list(locality_encoder.classes_))
locality = input("Enter Locality: ")

school = float(input("School Distance (km): "))
hospital = float(input("Hospital Distance (km): "))
metro = float(input("Metro Distance (km): "))


# Encode Inputs


# Make lookup case-insensitive
city_dict = {c.lower(): i for i, c in enumerate(city_encoder.classes_)}
locality_dict = {l.lower(): i for i, l in enumerate(locality_encoder.classes_)}

city = city.strip().lower()
locality = locality.strip().lower()

if city not in city_dict:
    print("Invalid City!")
    print("Choose from:", list(city_encoder.classes_))
    exit()

if locality not in locality_dict:
    print("Invalid Locality!")
    print("Choose from:", list(locality_encoder.classes_))
    exit()

city = city_dict[city]
locality = locality_dict[locality]


# Create DataFrame


sample = pd.DataFrame({
    "area_sqft": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "floors": [floors],
    "parking": [parking],
    "age_years": [age],
    "city": [city],
    "locality": [locality],
    "school_distance_km": [school],
    "hospital_distance_km": [hospital],
    "metro_distance_km": [metro]
})


#Prediction


price = model.predict(sample)[0]

print("\n" + "=" * 50)
print("Predicted House Price")
print("=" * 50)

print(f"Estimated Price : ₹ {price:,.2f}")

print("=" * 50)



