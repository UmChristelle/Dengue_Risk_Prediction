# -------------------------------
# Dengue Risk Prediction Project
# SDG 3: Ensure healthy lives and promote well-being for all at all ages
# Author: Christella
# -------------------------------

# Step 1: Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Step 2: Load dataset
# Make sure 'dengue_data.csv' is in the same folder as this script
data = pd.read_csv("dengue_data.csv")

# Inspect the data
print("First 5 rows of dataset:")
print(data.head())
print("\nDataset info:")
print(data.info())

# Step 3: Preprocessing
# Encode 'city' column (categorical)
le = LabelEncoder()
data['city'] = le.fit_transform(data['city'])

# Replace missing values of all numeric columns with median
medianFiller = lambda x: x.fillna(x.median())
data = data.apply(medianFiller)

numeric_cols = ['temperature', 'rainfall', 'humidity']
# Scale numeric features
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Step 4: Define features (X) and target (y)
X = data[['city', 'weekofyear', 'temperature', 'rainfall', 'humidity']]
y = data['dengue_cases']

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 8: Make sample predictions
# Example: Predict dengue cases for CityA (encoded as 0), Week 25, Temp=30, Rainfall=200, Humidity=80
sample = pd. DataFrame([[0, 25, 30, 200, 80]], columns=['city', 'weekofyear', 'temperature', 'rainfall', 'humidity'])
sample[numeric_cols] = scaler.transform(sample[numeric_cols])  # Apply same scaling
predicted_cases = model.predict(sample)
print(f"Predicted Dengue Cases for sample input: {int(predicted_cases[0])}")

# Step 9: Ethical considerations
print("\nEthical Considerations:")
print("- Ensure dataset represents all regions to avoid bias.")
# Step 10: Visualizations
import matplotlib.pyplot as plt

# Plot 1: Actual vs Predicted Dengue Cases (on test set)
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Perfect prediction line
plt.xlabel("Actual Dengue Cases")
plt.ylabel("Predicted Dengue Cases")
plt.title("Actual vs Predicted Dengue Cases")
plt.grid(True)
plt.show()

# Plot 2: Average Dengue Cases per City
avg_cases_city = data.groupby('city')['dengue_cases'].mean()
plt.figure(figsize=(6,4))
avg_cases_city.plot(kind='bar', color='green')
plt.xlabel("City (Encoded)")
plt.ylabel("Average Dengue Cases")
plt.title("Average Dengue Cases per City")
plt.grid(axis='y')
plt.show()
print("- Predictions are advisory, not medical diagnosis.")
print("- Helps allocate health resources efficiently to high-risk areas.")
