
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(10)
n = 150

area = np.random.randint(500, 3000, n)
bedrooms = np.random.randint(1, 5, n)
age = np.random.randint(0, 30, n)
distance = np.round(np.random.uniform(1, 25, n), 1)

price = (
    15 + 0.045 * area + 4 * bedrooms
    - 0.6 * age - 1.2 * distance
    + np.random.normal(0, 6, n)
)

houses = pd.DataFrame({
    "area_sqft": area,
    "bedrooms": bedrooms,
    "age_years": age,
    "distance_km": distance,
    "price_lakhs": price.round(2)
})

print("House Price Prediction using Linear Regression")
print("=" * 50)
print(houses.head().to_string(index=False))

print("\nCorrelation with price:")
print(houses.corr()["price_lakhs"].drop("price_lakhs").round(3))

X = houses.drop(columns="price_lakhs")
y = houses["price_lakhs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Simple Linear Regression
simple = LinearRegression().fit(X_train[["area_sqft"]], y_train)

print("\nSimple Linear Regression")
print("Intercept =", simple.intercept_)
print("Slope =", simple.coef_[0])
print("R2 =", r2_score(
    y_test, simple.predict(X_test[["area_sqft"]])
))

# Multiple Linear Regression
model = LinearRegression().fit(X_train, y_train)

print("\nMultiple Linear Regression")
print("Intercept =", model.intercept_)

for name, c in zip(X.columns, model.coef_):
    print(name, "=", c)

# Model Evaluation
pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)

print("\nModel Evaluation")
print("MAE =", mean_absolute_error(y_test, pred))
print("MSE =", mse)
print("RMSE =", np.sqrt(mse))
print("R2 =", r2_score(y_test, pred))

# New House Prediction
new_house = pd.DataFrame(
    [[1800, 3, 5, 8.0]], columns=X.columns
)

print("\nPredicted House Price:")
print(model.predict(new_house)[0].round(2), "lakhs")

# Visualization
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

xs = np.linspace(500, 3000, 100).reshape(-1, 1)

ax[0].scatter(X_train["area_sqft"], y_train, s=15)
ax[0].plot(
    xs,
    simple.predict(pd.DataFrame(xs, columns=["area_sqft"])),
    color="red"
)
ax[0].set_xlabel("Area (sqft)")
ax[0].set_ylabel("Price (lakhs)")
ax[0].set_title("Simple Linear Regression")

ax[1].scatter(pred, y_test - pred, color="purple")
ax[1].axhline(0, color="black", linestyle="--")
ax[1].set_xlabel("Predicted Price")
ax[1].set_ylabel("Residual")
ax[1].set_title("Residual Plot")

plt.tight_layout()
plt.savefig("exp05_regression.png", dpi=150)
plt.show()
