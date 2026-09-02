import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Optional XGBoost
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False


def compare_models():

    # ==========================
    # Load Dataset
    # ==========================

    df = pd.read_csv("house_price_dataset_10000 (1).csv")

    # ==========================
    # Encode Categorical Columns
    # ==========================

    for col in df.select_dtypes(include="object").columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])

    # ==========================
    # Features & Target
    # ==========================

    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # ==========================
    # Models
    # ==========================

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    if xgb_available:
        models["XGBoost"] = XGBRegressor(
            random_state=42,
            verbosity=0
        )

    # ==========================
    # Compare Models
    # ==========================

    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)

        rmse = mean_squared_error(
            y_test,
            pred
        ) ** 0.5

        r2 = r2_score(
            y_test,
            pred
        )

        results.append({

            "Model": name,

            "MAE": round(mae,2),

            "RMSE": round(rmse,2),

            "R2 Score": round(r2,4)

        })

    results_df = pd.DataFrame(results)

    return results_df


# =====================================
# Run separately from terminal
# =====================================

if __name__ == "__main__":

    df = compare_models()

    print("\nModel Comparison\n")

    print(df)

    best = df.sort_values(
        "R2 Score",
        ascending=False
    ).iloc[0]

    print("\nBest Model :", best["Model"])

    print("R2 Score :", best["R2 Score"])