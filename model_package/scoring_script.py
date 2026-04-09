import pandas as pd
import pickle
import joblib

# Load artifacts
model = joblib.load("model_package/fraud_automl_model.pkl")

with open("model_package/columns.pkl", "rb") as f:
    features = pickle.load(f)

with open("model_package/threshold.pkl", "rb") as f:
    threshold = pickle.load(f)

with open("model_package/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

categorical_cols = [
    'use_chip',
    'merchant_state',
    'card_brand',
    'card_type',
    'gender',
    'age_group'
]

def preprocess_input(df):
    data = df.copy()

    for col in features:
        if col not in data.columns:
            data[col] = "unknown"

    data = data[features]

    data[categorical_cols] = encoder.transform(
        data[categorical_cols].astype(str)
    )

    return data


def predict_fraud(df):
    data = preprocess_input(df)

    prob = model.predict_proba(data)[:, 1]
    pred = (prob >= threshold).astype(int)

    return pd.DataFrame({
        "fraud_probability": prob,
        "prediction": pred
    })


if __name__ == "__main__":
    df = pd.read_csv("model_package/sample_data.csv")
    print(predict_fraud(df))
