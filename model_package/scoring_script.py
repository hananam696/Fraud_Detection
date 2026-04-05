import pandas as pd
import pickle
import joblib

# Load artifacts
model = joblib.load("model_package/fraud_model.pkl")

with open("model_package/feature_names.pkl","rb") as f:
    features = pickle.load(f)

with open("model_package/threshold.pkl","rb") as f:
    threshold = pickle.load(f)

with open("model_package/encoders.pkl","rb") as f:
    encoders = pickle.load(f)


def preprocess_input(df):
    data = df.copy()

    # Ensure all required columns exist
    for col in features:
        if col not in data.columns:
            data[col] = 0

    # Apply encoding
    for col, le in encoders.items():
        if col in data.columns:
            data[col] = data[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    # Ensure correct order
    data = data[features]

    return data


def predict_fraud(df):
    data = preprocess_input(df)

    prob = model.predict_proba(data)[:,1]
    pred = (prob >= threshold).astype(int)

    return pd.DataFrame({
        "fraud_probability": prob,
        "prediction": pred
    })


if __name__ == "__main__":
    df = pd.read_csv("model_package/sample_data.csv")
    print(predict_fraud(df))
