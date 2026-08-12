"""
Football Match Outcome Prediction
----------------------------------
Generate a prediction using the trained neural network
and a predefined match example.
"""

from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

MODELS_DIR = ROOT_DIR / "models"

MODEL_PATH = MODELS_DIR / "modelo.keras"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"


# ============================================================
# SAMPLE MATCH
# ============================================================

SAMPLE_MATCH = {
    "L": 3,
    "X": 1,
    "V": 0,
    "L_3": 2,
    "X_3": 1,
    "V_3": 0,
    "Porcentaje 1": 53,
    "Porcentaje 2": 20,
    "Rank1": 20,
    "Rank2": 19,
    "Tendencia_L": 45,
    "Tendencia_V": 20,
}


# ============================================================
# LOAD MODEL AND PREPROCESSING ARTIFACTS
# ============================================================

def load_artifacts():
    """Load the trained model, scaler and label encoder."""

    print("Loading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    label_encoder = joblib.load(
        ENCODER_PATH
    )

    return model, scaler, label_encoder


# ============================================================
# BUILD INPUT
# ============================================================

def build_input_data(feature_names):
    """
    Build the input vector using the predefined
    sample match values.
    """

    missing_features = [
        feature
        for feature in feature_names
        if feature not in SAMPLE_MATCH
    ]

    if missing_features:
        raise ValueError(
            "Missing values for features: "
            f"{missing_features}"
        )

    values = [
        SAMPLE_MATCH[feature]
        for feature in feature_names
    ]

    return np.array(
        values,
        dtype=float
    ).reshape(1, -1)


# ============================================================
# PREDICTION
# ============================================================

def predict_match(
    model,
    scaler,
    label_encoder,
    input_data,
):
    """Generate prediction probabilities."""

    input_scaled = scaler.transform(
        input_data
    )

    probabilities = model.predict(
        input_scaled,
        verbose=0,
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    return probabilities, predicted_class


# ============================================================
# DISPLAY
# ============================================================

def display_prediction(
    probabilities,
    predicted_class,
    label_encoder,
):
    """Display input features and prediction."""

    print("\n======================================")
    print("FOOTBALL MATCH OUTCOME PREDICTION")
    print("======================================")

    print("\nInput features:")

    for feature, value in SAMPLE_MATCH.items():
        print(
            f"  {feature}: {value}"
        )

    print("\nPrediction probabilities:")

    for class_name, probability in zip(
        label_encoder.classes_,
        probabilities,
    ):
        print(
            f"  {class_name}: "
            f"{probability * 100:.2f}%"
        )

    print(
        f"\nPredicted outcome: "
        f"{predicted_class}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    model, scaler, label_encoder = load_artifacts()

    feature_names = list(
        scaler.feature_names_in_
    )

    input_data = build_input_data(
        feature_names
    )

    probabilities, predicted_class = predict_match(
        model,
        scaler,
        label_encoder,
        input_data,
    )

    display_prediction(
        probabilities,
        predicted_class,
        label_encoder,
    )


if __name__ == "__main__":
    main()