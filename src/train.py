
"""
Football Match Outcome Prediction
----------------------------------
Training pipeline for a multiclass neural network
that predicts football match outcomes.

Target classes:
    1 = Home win
    X = Draw
    2 = Away win
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "dataset.csv"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"


# ============================================================
# DATA
# ============================================================

def load_dataset():
    """Load and clean the dataset."""

    print("\nLoading dataset...")

    dataset = pd.read_csv(DATA_PATH)

    print(f"Dataset path: {DATA_PATH}")
    print(f"Rows before cleaning: {dataset.shape[0]}")

    # Remove automatically generated unnamed columns
    dataset = dataset.loc[
        :,
        ~dataset.columns.str.contains("^Unnamed")
    ]

    # Remove rows containing missing values
    dataset = dataset.dropna()

    print(f"Rows after cleaning: {dataset.shape[0]}")

    return dataset


def prepare_data(dataset):
    """Separate features and target and encode target classes."""

    X = dataset.iloc[:, :-1]
    y_raw = dataset.iloc[:, -1]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw.astype(str))

    print("\n======================================")
    print("DATA PREPARATION")
    print("======================================")

    print("\nFeatures:")
    print(X.columns.tolist())

    print("\nTarget:")
    print(y_raw.name)

    print("\nTarget classes:")
    print(label_encoder.classes_)

    print("\nClass mapping:")
    print(
        dict(
            zip(
                label_encoder.classes_,
                range(len(label_encoder.classes_))
            )
        )
    )

    return X, y, label_encoder


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_data(X, y):
    """Split data into training, validation and test sets."""

    train_size = 0.80
    validation_size = 0.10
    test_size = 0.10

    # First split: 90% temporary data / 10% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=0,
    )

    # Second split: 80% train / 10% validation
    validation_relative_size = (
        validation_size / (train_size + validation_size)
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=validation_relative_size,
        stratify=y_temp,
        random_state=0,
    )

    print("\n======================================")
    print("DATA SPLIT")
    print("======================================")

    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


# ============================================================
# FEATURE SCALING
# ============================================================

def scale_data(X_train, X_val, X_test):
    """Standardize features using statistics from training data."""

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler,
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(y_train):
    """Calculate balanced class weights."""

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    class_weight_dict = dict(
        zip(classes, weights)
    )

    print("\n======================================")
    print("CLASS WEIGHTS")
    print("======================================")

    print(class_weight_dict)

    return class_weight_dict


# ============================================================
# MODEL
# ============================================================

def build_model(input_size, n_classes):
    """Build and compile the neural network."""

    model = Sequential(
        [
            Input(shape=(input_size,)),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(n_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    class_weight_dict,
):
    """Train the neural network."""

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=1,
    )

    return history


# ============================================================
# TRAINING HISTORY
# ============================================================

def save_training_history(history):
    """Save training and validation loss/accuracy curves."""

    RESULTS_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)

    plt.plot(
        history.history["loss"],
        label="Train",
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation",
    )

    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)

    plt.plot(
        history.history["accuracy"],
        label="Train",
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation",
    )

    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()

    output_path = RESULTS_DIR / "training_history.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"\nTraining history saved to: {output_path}")


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    label_encoder,
):
    """Evaluate the trained model on the test set."""

    y_test_proba = model.predict(
        X_test,
        verbose=0,
    )

    y_test_pred = np.argmax(
        y_test_proba,
        axis=1,
    )

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        y_test_pred,
    )

    # F1 macro
    f1_macro = f1_score(
        y_test,
        y_test_pred,
        average="macro",
    )

    # Binarize target for multiclass ROC-AUC / AP
    y_test_bin = label_binarize(
        y_test,
        classes=range(
            len(label_encoder.classes_)
        ),
    )

    # ROC-AUC
    roc_auc_macro = roc_auc_score(
        y_test_bin,
        y_test_proba,
        multi_class="ovr",
        average="macro",
    )

    # Average Precision
    average_precision_macro = average_precision_score(
        y_test_bin,
        y_test_proba,
        average="macro",
    )

    # Classification report
    report = classification_report(
        y_test,
        y_test_pred,
        target_names=[
            str(c)
            for c in label_encoder.classes_
        ],
    )

    print("\n======================================")
    print("TEST SET EVALUATION")
    print("======================================")

    print("\nClassification report:")
    print(report)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 macro: {f1_macro:.4f}")
    print(f"ROC AUC (OVR, macro): {roc_auc_macro:.4f}")
    print(
        "Average Precision (macro): "
        f"{average_precision_macro:.4f}"
    )

    metrics = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "roc_auc_macro": float(roc_auc_macro),
        "average_precision_macro": float(
            average_precision_macro
        ),
    }

    return metrics, report


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(metrics, classification_report_text):
    """Save evaluation results to the results directory."""

    RESULTS_DIR.mkdir(exist_ok=True)

    # Save numerical metrics
    metrics_path = RESULTS_DIR / "metrics.json"

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    # Save classification report
    report_path = RESULTS_DIR / "classification_report.txt"

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(classification_report_text)

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Classification report saved to: {report_path}")


# ============================================================
# SAVE MODEL ARTIFACTS
# ============================================================

def save_artifacts(
    model,
    scaler,
    label_encoder,
):
    """Save trained model, scaler and label encoder."""

    MODELS_DIR.mkdir(exist_ok=True)

    model_path = MODELS_DIR / "modelo.keras"
    scaler_path = MODELS_DIR / "scaler.pkl"
    encoder_path = MODELS_DIR / "label_encoder.pkl"

    model.save(model_path)

    joblib.dump(
        scaler,
        scaler_path,
    )

    joblib.dump(
        label_encoder,
        encoder_path,
    )

    print("\n======================================")
    print("MODEL ARTIFACTS")
    print("======================================")

    print(f"Model:         {model_path}")
    print(f"Scaler:        {scaler_path}")
    print(f"Label encoder: {encoder_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    # 1. Load dataset
    dataset = load_dataset()

    # 2. Prepare features and target
    X, y, label_encoder = prepare_data(
        dataset
    )

    # 3. Train / validation / test split
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(
        X,
        y,
    )

    # 4. Scale features
    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler,
    ) = scale_data(
        X_train,
        X_val,
        X_test,
    )

    # 5. Calculate class weights
    class_weight_dict = calculate_class_weights(
        y_train
    )

    # 6. Build model
    model = build_model(
        input_size=X_train_scaled.shape[1],
        n_classes=len(
            label_encoder.classes_
        ),
    )

    print("\n======================================")
    print("MODEL ARCHITECTURE")
    print("======================================")

    model.summary()

    # 7. Train model
    history = train_model(
        model,
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        class_weight_dict,
    )

    # 8. Save training history
    save_training_history(history)

    # 9. Evaluate model
    metrics, classification_report_text = evaluate_model(
        model,
        X_test_scaled,
        y_test,
        label_encoder,
    )

    # 10. Save evaluation results
    save_results(
        metrics,
        classification_report_text,
    )

    # 11. Save trained model and preprocessing artifacts
    save_artifacts(
        model,
        scaler,
        label_encoder,
    )

    print("\n======================================")
    print("TRAINING PIPELINE COMPLETED")
    print("======================================")


if __name__ == "__main__":
    main()

