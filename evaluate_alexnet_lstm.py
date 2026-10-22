import csv
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    accuracy_score,
)

from train_alexnet_lstm_train import VideoSequence

MANIFEST_PATH = "prepared_dataset/manifest.csv"
MODEL_PATH = "models/trained/best_alexnet_lstm.keras"
BATCH_SIZE = 2
THRESHOLD = 0.5


def main():
    # Load only the held-out test videos.
    with open(MANIFEST_PATH, "r", newline="", encoding="utf-8") as file:
        rows = [
            row for row in csv.DictReader(file)
            if row["split"] == "test"
        ]

    if not rows:
        raise ValueError("No videos found in the test split.")

    y_true = np.array([int(row["label"]) for row in rows])

    print(f"Test videos: {len(rows)}")
    print(f"FAKE videos: {np.sum(y_true == 0)}")
    print(f"REAL videos: {np.sum(y_true == 1)}")

    # Load the best model selected using validation data.
    print("\nLoading best checkpoint...")
    model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)

    # Load test frames in batches.
    test_data = VideoSequence(
        rows,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("\nPredicting test videos...")
    probabilities = model.predict(test_data, verbose=1).reshape(-1)

    if len(probabilities) != len(y_true):
        raise ValueError(
            f"Prediction count ({len(probabilities)}) "
            f"does not match test count ({len(y_true)})."
        )

    # Output is the probability of REAL: 1 = REAL, 0 = FAKE.
    y_pred = (probabilities >= THRESHOLD).astype(int)

    print("\n" + "=" * 55)
    print("HELD-OUT TEST RESULTS")
    print("=" * 55)

    print(f"\nThreshold: {THRESHOLD}")
    print("Prediction rule: probability >= threshold means REAL")

    print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")

    print("\nConfusion Matrix")
    print("Rows = actual [FAKE, REAL]")
    print("Columns = predicted [FAKE, REAL]")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print("\nPrecision, Recall, and F1-Score")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["FAKE", "REAL"],
            digits=4,
            zero_division=0,
        )
    )

    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, probabilities)
        print(f"ROC-AUC: {auc:.4f}")
    else:
        print("ROC-AUC cannot be calculated: test set has only one class.")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
