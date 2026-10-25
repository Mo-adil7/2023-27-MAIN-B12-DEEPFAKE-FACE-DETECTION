import os
import csv
import io
import zipfile
from pathlib import Path

import h5py
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from train_alexnet_lstm_train import VideoSequence, build_model


PROJECT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_DIR / "prepared_dataset" / "manifest.csv"
MODEL_PATH = (
    PROJECT_DIR / "models" / "trained" / "best_alexnet_lstm.keras"
)

BATCH_SIZE = 2
THRESHOLD = 0.5


def restore_checkpoint_weights(model):
    """Restore all model weights from the inspected checkpoint layout."""
    encoder = model.get_layer("frame_features").layer

    conv_base = "layers/time_distributed/layer/layers"

    mapping = [
        (encoder.get_layer("conv_0"), f"{conv_base}/conv2d/vars"),
        (encoder.get_layer("conv_3"), f"{conv_base}/conv2d_1/vars"),
        (encoder.get_layer("conv_6"), f"{conv_base}/conv2d_2/vars"),
        (encoder.get_layer("conv_8"), f"{conv_base}/conv2d_3/vars"),
        (encoder.get_layer("conv_10"), f"{conv_base}/conv2d_4/vars"),
        (model.get_layer("temporal_lstm"), "layers/lstm/cell/vars"),
        (model.get_layer("dense"), "layers/dense/vars"),
        (model.get_layer("real_probability"), "layers/dense_1/vars"),
    ]

    restored = 0

    with zipfile.ZipFile(MODEL_PATH, "r") as archive:
        weight_bytes = archive.read("model.weights.h5")

    with h5py.File(io.BytesIO(weight_bytes), "r") as saved:
        for layer, group_path in mapping:
            if group_path not in saved:
                raise ValueError(f"Missing weight group: {group_path}")

            group = saved[group_path]
            keys = sorted(group.keys(), key=int)
            arrays = [group[key][()] for key in keys]
            expected = layer.get_weights()

            if len(arrays) != len(expected):
                raise ValueError(
                    f"{layer.name}: expected {len(expected)} arrays, "
                    f"found {len(arrays)}."
                )

            for index, (array, target) in enumerate(zip(arrays, expected)):
                if array.shape != target.shape:
                    raise ValueError(
                        f"{layer.name}, array {index}: "
                        f"saved shape {array.shape}, "
                        f"expected {target.shape}."
                    )

                if not np.isfinite(array).all():
                    raise ValueError(
                        f"{layer.name}: invalid values in saved weights."
                    )

            layer.set_weights(arrays)

            for actual, original in zip(layer.get_weights(), arrays):
                if not np.array_equal(actual, original):
                    raise ValueError(
                        f"Weight verification failed: {layer.name}"
                    )

            restored += len(arrays)
            print(f"Restored: {layer.name} ({len(arrays)} arrays)")

    if restored != len(model.weights):
        raise ValueError(
            f"Restored {restored} arrays, but the model "
            f"contains {len(model.weights)} weight arrays."
        )

    print(f"\nAll {restored} model weight arrays restored and verified.")


def main():
    os.chdir(PROJECT_DIR)

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest missing: {MANIFEST_PATH}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {MODEL_PATH}")

    with MANIFEST_PATH.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        rows = [
            row for row in csv.DictReader(file)
            if row["split"] == "test"
        ]

    if not rows:
        raise ValueError("No test videos found in the manifest.")

    y_true = np.array(
        [int(row["label"]) for row in rows],
        dtype=np.int32,
    )

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Expected labels: 0 = FAKE, 1 = REAL.")

    fake_count = int(np.sum(y_true == 0))
    real_count = int(np.sum(y_true == 1))

    print(f"\nTest videos: {len(rows)}")
    print(f"FAKE videos: {fake_count}")
    print(f"REAL videos: {real_count}")

    print("\nBuilding AlexNet + LSTM...")
    model = build_model()

    print("\nRestoring trained checkpoint weights...")
    restore_checkpoint_weights(model)

    test_data = VideoSequence(
        rows,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("\nPredicting test videos...")
    probabilities = np.asarray(
        model.predict(test_data, verbose=1)
    ).reshape(-1)

    if len(probabilities) != len(y_true):
        raise ValueError("Prediction count does not match test count.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Predictions contain invalid values.")

    # The trained model outputs P(REAL).
    y_pred = (probabilities >= THRESHOLD).astype(np.int32)

    majority_baseline = max(fake_count, real_count) / len(y_true)

    print("\n" + "=" * 55)
    print("HELD-OUT TEST RESULTS")
    print("=" * 55)

    print(f"\nThreshold: {THRESHOLD}")
    print("Rule: P(REAL) >= 0.5 means REAL")

    print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Majority-class baseline: {majority_baseline:.4f}")

    if fake_count > 0 and real_count > 0:
        print(
            "Balanced accuracy: "
            f"{balanced_accuracy_score(y_true, y_pred):.4f}"
        )

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

    if fake_count > 0 and real_count > 0:
        print(
            f"ROC-AUC: {roc_auc_score(y_true, probabilities):.4f}"
        )
    else:
        print("ROC-AUC unavailable: only one class in test split.")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()