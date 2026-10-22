import os
import csv
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.utils.class_weight import compute_class_weight

# --------------------------------------------------
# Configuration
# --------------------------------------------------
tf.keras.utils.set_random_seed(42)

MANIFEST_PATH = "prepared_dataset/manifest.csv"
WEIGHTS_PATH = "models/pretrained/alexnet_conv_weights.npz"
OUTPUT_DIR = "models/trained"

IMG_SIZE = 224
SEQUENCE_LENGTH = 8
BATCH_SIZE = 2
EPOCHS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use GPU memory growth if a GPU is available.
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as exc:
        print("GPU memory setting:", exc)

print("TensorFlow:", tf.__version__)
print("GPUs:", gpus)


# --------------------------------------------------
# Load manifest
# --------------------------------------------------
def load_manifest():
    if not os.path.isfile(MANIFEST_PATH):
        raise FileNotFoundError(MANIFEST_PATH)

    rows = []

    with open(MANIFEST_PATH, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            row["label"] = int(row["label"])
            rows.append(row)

    required = {"train", "validation", "test"}
    available = {row["split"] for row in rows}

    if not required.issubset(available):
        raise ValueError(
            f"Missing dataset splits. Found: {sorted(available)}"
        )

    for row in rows:
        if row["label"] not in (0, 1):
            raise ValueError(
                f"Unexpected label for {row['video_id']}: {row['label']}"
            )

    print(f"Manifest videos: {len(rows)}")
    return rows


# --------------------------------------------------
# Load video frames in batches
# --------------------------------------------------
def load_video_frames(frames_dir):
    frames = []

    for index in range(SEQUENCE_LENGTH):
        path = os.path.join(
            frames_dir,
            f"frame_{index:02d}.png"
        )

        image = tf.keras.utils.load_img(
            path,
            target_size=(IMG_SIZE, IMG_SIZE),
            color_mode="rgb"
        )

        array = tf.keras.utils.img_to_array(image)
        frames.append(array)

    return np.stack(frames).astype(np.float32)


class VideoSequence(tf.keras.utils.Sequence):
    def __init__(self, rows, batch_size, shuffle=False):
        self.rows = list(rows)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.rows))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.rows) / self.batch_size)

    def __getitem__(self, index):
        batch_indices = self.indices[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        batch_rows = [self.rows[i] for i in batch_indices]

        x = np.empty(
            (
                len(batch_rows),
                SEQUENCE_LENGTH,
                IMG_SIZE,
                IMG_SIZE,
                3
            ),
            dtype=np.float32
        )

        y = np.empty((len(batch_rows),), dtype=np.float32)

        for i, row in enumerate(batch_rows):
            x[i] = load_video_frames(row["frames_dir"])
            y[i] = row["label"]

        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# --------------------------------------------------
# Pretrained AlexNet frame encoder
# --------------------------------------------------
def alexnet_frame_encoder():
    image = layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3),
        name="frame"
    )

    # Match ImageNet preprocessing used for pretrained weights.
    x = layers.Lambda(
        lambda img: (
            img / 255.0
            - tf.constant([0.485, 0.456, 0.406], dtype=img.dtype)
        ) / tf.constant([0.229, 0.224, 0.225], dtype=img.dtype),
        name="imagenet_preprocessing"
    )(image)

    x = layers.ZeroPadding2D(2)(x)
    x = layers.Conv2D(
        64, 11, strides=4,
        padding="valid", activation="relu",
        name="conv_0"
    )(x)
    x = layers.MaxPooling2D(3, strides=2)(x)

    x = layers.ZeroPadding2D(2)(x)
    x = layers.Conv2D(
        192, 5, padding="valid",
        activation="relu", name="conv_3"
    )(x)
    x = layers.MaxPooling2D(3, strides=2)(x)

    x = layers.ZeroPadding2D(1)(x)
    x = layers.Conv2D(
        384, 3, padding="valid",
        activation="relu", name="conv_6"
    )(x)

    x = layers.ZeroPadding2D(1)(x)
    x = layers.Conv2D(
        256, 3, padding="valid",
        activation="relu", name="conv_8"
    )(x)

    x = layers.ZeroPadding2D(1)(x)
    x = layers.Conv2D(
        256, 3, padding="valid",
        activation="relu", name="conv_10"
    )(x)

    x = layers.MaxPooling2D(3, strides=2)(x)
    x = layers.GlobalAveragePooling2D(name="spatial_pool")(x)

    encoder = Model(image, x, name="alexnet_frame_encoder")

    if not os.path.isfile(WEIGHTS_PATH):
        raise FileNotFoundError(WEIGHTS_PATH)

    with np.load(WEIGHTS_PATH) as weights:
        for name in [
            "conv_0", "conv_3", "conv_6", "conv_8", "conv_10"
        ]:
            encoder.get_layer(name).set_weights([
                weights[f"{name}_kernel"],
                weights[f"{name}_bias"]
            ])

    encoder.trainable = False
    print("Loaded pretrained AlexNet convolution weights.")

    return encoder


# --------------------------------------------------
# AlexNet + LSTM classifier
# --------------------------------------------------
def build_model():
    encoder = alexnet_frame_encoder()

    video = layers.Input(
        shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3),
        name="video_frames"
    )

    x = layers.TimeDistributed(
        encoder,
        name="frame_features"
    )(video)

    x = layers.LSTM(128, name="temporal_lstm")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)

    # Manifest labels: 0 = FAKE, 1 = REAL.
    # Therefore this output represents P(REAL).
    output = layers.Dense(
        1,
        activation="sigmoid",
        name="real_probability"
    )(x)

    model = Model(video, output, name="alexnet_lstm_deepfake")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return model


# --------------------------------------------------
# Training
# --------------------------------------------------
def main():
    rows = load_manifest()

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validation"]

    # Do not load or evaluate the test split during training.
    train_labels = np.array(
        [row["label"] for row in train_rows],
        dtype=np.int32
    )

    classes = np.unique(train_labels)

    if len(classes) != 2:
        raise ValueError(
            f"Training data must contain both classes; found {classes}"
        )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_labels
    )

    class_weights = {
        int(label): float(weight)
        for label, weight in zip(classes, weights)
    }

    print("Training videos:", len(train_rows))
    print("Validation videos:", len(val_rows))
    print("Class weights:", class_weights)

    train_data = VideoSequence(
        train_rows,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_data = VideoSequence(
        val_rows,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = build_model()
    model.summary()

    checkpoint_path = os.path.join(
        OUTPUT_DIR,
        "best_alexnet_lstm.keras"
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(OUTPUT_DIR, "training_log.csv")
        ),
        tf.keras.callbacks.TerminateOnNaN()
    ]

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    final_path = os.path.join(
        OUTPUT_DIR,
        "final_alexnet_lstm.keras"
    )

    model.save(final_path)

    print("\nTraining finished.")
    print("Best checkpoint:", checkpoint_path)
    print("Final model:", final_path)
    print(
        "The held-out test split was not used for training "
        "or checkpoint selection."
    )


if __name__ == "__main__":
    main()
