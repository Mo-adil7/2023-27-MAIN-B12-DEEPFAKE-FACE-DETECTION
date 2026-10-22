import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# Reproducibility
tf.keras.utils.set_random_seed(42)

# GPU memory management
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as exc:
        print("GPU memory setting:", exc)

print("TensorFlow:", tf.__version__)
print("GPUs:", gpus)

WEIGHTS_PATH = "models/pretrained/alexnet_conv_weights.npz"
IMG_SIZE = 227
SEQUENCE_LENGTH = 8
NUM_CLASSES = 1


def alexnet_frame_encoder():
    """Build an AlexNet-style frame encoder and load converted convolution weights."""
    image = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="frame")

    x = layers.Conv2D(
        64, 11, strides=4, padding="valid", activation="relu", name="conv_0"
    )(image)
    x = layers.MaxPooling2D(3, strides=2, padding="valid")(x)

    x = layers.Conv2D(
        192, 5, padding="same", activation="relu", name="conv_3"
    )(x)
    x = layers.MaxPooling2D(3, strides=2, padding="valid")(x)

    x = layers.Conv2D(
        384, 3, padding="same", activation="relu", name="conv_6"
    )(x)
    x = layers.Conv2D(
        256, 3, padding="same", activation="relu", name="conv_8"
    )(x)
    x = layers.Conv2D(
        256, 3, padding="same", activation="relu", name="conv_10"
    )(x)
    x = layers.MaxPooling2D(3, strides=2, padding="valid")(x)

    # Compact spatial representation for the LSTM.
    x = layers.GlobalAveragePooling2D(name="spatial_pool")(x)
    encoder = Model(image, x, name="alexnet_frame_encoder")

    if not os.path.isfile(WEIGHTS_PATH):
        raise FileNotFoundError(f"Converted weights not found: {WEIGHTS_PATH}")

    with np.load(WEIGHTS_PATH) as weights:
        mapping = {
            "conv_0": "conv_0",
            "conv_3": "conv_3",
            "conv_6": "conv_6",
            "conv_8": "conv_8",
            "conv_10": "conv_10",
        }

        for keras_name, source_name in mapping.items():
            kernel = weights[f"{source_name}_kernel"]
            bias = weights[f"{source_name}_bias"]
            encoder.get_layer(keras_name).set_weights([kernel, bias])

    print("Loaded pretrained weights for five convolutional layers.")
    return encoder


def build_model():
    encoder = alexnet_frame_encoder()

    # Initially freeze AlexNet to train the temporal classifier more safely.
    encoder.trainable = False

    video = layers.Input(
        shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3), name="video_frames"
    )

    # Apply the shared AlexNet encoder to every frame.
    x = layers.TimeDistributed(encoder, name="frame_features")(video)
    x = layers.LSTM(128, name="temporal_lstm")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    output = layers.Dense(NUM_CLASSES, activation="sigmoid", name="fake_probability")(x)

    model = Model(video, output, name="alexnet_lstm_deepfake")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()

    # Synthetic smoke test; this does not train on real videos.
    test_batch = tf.zeros(
        (1, SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32
    )
    prediction = model(test_batch, training=False)

    print("Smoke test passed.")
    print("Input shape:", test_batch.shape)
    print("Output shape:", prediction.shape)
    print("Example fake probability:", float(prediction[0, 0]))
