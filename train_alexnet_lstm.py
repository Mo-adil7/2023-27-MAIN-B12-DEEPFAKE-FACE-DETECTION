
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

tf.keras.utils.set_random_seed(42)

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as exc:
        print("GPU memory setting:", exc)

print("TensorFlow:", tf.__version__)
print("GPUs:", gpus)

WEIGHTS_PATH = "models/pretrained/alexnet_conv_weights.npz"
IMG_SIZE = 224
SEQUENCE_LENGTH = 8


def alexnet_frame_encoder():
    image = layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3), name="frame"
    )

    # Input: RGB pixels in the range 0–255.
    x = layers.Lambda(
        lambda image: (
            image / 255.0
            - tf.constant([0.485, 0.456, 0.406], dtype=image.dtype)
        ) / tf.constant([0.229, 0.224, 0.225], dtype=image.dtype),
        name="imagenet_preprocessing",
    )(image)

    # AlexNet convolution 1
    x = layers.ZeroPadding2D(2, name="pad_conv_0")(x)
    x = layers.Conv2D(
        64, 11, strides=4, padding="valid",
        activation="relu", name="conv_0"
    )(x)
    x = layers.MaxPooling2D(3, strides=2)(x)

    # AlexNet convolution 2
    x = layers.ZeroPadding2D(2, name="pad_conv_3")(x)
    x = layers.Conv2D(
        192, 5, padding="valid",
        activation="relu", name="conv_3"
    )(x)
    x = layers.MaxPooling2D(3, strides=2)(x)

    # AlexNet convolutions 3, 4 and 5
    x = layers.ZeroPadding2D(1, name="pad_conv_6")(x)
    x = layers.Conv2D(
        384, 3, padding="valid",
        activation="relu", name="conv_6"
    )(x)

    x = layers.ZeroPadding2D(1, name="pad_conv_8")(x)
    x = layers.Conv2D(
        256, 3, padding="valid",
        activation="relu", name="conv_8"
    )(x)

    x = layers.ZeroPadding2D(1, name="pad_conv_10")(x)
    x = layers.Conv2D(
        256, 3, padding="valid",
        activation="relu", name="conv_10"
    )(x)

    x = layers.MaxPooling2D(3, strides=2)(x)
    x = layers.GlobalAveragePooling2D(name="spatial_pool")(x)

    encoder = Model(image, x, name="alexnet_frame_encoder")

    if not os.path.isfile(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Converted AlexNet weights not found: {WEIGHTS_PATH}"
        )

    with np.load(WEIGHTS_PATH) as weights:
        for name in ["conv_0", "conv_3", "conv_6", "conv_8", "conv_10"]:
            encoder.get_layer(name).set_weights([
                weights[f"{name}_kernel"],
                weights[f"{name}_bias"],
            ])

    print("Loaded pretrained AlexNet convolution weights.")
    return encoder


def build_model():
    encoder = alexnet_frame_encoder()
    encoder.trainable = False

    video = layers.Input(
        shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3),
        name="video_frames",
    )

    x = layers.TimeDistributed(encoder, name="frame_features")(video)
    x = layers.LSTM(128, name="temporal_lstm")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    x = layers.Dense(64, activation="relu", name="dense_features")(x)

    output = layers.Dense(
        1, activation="sigmoid", name="fake_probability"
    )(x)

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

    # Architecture test only; no training happens here.
    test_batch = tf.zeros(
        (1, SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3),
        dtype=tf.float32,
    )
    prediction = model(test_batch, training=False)

    print("\nSmoke test passed.")
    print("Input shape:", test_batch.shape)
    print("Output shape:", prediction.shape)
    print("Example fake probability:", float(prediction[0, 0]))
    print("This probability is NOT an accuracy measurement.")
