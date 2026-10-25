"""Inference for the project's inspected AlexNet + LSTM checkpoint."""
import io
import threading
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "models" / "trained" / "best_alexnet_lstm.keras"
_model = None
_lock = threading.Lock()


def build_model():
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    class ImageNetPreprocessing(layers.Layer):
        def call(self, image):
            mean = tf.constant([0.485, 0.456, 0.406], dtype=image.dtype)
            std = tf.constant([0.229, 0.224, 0.225], dtype=image.dtype)
            return (image / 255.0 - mean) / std

    image = layers.Input((224, 224, 3), name="frame")
    x = ImageNetPreprocessing(name="imagenet_preprocessing")(image)
    for name, filters, kernel, stride, pad, pool in [
        ("conv_0", 64, 11, 4, 2, True),
        ("conv_3", 192, 5, 1, 2, True),
        ("conv_6", 384, 3, 1, 1, False),
        ("conv_8", 256, 3, 1, 1, False),
        ("conv_10", 256, 3, 1, 1, True),
    ]:
        x = layers.ZeroPadding2D(pad)(x)
        x = layers.Conv2D(filters, kernel, strides=stride,
                          activation="relu", name=name)(x)
        if pool:
            x = layers.MaxPooling2D(3, strides=2)(x)
    x = layers.GlobalAveragePooling2D(name="spatial_pool")(x)
    encoder = Model(image, x, name="alexnet_frame_encoder")
    encoder.trainable = False
    video = layers.Input((8, 224, 224, 3), name="video_frames")
    x = layers.TimeDistributed(encoder, name="frame_features")(video)
    x = layers.LSTM(128, name="temporal_lstm")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    x = layers.Dense(64, activation="relu", name="dense")(x)
    output = layers.Dense(1, activation="sigmoid", name="real_probability")(x)
    return Model(video, output, name="alexnet_lstm_deepfake")


def restore_weights(model, checkpoint=None):
    """Load every model array explicitly; fail on missing/mismatched weights."""
    import h5py
    checkpoint = CHECKPOINT if checkpoint is None else Path(checkpoint)
    encoder = model.get_layer("frame_features").layer
    base = "layers/time_distributed/layer/layers"
    mapping = [(encoder.get_layer(name), f"{base}/{stored}/vars")
               for name, stored in zip(
                   ["conv_0", "conv_3", "conv_6", "conv_8", "conv_10"],
                   ["conv2d", "conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4"])]
    mapping += [
        (model.get_layer("temporal_lstm"), "layers/lstm/cell/vars"),
        (model.get_layer("dense"), "layers/dense/vars"),
        (model.get_layer("real_probability"), "layers/dense_1/vars"),
    ]
    loaded = 0
    with zipfile.ZipFile(checkpoint) as archive:
        data = archive.read("model.weights.h5")
    with h5py.File(io.BytesIO(data), "r") as saved:
        for layer, path in mapping:
            group = saved[path]
            arrays = [group[k][()] for k in sorted(group.keys(), key=int)]
            expected = layer.get_weights()
            if len(arrays) != len(expected):
                raise ValueError(f"Checkpoint array count mismatch: {layer.name}")
            for a, b in zip(arrays, expected):
                if a.shape != b.shape or not np.isfinite(a).all():
                    raise ValueError(f"Invalid checkpoint weights: {layer.name}")
            layer.set_weights(arrays)
            if not all(np.array_equal(a, b) for a, b in zip(arrays, layer.get_weights())):
                raise ValueError(f"Checkpoint verification failed: {layer.name}")
            loaded += len(arrays)
    if loaded != len(model.weights):
        raise ValueError("Some model weights were not restored.")
    return loaded


def prepare_video(path, output_dir):
    """Match prepare_dataset.py: same indices, resize, centre crop and RGB."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("This video could not be opened. Try another video or an MP4 copy.")
        count_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not np.isfinite(count_raw) or count_raw < 1:
            raise ValueError("The video has no readable frame count.")
        count = int(count_raw)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps = fps if np.isfinite(fps) and fps > 0 else None
        indices = [round(i * (count - 1) / 7) for i in range(8)]
        frames, previews = [], []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        width = height = None
        for number, index in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"Could not read sampled frame {index}. Try a different video.")
            height, width = frame.shape[:2]
            scale = 256 / min(height, width)
            w, h = round(width * scale), round(height * scale)
            resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            left, top = (w - 224) // 2, (h - 224) // 2
            crop = resized[top:top + 224, left:left + 224]
            if crop.shape != (224, 224, 3):
                raise ValueError("Unable to prepare a complete video frame.")
            frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            name = f"frame_{number:02d}.png"
            if not cv2.imwrite(str(output_dir / name), crop):
                raise OSError("Could not save the sampled frame preview.")
            previews.append({"filename": name, "index": index,
                             "time": round(index / fps, 2) if fps else None})
        metadata = {"frame_count": count, "fps": round(fps, 2) if fps else None,
                    "duration": round(count / fps, 2) if fps else None,
                    "width": width, "height": height}
        return np.stack(frames).astype(np.float32), metadata, previews
    finally:
        cap.release()


def predict_video(path, output_dir):
    global _model
    frames, metadata, previews = prepare_video(path, output_dir)
    with _lock:
        if _model is None:
            candidate = build_model()
            restored = restore_weights(candidate)
            _model = candidate
            print(f"Classifier ready: {restored} weight arrays verified.")
        probability = float(np.asarray(_model(frames[None], training=False)).reshape(-1)[0])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("The classifier returned an invalid score.")
    return {"prediction": "REAL" if probability >= 0.5 else "FAKE",
            "real_score": round(probability * 100, 2),
            "fake_score": round((1 - probability) * 100, 2),
            "threshold": 0.5, "metadata": metadata, "frames": previews}
