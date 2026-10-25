import os
import numpy as np
import torch

WEIGHTS_PATH = "models/pretrained/alexnet-owt-7be5be79.pth"
OUTPUT_PATH = "models/pretrained/alexnet_conv_weights.npz"

if not os.path.isfile(WEIGHTS_PATH):
    raise FileNotFoundError(f"Pretrained weights not found: {WEIGHTS_PATH}")

weights = torch.load(
    WEIGHTS_PATH,
    map_location="cpu",
    weights_only=True,
)

# Map PyTorch AlexNet convolutional layers to TensorFlow kernel layout.
# PyTorch: (out_channels, in_channels, height, width)
# TensorFlow: (height, width, in_channels, out_channels)
layer_indices = [0, 3, 6, 8, 10]
converted = {}

for index in layer_indices:
    weight_key = f"features.{index}.weight"
    bias_key = f"features.{index}.bias"

    if weight_key not in weights or bias_key not in weights:
        raise KeyError(f"Missing expected AlexNet tensors for layer {index}")

    weight = weights[weight_key].detach().cpu().numpy()
    bias = weights[bias_key].detach().cpu().numpy()

    # OIHW -> HWIO
    converted[f"conv_{index}_kernel"] = np.transpose(weight, (2, 3, 1, 0))
    converted[f"conv_{index}_bias"] = bias

    print(
        f"Converted layer {index}: "
        f"{tuple(weight.shape)} -> "
        f"{converted[f'conv_{index}_kernel'].shape}"
    )

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
np.savez_compressed(OUTPUT_PATH, **converted)

# Reopen and validate the saved arrays.
with np.load(OUTPUT_PATH) as saved:
    expected_keys = 10
    if len(saved.files) != expected_keys:
        raise RuntimeError(
            f"Expected {expected_keys} arrays, found {len(saved.files)}"
        )

    for key in saved.files:
        if not np.isfinite(saved[key]).all():
            raise ValueError(f"Non-finite values found in {key}")

print("\nConversion successful.")
print(f"Saved to: {OUTPUT_PATH}")
print(f"Arrays saved: {len(converted)}")