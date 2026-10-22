# Deepfake Face Detection Using AlexNet and LSTM

## Project Overview

This is an educational deepfake video detection project. It uses an AlexNet-based feature extractor and an LSTM network to classify videos as REAL or FAKE using eight extracted video frames.

The project includes a Flask web application, dataset preparation scripts, training code, and evaluation tools.

## Model Architecture

- Feature extractor: AlexNet convolutional layers with pretrained weights.
- Temporal model: Long Short-Term Memory (LSTM).
- Classifier: Dense layers with a sigmoid output.
- Input: Eight RGB frames resized to 224 x 224 pixels.
- Output: Probability that the video is REAL.

## Dataset

The dataset contains videos organized into REAL and FAKE source folders. Videos were split into training, validation, and test sets at the video level.

The source videos and extracted frames are not included in this repository.

Labels were inferred from folder names and were not independently verified.

## Test Results

The selected model was evaluated on 902 held-out test videos.

| Metric | Result |
|---|---:|
| Accuracy | 74.50% |
| ROC-AUC | 0.6689 |
| FAKE precision | 91.24% |
| FAKE recall | 78.62% |
| FAKE F1-score | 84.46% |
| REAL precision | 21.66% |
| REAL recall | 43.93% |
| REAL F1-score | 29.01% |

### Confusion Matrix

Rows represent actual classes; columns represent predicted classes.

| | Predicted FAKE | Predicted REAL |
|---|---:|---:|
| Actual FAKE | 625 | 170 |
| Actual REAL | 60 | 47 |

The current model performs better at identifying FAKE videos than REAL videos. It is an experimental prototype and should not be treated as a reliable real-world deepfake verification system.

See `results/test_evaluation.txt` for the evaluation report.

## Project Structure

- `app/` - Flask application and HTML templates.
- `models/pretrained/` - Converted pretrained AlexNet weights.
- `models/trained/` - Trained model checkpoints and training log.
- `results/` - Test evaluation report.
- `prepare_dataset.py` - Dataset preparation.
- `train_alexnet_lstm_train.py` - Model training.
- `evaluate_alexnet_lstm.py` - Test evaluation.
- `convert_alexnet_weights.py` - Pretrained weight conversion.
- `requirements.txt` - Python dependencies.

## Setup

Use Python 3.11 and create a virtual environment.

On Linux or WSL:

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt

The environment may require compatible TensorFlow and GPU driver versions. The model was developed with TensorFlow 2.15.1.

## Training and Evaluation

After arranging the dataset in the folder structure expected by the preparation script, run:

    python prepare_dataset.py
    python train_alexnet_lstm_train.py
    python evaluate_alexnet_lstm.py

Training and evaluation require the dataset and converted pretrained AlexNet weights.

## Limitations

- The model has limited discrimination between REAL and FAKE videos.
- The dataset is imbalanced.
- Labels were inferred from folder names.
- The split does not guarantee separation of identities or sources across subsets.
- Performance on independent datasets has not been established.

## Future Improvements

- Improve performance on REAL videos.
- Use identity-aware and source-aware dataset splits.
- Evaluate on independent datasets.
- Study classification thresholds using validation data.
- Improve the Flask application's security and usability.

## Disclaimer

This project is intended for educational and research purposes. Predictions are not definitive proof that a video is authentic or manipulated.
