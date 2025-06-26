# EmoTrack: Real-Time Emotion Detection

**EmoTrack** is a deep learning-based project for real-time facial emotion recognition using the FER2013 dataset and MobileNetV2. It trains a model to classify seven emotions (**Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise**) and performs webcam inference with emotion logging and visualization.

## Features

- **Dataset:** Uses FER2013 dataset (~28,709 train, ~7,178 test images) with 48x48 grayscale images.
- **Model:** MobileNetV2 with ImageNet weights, custom head (Dense 128 → 64 → 7), achieving ~41% test accuracy.
- **Preprocessing:** Converts grayscale to RGB, resizes to 224x224, normalizes with `MobileNetV2`'s `preprocess_input`.
- **Augmentation:** Applies rotation, zoom, flip, and shifts to training data.
- **Training:** Uses Adam optimizer, class weights for imbalance, early stopping, and learning rate scheduling.
- **Evaluation:** Generates confusion matrix and classification report for test set.
- **Webcam Inference:** Detects faces using Haar Cascade, overlays emotion labels, and logs predictions.
- **Visualization:** Plots training curves, confusion matrix, and emotion frequency from logs.

## Requirements

- Python 3.8+
- Libraries:
  - `tensorflow`
  - `opencv-python`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
- FER2013 dataset in `./dataset/train` and `./dataset/test` with subfolders for each emotion
- Webcam for real-time inference

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/povsalman/EmoTrack-Emotion-Detection-Using-AI.git
   cd emotrack
   ```

2. Install dependencies:

   ```bash
   pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn
   ```

3. Download FER2013 dataset and place in `./dataset/` with structure:

   ```
   dataset/
   ├── train/
   │   ├── Angry/
   │   ├── Disgust/
   │   └── ...
   └── test/
       ├── Angry/
       ├── Disgust/
       └── ...
   ```

## Usage

1. Open the Jupyter Notebook:

   ```bash
   jupyter notebook base_code.ipynb
   ```

2. Run cells sequentially:

   - Install/import dependencies
   - Load/visualize dataset
   - Preprocess data
   - Create train/validate/test generators (70/30 train split)
   - Build MobileNetV2 model
   - Train model (12 epochs, saves `base_best_emotion_model.h5`)
   - Evaluate model (prints metrics, plots confusion matrix)
   - Run webcam inference (press `'q'` to exit, saves logs in `./logs/`)
   - Visualize emotion frequency from logs

3. **Outputs:**
   - Model: `./base_best_emotion_model.h5`
   - Logs: `./logs/emotion_log_*.csv`
   - Plots: Displayed in notebook

## File Structure

- `base_code.ipynb`: Main Jupyter Notebook with all steps
- `dataset/`: FER2013 dataset (not included, user-provided)
- `logs/`: Emotion prediction logs (generated during inference)
- `base_best_emotion_model.h5`: Trained model (generated after training)

## Notes

- Training may take time; use a GPU for faster results.
- Ensure OpenCV’s Haar Cascade (`haarcascade_frontalface_default.xml`) is accessible.
- Accuracy depends on dataset quality and hardware; fine-tuning (optional) can boost performance.
- Another version of code is also provide named `improved_code.ipynb` with model `improved_best_emotion_model.h5`
