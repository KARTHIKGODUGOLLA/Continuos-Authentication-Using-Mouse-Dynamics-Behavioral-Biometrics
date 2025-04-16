
# Mouse Dynamics Behavioral Biometrics (LSTM-Based)

This project implements an end-to-end system to model and authenticate users based on their mouse movement dynamics using LSTM and BiLSTM models. It is structured into three main stages: **Preprocessing**, **Training**, and **Evaluation**.

---

##  High-Level Pipeline Overview

1. **Preprocessing**
   - File: `mouse_preprocessor.ipynb`
   - Purpose: Convert raw mouse session data into segmented, padded sequences suitable for LSTM input.

2. **Training**
   - File: `binary_biLSTM.ipynb` or `train_binary_synth()`
   - Purpose: Train user-specific binary classifiers using BiLSTM to distinguish genuine users from imposters.

3. **Evaluation**
   - File: `test_binary_LSTM.ipynb`
   - Purpose: Evaluate each trained model using AUC, EER, and confusion matrix by grouping predictions.

---

##  Module Details

### 1️⃣ `mouse_preprocessor.ipynb`

**Purpose:**  
Converts raw CSVs of mouse data into structured NumPy arrays for deep learning.

**Key Functions:**
- `preprocessor()`: Loads raw mouse CSV files and merges them into a single DataFrame with User, Session, X, Y, Event, Timestamp.
- `perception_windows(data)`: Splits raw mouse movement into sequences at pauses ≥250ms. Computes delta features (ΔX, ΔY, ΔT).
- `filter_outliers(lens, X)`: Applies IQR filtering to keep only sequences within acceptable length bounds.
- `zero_pad(X, y)`: Pads sequences to uniform length for LSTM input.

**Key Data Structures:**
- `X_train`, `X_val`: 3D NumPy arrays of shape `(num_samples, time_steps, 3)`
- `y_train`, `y_val`: Corresponding user labels

---

### 2️⃣ `binary_biLSTM.ipynb`

**Purpose:**  
Train one binary classifier per user to distinguish their mouse behavior from that of others.

**Key Logic:**
- Load a `.npy` file for one user as positive class.
- Combine all other users’ data as negative class (imposters).
- Build and compile a `BiLSTM` model using `Keras`.

**Key Variables:**
- `gen_train_X`, `imp_train_X`: Genuine and imposter sequences
- `train_binary()` or `train_binary_synth()`: Main training loop
  - Supports oversampling of hard negatives or synthetic generation
- `model.fit(...)`: Trains the model and tracks validation AUC

**Model Architecture:**
- `Masking → BiLSTM(128) → BatchNorm → Dropout → BiLSTM(128) → Dense(1, sigmoid)`

---

### 3️⃣ `test_binary_LSTM.ipynb`

**Purpose:**  
Evaluate trained `.keras` models per user and generate metrics.

**Key Functions:**
- `train_test()`: Splits each user's `.npy` data into validation sets
- `grouping(y, n, outs)`: Groups `n` sequences to form one meta-decision per user
- `binary_test(splits, n, model_path)`: Loads each `.keras` model and evaluates it

**Key Metrics:**
- `AUC`: Area Under the ROC Curve
- `EER`: Equal Error Rate
- `Confusion Matrix`: Visual performance breakdown

---

##  Output

| File/Folder                    | Description                                  |
|-------------------------------|----------------------------------------------|
| `Gmail Data Splits/`          | Processed `.npy` sequence files per user     |
| `Gmail Models/`               | Trained `.keras` models per user             |
| `binary_results.csv`          | Evaluation metrics from `binary_test()`      |

---

##  Summary

This repository implements a fully functional mouse dynamics authentication system using deep learning. The code is modular, well-commented, and includes robust training and evaluation components. Ideal for behavioral biometrics research and practical authentication systems.


##  Project Structure

```
gmail-biometrics-capstone/
│
├── data/
│   ├── mouse/                         # Raw mouse data per user (CSV format)
│   └── data_splits/                  # Processed .npy sequences after preprocessing
│
├── models/                           # Directory to store trained BiLSTM models
│   ├── models_BiLSTM_Oversampling/   # Models trained using oversampled error cases (train_binary_over)
│   └── models_BiLSTM_Synth/          # Models trained using synthetic augmented data (train_binary_synth)
│
├── src/                              # Jupyter notebooks for core logic
│   ├── mouse_preprocessor.ipynb          # Data preprocessing: segmentation and padding
│   ├── binary_biLSTM.ipynb               # Model training: per-user binary BiLSTM (real + synth)
│   └── test_binary_LSTM.ipynb            # Evaluation: ROC, AUC, EER, confusion matrix
│
├── README.md                         # Project overview, setup, and structure
├── requirements.txt                  # List of required Python packages
└── LICENSE                           # Project license file
```
