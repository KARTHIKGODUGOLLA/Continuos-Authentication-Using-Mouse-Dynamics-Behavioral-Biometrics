
# Mouse Dynamics Behavioral Biometrics (LSTM-Based)

This project implements an end-to-end system to model and authenticate users based on their mouse movement dynamics using LSTM and BiLSTM models. It is structured into three main stages: **Preprocessing**, **Training**, and **Evaluation**.

## How to Run This Project

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare raw data**  
   Place user mouse session CSVs in `data/mouse/`.

3. **Preprocess data**  
   Run `mouse_preprocessor.ipynb` to generate `.npy` sequences in `data/data_splits/`.

4. **Train models**  
   Use `binary_biLSTM.ipynb` to train models using one of:
   - `train_binary()`
   - `train_binary_over()`
   - `train_binary_synth()`

5. **Evaluate**  
   Run `test_binary_LSTM.ipynb` to evaluate trained models with AUC, EER, and confusion matrices.

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

**Key Functions:**
***Standard Training***
- Function: `train_binary()`
- Trains the model using 80/20 split of genuine and imposter sequences.
- Uses early stopping and validation AUC to save the best-performing model.
***Oversampling Training***
- Function: `train_binary_over()`
- After each training round, misclassified sequences (false positives/negatives) are added back to the training set.
- Useful in this project to emphasize borderline cases by oversampling misclassified sequences, helping the model learn subtle distinctions between genuine and imposter behavior.

***Synthetic Augmentation Training***
- Function: `train_binary_synth()`
- Calls `synthetic_interp()` to generate realistic augmented sequences by perturbing mouse movement deltas at turning points.
-  helpful in this project where genuine samples are heavily outnumbered by imposter samples from all other users.

- `gen_train_X`, `imp_train_X`: Genuine and imposter sequences
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


##  Summary

This repository implements a fully functional mouse dynamics authentication system using deep learning. The code is modular, well-commented, and includes robust training and evaluation components. Ideal for behavioral biometrics research and practical authentication systems.


##  Project Folder Structure

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

##  Authors
- Karthik Godugolla(Student)
- Daqing Hou (Faculty Adviser)


