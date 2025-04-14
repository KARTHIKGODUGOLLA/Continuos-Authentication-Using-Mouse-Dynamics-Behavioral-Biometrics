
# 🧾 README: Raw Mouse Data Processing for Behavioral Biometrics

## 📁 Dataset Source
This project uses the **Multimodal Gmail Dataset for Behavioral Biometrics Research**, which captures behavioral data from 43 users completing realistic Gmail tasks using a Firefox browser extension.

### 📂 Raw Data Structure
The dataset is organized as follows:
```
Gmail Dataset Formatted/
└── mouse/
    ├── 1067144/
    │   ├── session_0.csv
    ├── 1513190/
    │   ├── session_0.csv
    │   ├── session_1.csv
    │   └── ...
    └── ...
```

Each user folder contains multiple `session_*.csv` files capturing **mouse movement events** during Gmail usage.

---

## 🧠 Raw Data Format (Per CSV File)
Each row in a session file represents a single mouse event:

| Tab ID | ClientID | Type     | Mouse X | Mouse Y | Timestamp     |
|--------|----------|----------|---------|---------|---------------|
| 1067144| 1067144  | movement | 1215    | 116     | 1707232517759 |

- **Type**: Filtered to `movement` only.
- **Timestamp**: Recorded in **milliseconds** (Unix epoch time).
- **Mouse X/Y**: Mouse position on screen.

---

## 🔁 Trajectory Splitting Logic
Each session is processed by detecting **trajectories**, defined as:
> A sequence of continuous mouse movements **separated by pauses ≥100ms**.

This is based on human perception thresholds and is implemented in the `trajectories()` function.

### ✅ Example:
```text
Timestamp (ms): 1000 → 1040 → 1070 → 1200 → 1225 → 1455
Differences:     +40    +30    +130    +25    +230
```

Results in 3 trajectories:
- [1000–1070]
- [1200–1225]
- [1455–...]

---

## 📊 Feature Extraction
Each **trajectory** is converted into a **row** of numerical features, saved into `mouse_feature_vectors.csv`.

Features include:
- Duration, Curve Length
- Min/Max/Mean: Velocity, Acceleration
- Angle of Movement
- Session ID (preserved as a column)

---

## 📦 Output Format
The final `mouse_feature_vectors.csv` contains all user trajectories:

| Tab_ID | duration | velocity | ... | session |
|--------|----------|----------|-----|---------|
| 1067144 | 120 ms  | 0.13     | ... | 0 |
| 1067144 | 180 ms  | 0.09     | ... | 1 |
| ...     | ...      | ...      | ... | ... |

All data from **all users and sessions** is flattened into one CSV file for modeling, but can be grouped later by user/session.

---

## 🧪 Model Input
This CSV can be used for:
- Classical ML models (RF, SVM, GBM)
- Sequence-based models (LSTM/BiLSTM) after reshaping into time series format.

