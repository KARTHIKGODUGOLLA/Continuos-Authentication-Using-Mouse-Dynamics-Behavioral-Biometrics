"""
This module builds and trains a binary classification model using BiLSTM
for continuous authentication based on mouse dynamics data.
It includes data augmentation, preprocessing, model definition,
and evaluation with AUC and EER metrics.

Author: Karthik Godugolla
Date: May 2025
"""

#  Importing necessary libraries
import numpy as np
#  Importing necessary libraries
import pandas as pd
#  Importing necessary libraries
import os
#  Splitting data into train and test sets
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Masking, Bidirectional, Input
from tensorflow.keras.metrics import AUC, Precision
from sklearn.utils.class_weight import compute_class_weight
#  Calculating the AUC (Area Under Curve)
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from scipy.optimize import brentq
from scipy.interpolate import interp1d, CubicSpline
#  Importing necessary libraries
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle

def synthetic_interp(gen_X, gen_y, seqs_needed, noise_std=5, ang_thrsh=25):
    """
    Generates synthetic mouse movement sequences by replicating and modifying
    existing sequences with optional noise and angle thresholding.

    Args:
        gen_X (np.ndarray): Original input sequences to replicate.
        gen_y (np.ndarray): Corresponding labels for the sequences.
        seqs_needed (int): Total number of sequences desired after augmentation.
        noise_std (float, optional): Standard deviation of Gaussian noise to apply. Default is 5.
        ang_thrsh (float, optional): Angle threshold in degrees to trigger modifications. Default is 25.

    Returns:
        tuple: (new_X, new_y) where both are numpy arrays of augmented sequences and labels.
    """
    num_reps = (seqs_needed // gen_X.shape[0]) - 1

    new_X = []
    new_y = []

    for seq, y_val in zip(gen_X, gen_y):
        new_X.append(seq.copy())
        new_y.append(y_val)

        non_zero_mask = ~np.all(seq == 0, axis=1)
        non_zero_inds = np.where(non_zero_mask)[0]

        x_deltas = seq[non_zero_inds, 0]
        y_deltas = seq[non_zero_inds, 1]
        t_deltas = seq[non_zero_inds, 2]

        # Find the angle at each time step (arctan2 so we aren't constrained from 0-90 degrees)
        angles = np.arctan2(y_deltas, x_deltas)
        angles = np.unwrap(angles)

        # Get the list of changes in angle and convert to degrees
        angle_changes = np.diff(angles, prepend=angles[0])
        angle_changes = angle_changes * (180 / np.pi)

        # Define turning points as points where the abs val of an angle exceeds the threshold
        turning_inds = np.where(np.abs(angle_changes) > ang_thrsh)[0] - 1
        turning_inds = turning_inds[turning_inds >= 0]
        turning_points = non_zero_inds[turning_inds]

        # If we have no turning points (mostly straight line) then noise the deltas between start and end
        if len(turning_inds) == 0:
            turning_inds = np.arange(1, len(non_zero_inds) - 1)
        turning_points = non_zero_inds[turning_inds]

        # print(f'Turning point count: {len(turning_points)} out of {len(x_deltas)}')

        # Redefine deltas to be at the turning inds
        x_deltas = x_deltas[turning_inds]
        y_deltas = y_deltas[turning_inds]
        t_deltas = t_deltas[turning_inds]

        # Calculate original velocity from x and y components
        old_vx = x_deltas / t_deltas
        old_vy = y_deltas / t_deltas
        old_v = np.sqrt(old_vx**2 + old_vy**2)

        # Find the original magnitudes of distance
        old_distances = np.sqrt(x_deltas**2 + y_deltas**2)

        for _ in range(num_reps):
            # For each augmented trajectory, randomly pick whether the sign of x and y will be the same or flipped
            inverse = np.random.choice([True, False])
            seq_new = np.copy(seq)

            max_dist = np.max(old_distances)

            # Define a weight between 0 and 1 for the noise based on magnitude of dist. (bigger the dist. bigger the weight)
            if max_dist > 0:
                weights = old_distances / max_dist
            else:
                weights = np.ones_like(old_distances)

            # Get the sign (+ or -) of the delta at each time step
            sign_x = np.sign(seq_new[turning_points, 0])
            sign_y = np.sign(seq_new[turning_points, 1])

            # Create gaussian noise to apply to X, Y using weights to ensure the noise added doesn't drastically change small deltas
            noise_x = np.random.normal(0, noise_std, seq_new[turning_points, 0].shape) * weights
            noise_y = np.random.normal(0, noise_std, seq_new[turning_points, 1].shape) * weights

            # Apply the noise of the same sign or inverse of the sign to X, Y (this keeps the new traj from zig zagging)
            seq_new[turning_points, 0] += noise_x * sign_x * (-1 if inverse else 1)
            seq_new[turning_points, 1] += noise_y * sign_y * (-1 if inverse else 1)

            # Get the new deltas for x and y, and find the new magnitude of dist from new x and y deltas
            new_x_deltas = seq_new[turning_points, 0]
            new_y_deltas = seq_new[turning_points, 1]
            new_distances = np.sqrt(new_x_deltas**2 + new_y_deltas**2)

            # Ensure that we aren't trying to find new times for inds where v was 0 (likely clicks)
            valid = np.where(old_v > 0)[0]

            # Calculate new time as t_new = new_distances / old_v
            seq_new[turning_points[valid], 2] = (new_distances[valid] / old_v[valid])

            # Use ceiling to ensure at least 1 gets added to the turning points and to keep the deltas int pixel values
            new_X.append(np.ceil(seq_new))
            new_y.append(y_val)

    new_X = np.array(new_X)
    new_y = np.array(new_y)

    return new_X, new_y
    if new_X.shape[0] > seqs_needed:
        replace = False
    else:
        replace = True

    inds = np.random.choice(len(new_X), seqs_needed, replace=replace)
    new_X = new_X[inds]
    new_y = new_y[inds]

    return new_X, new_y

def train_binary(data_path, out_path, max_loops=5, patience=2):
    # Ensure the output directory exists
    os.makedirs(out_path, exist_ok=True)

    # Get a sorted list of all .npy files (one per user)
    all_files = np.sort([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')])

    # Loop through each user file (starting from index 18 here)
    for file in all_files[18:]:
        best_auc = -np.inf  # Track the best AUC achieved
        wait = 0  # Patience counter for early stopping

        # Loop to allow multiple training attempts per user
        for loop in range(max_loops):
            print(f"\n--- Loop {loop + 1}/{max_loops} ---")
            user = os.path.splitext(os.path.basename(file))[0]  # Extract user ID from filename

            # Load genuine user data
            gen_data = np.load(file)

            # Load and stack data from all other users (imposters)
            imp_files = [f for f in all_files if f != file]
            imp_data = np.vstack([np.load(f) for f in imp_files])

            # 80/20 train/val split for genuine and imposter data
            split_idx_gen = int(len(gen_data) * 0.8)
            split_idx_imp = int(len(imp_data) * 0.8)
            gen_train_X, gen_val_X = gen_data[:split_idx_gen], gen_data[split_idx_gen:]
            imp_train_X, imp_val_X = imp_data[:split_idx_imp], imp_data[split_idx_imp:]

            # Label the data: 1 for genuine, 0 for imposters
            gen_train_y = np.ones(len(gen_train_X))
            gen_val_y = np.ones(len(gen_val_X))
            imp_train_y = np.zeros(len(imp_train_X))
            imp_val_y = np.zeros(len(imp_val_X))

            # Combine training and validation datasets
            X_train = np.vstack([gen_train_X, imp_train_X])
            y_train = np.concatenate([gen_train_y, imp_train_y])
            X_val = np.vstack([gen_val_X, imp_val_X])
            y_val = np.concatenate([gen_val_y, imp_val_y])

            # Shuffle the training data
            X_train, y_train = shuffle(X_train, y_train, random_state=42)

            # Define model save path
            model_path = os.path.join(out_path, f'{user}_min_val_loss.keras')

            # Build or load the model depending on the loop
            if loop == 0:
                # Build a new BiLSTM model
                model = Sequential([
                    Input(shape=(X_train.shape[1], X_train.shape[2])),
                    Masking(mask_value=0.0),
                    Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2)),
                    BatchNormalization(),
                    Dropout(0.3),
                    Bidirectional(LSTM(128, recurrent_dropout=0.2)),
                    Dense(1, activation='sigmoid')
                ])
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC(name='auc')])
            else:
                # Load previously saved best model
                model = load_model(model_path)

            # Set up model checkpoint to save only if validation AUC improves
            mc = ModelCheckpoint(
                model_path,
                monitor='val_auc',
                mode='max',
                initial_value_threshold=best_auc,
                save_best_only=True,
                verbose=1
            )

            # Train the model
            history = model.fit(
                X_train, y_train,
                epochs=10,
                batch_size=512,
                validation_data=(X_val, y_val),
                callbacks=[mc]
            )

            # Predict on the training set
            y_pred_train = model.predict(X_train).flatten()
            y_pred_train_labels = (y_pred_train >= 0.5).astype(int)

            # Compute confusion matrix on training data
            cm_train = confusion_matrix(y_train, y_pred_train_labels)
            print(f'Confusion Matrix:\n{cm_train}')

            # Identify false negatives and false positives
            fn_indices = np.where((y_train == 1) & (y_pred_train_labels == 0))[0]
            fp_indices = np.where((y_train == 0) & (y_pred_train_labels == 1))[0]

            # Oversample difficult cases to help the model focus on errors
            oversampled_X = np.vstack([X_train, X_train[fn_indices], X_train[fp_indices]])
            oversampled_y = np.concatenate([y_train, y_train[fn_indices], y_train[fp_indices]])

            # Shuffle the new training data
            X_train, y_train = shuffle(oversampled_X, oversampled_y, random_state=42)

            # Get the best validation AUC from this training loop
            current_auc = max(history.history['val_auc'])

            # Check for improvement
            if current_auc > best_auc:
                best_auc = current_auc
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print('Stopping early on plateau')
                    break


def train_binary_over(data_path, out_path):
    """
    Trains a binary LSTM model for user authentication using oversampled training data.
    For each user, the model is trained to distinguish between genuine and imposter mouse dynamics sequences.
    Oversampling is used to balance the genuine and imposter classes.

    Args:
        data_path (str): Path to the folder containing .npy files (each representing a user's sequences).
        out_path (str): Path to save the trained models (.keras format).

    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(out_path, exist_ok=True)

    # Load all user data files (.npy)
    all_files = np.sort([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')])

    for file in all_files:
        # Extract user ID from file name
        user = os.path.splitext(os.path.basename(file))[0]

        # Load genuine data for the current user
        gen_data = np.load(file)

        # Load imposter data (from all other users)
        imp_files = [f for f in all_files if f != file]
        imp_data = np.vstack([np.load(f) for f in imp_files])

        # Split genuine and imposter data into training and validation sets (80/20 split)
        split_idx_gen = int(len(gen_data) * 0.8)
        split_idx_imp = int(len(imp_data) * 0.8)
        gen_train_X, gen_val_X = gen_data[:split_idx_gen], gen_data[split_idx_gen:]
        imp_train_X, imp_val_X = imp_data[:split_idx_imp], imp_data[split_idx_imp:]

        # Oversample genuine training data to match imposter data size
        gen_inds = np.random.choice(len(gen_train_X), len(imp_train_X), replace=True)
        gen_train_X = gen_train_X[gen_inds]

        # Create corresponding labels
        gen_train_y = np.ones(len(gen_train_X))
        gen_val_y = np.ones(len(gen_val_X))
        imp_train_y = np.zeros(len(imp_train_X))
        imp_val_y = np.zeros(len(imp_val_X))

        # Combine genuine and imposter data for training and validation
        X_train = np.vstack([gen_train_X, imp_train_X])
        y_train = np.concatenate([gen_train_y, imp_train_y])
        X_val = np.vstack([gen_val_X, imp_val_X])
        y_val = np.concatenate([gen_val_y, imp_val_y])

        # Shuffle the training data
        X_train, y_train = shuffle(X_train, y_train, random_state=42)

        # Define path to save the best model for the current user
        model_path = os.path.join(out_path, f'{user}_min_val_loss.keras')

        # Set up early stopping based on validation AUC to prevent overfitting
        early_stop = EarlyStopping(
            monitor='val_auc',
            patience=10
        )

        # Build a standard LSTM model for binary classification
        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            Masking(mask_value=0.0),
            LSTM(128, return_sequences=True, recurrent_dropout=0.2),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(128, recurrent_dropout=0.2),
            Dense(1, activation='sigmoid')
        ])

        # Compile the model with AUC and accuracy metrics
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC(name='auc')])

        # Save the model checkpoint only if validation AUC improves
        mc = ModelCheckpoint(model_path, monitor='val_auc', mode='max', save_best_only=True, verbose=1)

        # Train the model on oversampled data
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=512,
            validation_data=(X_val, y_val),
            callbacks=[mc, early_stop]
        )


def train_binary_synth(data_path, out_path):
    # Create output directory if it doesn't exist
    os.makedirs(out_path, exist_ok=True)

    # Load all .npy files (each representing one user's data)
    all_files = np.sort([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')])

    for file in all_files:
        # Extract user ID from filename
        user = os.path.splitext(os.path.basename(file))[0]

        # Load genuine user data
        gen_data = np.load(file)

        # Load and stack data from all other users as imposter data
        imp_files = [f for f in all_files if f != file]
        imp_data = np.vstack([np.load(f) for f in imp_files])

        # 80/20 train-validation split for genuine and imposter data
        split_idx_gen = int(len(gen_data) * 0.8)
        split_idx_imp = int(len(imp_data) * 0.8)
        gen_train_X, gen_val_X = gen_data[:split_idx_gen], gen_data[split_idx_gen:]
        imp_train_X, imp_val_X = imp_data[:split_idx_imp], imp_data[split_idx_imp:]

        # Label data: 1 for genuine, 0 for imposters
        gen_train_y = np.ones(len(gen_train_X))
        gen_val_y = np.ones(len(gen_val_X))
        imp_train_y = np.zeros(len(imp_train_X))
        imp_val_y = np.zeros(len(imp_val_X))

        # (optional) Downsampling real genuine data
        # gen_inds = np.random.choice(len(gen_train_X), len(imp_train_X)//2, replace=True)
        # gen_train_X = gen_train_X[gen_inds]

        # Augment genuine data using synthetic interpolation
        seq_needed = len(imp_train_X)
        gen_train_X, gen_train_y = synthetic_interp(gen_train_X, gen_train_y, seq_needed, 0.25, 30)

        # Combine genuine and imposter training data
        X_train = np.vstack([gen_train_X, imp_train_X])
        y_train = np.concatenate([gen_train_y, imp_train_y])

        # Combine genuine and imposter validation data
        X_val = np.vstack([gen_val_X, imp_val_X])
        y_val = np.concatenate([gen_val_y, imp_val_y])

        # Shuffle training data
        X_train, y_train = shuffle(X_train, y_train, random_state=42)

        # Define path to save the best model
        model_path = os.path.join(out_path, f'{user}_min_val_loss.keras')

        # Define early stopping based on validation AUC
        early_stop = EarlyStopping(
            monitor='val_auc',
            patience=10
        )

        # Build the BiLSTM model
        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            Masking(mask_value=0.0),
            Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2)),
            BatchNormalization(),
            Dropout(0.3),
            Bidirectional(LSTM(128, recurrent_dropout=0.2)),
            Dense(1, activation='sigmoid')
        ])

        # Alternative architecture with single-directional LSTM (commented out)
        # model = Sequential([
        #     Input(shape=(X_train.shape[1], X_train.shape[2])),
        #     Masking(mask_value=0.0),
        #     LSTM(128, return_sequences=True, recurrent_dropout=0.2),
        #     BatchNormalization(),
        #     Dropout(0.3),
        #     LSTM(128, recurrent_dropout=0.2),
        #     Dense(1, activation='sigmoid')
        # ])

        # Compile the model with binary cross-entropy and AUC metric
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC(name='auc')])

        # Model checkpoint to save only the best model based on validation AUC
        mc = ModelCheckpoint(model_path, monitor='val_auc', mode='max', save_best_only=True, verbose=1)

        # Train the model with validation and callbacks
        history = model.fit(X_train, y_train, epochs=100, batch_size=512, validation_data=(X_val, y_val), callbacks=[mc, early_stop])

if __name__ == "__main__":
    #replace the first argument path with the .npy sequences generated from the mouse_preprocessor file(stored in the folder data/data splits/binary) and the second argument with the output path where you want to store the trained models for testing with test_binary_LSTM
    train_binary_synth(r'data\data splits\binary', r'trained_models')

