"""
This module preprocesses mouse dynamics data for behavioral biometrics authentication.
It supports reading data from individual CSV files organized in user subfolders,
standardizes column names, assigns user/session IDs, and combines all data for model training.

Author: Karthik Godugolla
Date: May 2025
"""

# Importing necessary libraries
import pandas as pd
#  Importing necessary libraries
import numpy as np
#  Importing necessary libraries
import os
from tqdm import tqdm
from collections import Counter

def preprocessor(path, user_col, x_col, y_col, e_col, t_col, subfolder_users=False):
    """
    Loads and preprocesses mouse dynamics data from a directory of CSV files.

    Args:
        path (str): Path to the root directory containing mouse data files.
        user_col (str): Column name in the CSV representing the user ID.
        x_col (str): Column name for the X coordinate of the mouse.
        y_col (str): Column name for the Y coordinate of the mouse.
        e_col (str): Column name representing the event type (e.g., move, click).
        t_col (str): Column name for the timestamp of the event.
        subfolder_users (bool): If True, expects subfolders for each user containing session CSVs.

    Returns:
        pandas.DataFrame: Combined DataFrame of all user sessions, with columns standardized to:
            ['User', 'X', 'Y', 'Event', 'Timestamp', 'Session'].
    """
    # Initialize an empty list to collect DataFrames from each file
    all_data = []
    user = 0  # Counter to assign unique user IDs if not found in file
    sess_col = 'Session'

    #  If data is stored in subfolders (one per user)
    if subfolder_users:
        for subfolder in os.listdir(path):
            sess = 0  # Initialize session count for each user
            subfolder_path = os.path.join(path, subfolder)

            # Skip if it's not a directory
            if not os.path.isdir(subfolder_path):
                continue

            for file in os.listdir(subfolder_path):
                print(subfolder, file)
                file_path = os.path.join(subfolder_path, file)

                # Skip if it's not a file
                if not os.path.isfile(file_path):
                    continue

                try:
                    #  Reading a CSV file into a DataFrame
                    df = pd.read_csv(file_path)
                except pd.errors.ParserError as e:
                    print('ParserError in file: ', file_path)
                    print(e)
                    continue

                #  Remove duplicates and sort by timestamp
                df = df.drop_duplicates(subset=t_col, keep='first')
                df = df.sort_values(by=t_col).reset_index(drop=True)

                # If user_col not provided or missing in file, add it manually
                if not user_col or user_col not in df.columns:
                    df['User'] = [user] * len(df)
                    user_col = 'User'

                # Add a session number column
                df['Session'] = sess

                # Store this DataFrame
                all_data.append(df)
                sess += 1  # Move to next session
            user += 1  # Move to next user

    #  If all CSVs are in a flat folder (no subfolders)
    else:
        for file in os.listdir(path):
            file_path = os.path.join(path, file)

            if not os.path.isfile(file_path):
                continue

            try:
                # Reading a CSV file into a DataFrame
                df = pd.read_csv(file_path)
            except Exception as e:
                print('ParserError in file: ', file_path)
                print(e)
                continue

            # Add a synthetic 'User' column if missing
            if not user_col or user_col not in df.columns:
                df['User'] = [user] * len(df)
                user_col = 'User'

            all_data.append(df)
        user += 1

    #  Combine all session data into one DataFrame
    all_data = pd.concat(all_data, ignore_index=True)

    #  Keep only the relevant columns and standardize column names
    cols = [user_col, sess_col, x_col, y_col, e_col, t_col]
    all_data = all_data[cols]
    all_data.columns = ['User', 'Session', 'X', 'Y', 'Event', 'Timestamp']

    return all_data


def perception_windows(data):
    # Initialize storage for training and validation sequences, labels, lengths, and raw (X, Y, Timestamp) data
    X_train = []
    X_val = []
    y_train = []
    y_val = []
    lens = []
    raw_X = []

    # Total number of users in the dataset
    total_users = len(np.unique(data['User']))

    # Loop through each user
    for user in tqdm(np.unique(data['User'])):
        # Filter data for the current user
        user_X = data[data['User'] == user]

        # Store valid trajectories and raw trajectories for this user
        user_valid_trajs = []
        user_raw_X = []

        # Loop through each session of the current user
        for sess in np.unique(user_X['Session']):
            # Extract session-specific data
            sess_X = user_X[user_X['Session'] == sess].copy()
            raw_sess_X = sess_X[['X', 'Y', 'Timestamp']].copy()

            # Compute time difference between events
            time_diff = sess_X['Timestamp'].diff().fillna(0)

            # Find break points where pause between events >= 250 ms
            p_idx = time_diff >= 250

            # Compute derivatives (ΔX, ΔY, ΔT)
            sess_X[['Delta_X', 'Delta_Y', 'Delta_T']] = raw_sess_X[['X', 'Y', 'Timestamp']].diff().fillna(0)

            # Reset the deltas at pause points to prevent carry-over across split
            sess_X.loc[p_idx, ['Delta_X', 'Delta_Y', 'Delta_T']] = 0

            # Keep only delta features
            sess_X = sess_X[['Delta_X', 'Delta_Y', 'Delta_T']]

            # Split session into trajectory segments using the pause indices
            segs = np.split(sess_X.values, np.where(p_idx)[0])
            segs_raw = np.split(raw_sess_X.values, np.where(p_idx)[0])

            # Keep only segments with at least 5 points
            user_valid_trajs.extend([window for window in segs if len(window) >= 5])
            user_raw_X.extend([window for window in segs_raw if len(window) >= 5])

        # Filter out users with too few valid trajectories
        if int(len(user_valid_trajs) * 0.2) < total_users - 1:
            continue

        # Split user's trajectories into 80% training and 20% validation
        split_idx = int(len(user_valid_trajs) * 0.8)

        # Add to global dataset
        X_train.extend(user_valid_trajs[:split_idx])
        y_train.extend([user] * len(user_valid_trajs[:split_idx]))
        X_val.extend(user_valid_trajs[split_idx:])
        y_val.extend([user] * len(user_valid_trajs[split_idx:]))
        raw_X.extend(user_raw_X[:split_idx])
        lens.extend([len(compat_seg) for compat_seg in user_valid_trajs])

    # Return training and validation sets, along with sequence lengths and raw XYT data
    return X_train, y_train, X_val, y_val, lens, raw_X


def filter_outliers(lens, X):
    #  Print the min and max sequence lengths before filtering
    print(np.min(lens), np.max(lens))

    # Convert the list of lengths to a NumPy array for processing
    lens = np.array(lens)

    #  Sort the lengths and keep their original indices
    lens_sort_inds = np.argsort(lens)
    lens_sorted = lens[lens_sort_inds]

    # Calculate Q1 and Q3 (25th and 75th percentiles) using midpoint method
    q1_lens = np.percentile(lens_sorted, 25, method='midpoint')
    q3_lens = np.percentile(lens_sorted, 75, method='midpoint')

    # Print Q1 and Q3 for inspection
    print(q1_lens, q3_lens)

    #  Compute the interquartile range (IQR)
    IQR_lens = q3_lens - q1_lens

    # Define lower and upper bounds using IQR (Tukey's method)
    low_lim_lens = q1_lens - 1.5 * IQR_lens
    up_lim_lens = q3_lens + 1.5 * IQR_lens

    # Print the bounds for filtering
    print(low_lim_lens, up_lim_lens)

    #  Create a mask to filter out lengths outside [low_lim, up_lim]
    mask = (low_lim_lens < lens_sorted) & (lens_sorted < up_lim_lens)

    # Reorder the mask to match the original unsorted indices
    mask = mask[np.argsort(lens_sort_inds)]

    # Apply the mask to remove outliers from the length list
    lens = lens[mask]

    #  Print how many lengths remain after filtering and their mean
    print(len(lens))
    print(np.mean(lens))

    #  Set minimum and maximum lengths to keep for zero padding/truncation
    bottom_n = int(min(lens))  # lower bound (used as min sequence length)
    top_n = int(max(lens))     # upper bound (used as max sequence length)

    return bottom_n, top_n


def zero_pad(X, y):
    # Temporary lists to store filtered and valid sequences and labels
    tmp_X = []
    tmp_y = []
    traj_list = []  # This list is unused and can be removed if not used later

    # Initialize a zero-padded 3D NumPy array to hold all padded sequences
    # Shape: (number of sequences, max allowed length, 3 features per timestep)
    outs = np.zeros((len(X), top_n, 3))

    a = 0  # Index for tracking how many valid sequences are added

    # Loop through all sequences with their labels
    for i, item in tqdm(enumerate(X), total=len(X)):
        # Only include sequences longer than the bottom threshold
        if len(item) > bottom_n:
            # If sequence is shorter than top_n, pad it on the left (right-aligned)
            if len(item) < top_n:
                needed_zeros = top_n - len(item)
                tmp_y.append(y[i])
                outs[a, -len(item):, :] = item
            else:
                # If sequence is too long, truncate it from the front
                outs[a, -len(item):, :] = item[:top_n]
                tmp_y.append(y[i])
            a += 1

    # Slice the padded output array to include only the filled rows
    X = outs[:len(tmp_y)]
    y = tmp_y

    # Print how many sequences are retained
    print(len(X))

    # Convert to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    return X, y


# Run the custom preprocessor function to load and clean mouse movement data
# Parameters:
# - Path to the dataset
# - 'None' for user_col, so it assigns synthetic user IDs
# - Column names for X, Y, event type, and timestamp
# - subfolder_users=True indicates that each user's data is stored in a separate subfolder
data = preprocessor('data\mouse',
                    None, 'Mouse X', 'Mouse Y', 'Type', 'TimeStamp', True)

# Optional: Save the full cleaned data as a CSV file for inspection
# data.to_csv('Gmail.csv', index=False)

# Optional: Save the full cleaned data as a NumPy array for future use
# np.save('Gmail.npy', data.to_numpy())

# Ensure the important numeric columns are properly converted from strings (if needed)
cols_to_fix = ['X', 'Y', 'Timestamp']
for col in cols_to_fix:
    # Convert column to numeric, setting non-convertible values to NaN
    data[col] = pd.to_numeric(data[col], errors='coerce')


# Convert the 'Timestamp' column to numeric values
# Any non-numeric or malformed values will be set to NaN (coerced)
data['Timestamp'] = pd.to_numeric(data['Timestamp'], errors='coerce')

# Process the cleaned DataFrame into sequences suitable for LSTM modeling
# This function performs:
# - User-wise and session-wise segmentation
# - Trajectory extraction using 250ms pause detection
# - Derivative computation (ΔX, ΔY, ΔT)
# - Sequence filtering (length ≥ 5)
# - 80/20 train-validation split per user
# Returns:
# - X_train, y_train: list of padded input sequences and their labels (train set)
# - X_val, y_val: same for validation set
# - lens: list of lengths of all sequences
# - raw_X: original X/Y/T data for visualization or further analysis
X_train, y_train, X_val, y_val, lens, raw_X = perception_windows(data)


print(len(X_train), len(X_val))

# Apply interquartile range (IQR) based filtering to remove outlier sequences
# from the combined training and validation sets based on their lengths
# This function returns:
# - bottom_n: the minimum sequence length threshold (for padding/truncation)
# - top_n: the maximum sequence length threshold
bottom_n, top_n = filter_outliers(lens, X_train + X_val)

# Print the selected lower and upper bounds for sequence length
print(bottom_n, top_n)


# Pad or truncate each training sequence to have the same length (top_n)
# Only sequences longer than bottom_n are retained
# Resulting shape: (num_sequences, top_n, 3), where 3 = ΔX, ΔY, ΔT
X_train, y_train = zero_pad(X_train, y_train)

# Apply the same zero-padding/truncation process to the validation set
X_val, y_val = zero_pad(X_val, y_val)

# Apply zero-padding to raw (X, Y, Timestamp) sequences for visualization or auxiliary evaluation
# Since these don't have class labels, a dummy label array of ones is passed
raw_X, _ = zero_pad(raw_X, np.ones(len(raw_X)))

# Save each user's padded sequences for binary classification
save_path = r"data\data splits\binary"
os.makedirs(save_path, exist_ok=True)

# Save padded sequences per user
for user_id in np.unique(y_train):
    user_sequences = X_train[y_train == user_id]
    np.save(f"{save_path}{user_id}.npy", user_sequences)


