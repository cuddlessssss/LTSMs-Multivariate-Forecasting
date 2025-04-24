#script2 

#-----------------------------
#Troubleshooting: ERROR: Could not find a version that satisfies the requirement tensorflow (from versions: none)
#ERROR: No matching distribution found for tensorflow
#Fix -- Only Python 3.8 – 3.11 (as of TensorFlow 2.15+)

#-----------------------------
#Download older version of python first on Python.org
#+ Run IN COMMAND PROMPT this to start a Virtual Environment (Recommended)
#py -3.10 --version

#Setting up the Virual Environment
#py -3.10 -m venv tf-env
#tf-env\Scripts\activate
#pip install --upgrade pip
#pip install tensorflow

# (Alternative) /opt/homebrew/bin/python3.10 -m venv tf_env -> setup virtual environment 
# source tf_env/bin/activate
#-----------------------------
# pip install --upgrade pip
#for mac: pip install pandas numpy tensorflow scikit-learn matplotlib openpyxl
#-----------------------------

# Open the Command Palette (Cmd + Shift + P).

# Type and select: Python: Select Interpreter.

# Choose: ./tf_env/bin/python (your virtual env).

# Open a new terminal (important!) and activate it if not already active.

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ============================
# 1. Load and Prepare Excel Data
# ============================

# Load data from Excel (ensure to provide the correct path and sheet name if needed)
df = pd.read_excel('your_data.xlsx', sheet_name='Sheet1')

# Sort by date to ensure the data is in chronological order
df = df.sort_values('date')

# Define the features and the target variable
features = ['sell_out', 'feature_1', 'feature_2', 'feature_3']

# Normalize the features
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(df[features])
df_scaled = pd.DataFrame(scaled_values, columns=features)
df_scaled['date'] = df['date'].values  # Keep the date for reference

# ============================
# 2. Create Sliding Windows
# ============================

def create_sequences(data, input_timesteps=24, output_timesteps=12):
    """
    data: np.array of shape (total_time_steps, num_features)
    Returns: X of shape (num_samples, input_timesteps, num_features)
             Y of shape (num_samples, output_timesteps)
    """
    X, Y = [], []
    for i in range(len(data) - input_timesteps - output_timesteps + 1):
        x_seq = data[i: i + input_timesteps]
        y_seq = data[i + input_timesteps: i + input_timesteps + output_timesteps, 0]  # sell_out is the target
        X.append(x_seq)
        Y.append(y_seq)
    return np.array(X), np.array(Y)

# Create sequences for training
sequence_data = df_scaled[features].values
X, Y = create_sequences(sequence_data, input_timesteps=24, output_timesteps=12)

# Train-test split (keeping the chronological order intact)
# !!! Printing the shape of X and Y before the split to ensure sufficient data
print(f"Shape of X: {X.shape}")
print(f"Shape of Y: {Y.shape}")

# Ensure enough data exists before performing train-test split
# !!! Make sure X_Train, X_val, Y_Train and Y_val are properly initialized
# test size is important
if len(X) > 1:
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.5, shuffle=False)
    print(f"Train shapes - X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"Validation shapes - X_val: {X_val.shape}, Y_val: {Y_val.shape}")
else:
    print("Not enough data to split into train and test sets.")
    # Fallback: Use all data for training and validation
    X_train, Y_train = X, Y
    X_val, Y_val = X, Y
    print(f"Fallback - X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"Fallback - X_val: {X_val.shape}, Y_val: {Y_val.shape}")

# ============================
# 3. Build the LSTM Model
# ============================

def create_lstm_forecaster(input_timesteps, num_features, output_timesteps=12, use_attention=False):
    """
    Creates an LSTM model for time series forecasting with optional attention mechanism.
    """
    inputs = layers.Input(shape=(input_timesteps, num_features))
    
    # LSTM layers
    x = layers.LSTM(128, return_sequences=use_attention)(inputs)

    if use_attention:
        # Attention mechanism (optional)
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(128)(attention)
        attention = layers.Permute([2, 1])(attention)
        x = layers.Multiply()([x, attention])
        x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(x)
    else:
        x = layers.LSTM(64)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(output_timesteps)(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# Create the model
model = create_lstm_forecaster(input_timesteps=24, num_features=len(features), output_timesteps=12, use_attention=True)
model.summary()

# ============================
# 4. Train the Model
# ============================

model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=50,
    batch_size=32,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

# ============================
# 5. Make 12-Month Prediction
# ============================

# Prepare the latest input data for prediction (the last 24 months)
latest_input_df = df_scaled[features].iloc[-24:]
latest_input = latest_input_df.values.reshape(1, 24, len(features))  # shape: (1, 24, num_features)

# Predict the next 12 months
future_pred_scaled = model.predict(latest_input).flatten()

# Inverse scale the predictions (restore the original scale)
dummy_input = np.zeros((12, len(features)))  # dummy array for inverse scaling

dummy_input[:, 0] = future_pred_scaled  # put the predicted sell_out values in the first column
future_pred_real = scaler.inverse_transform(dummy_input)[:, 0]  # get the real values for sell_out

# Plot the forecasted sell-out values for the next 12 months
plt.plot(future_pred_real, marker='o')
plt.title("12-Month Forecasted Sell-Out")
plt.xlabel("Months Ahead")
plt.ylabel("Sell-Out Value")
plt.grid(True)
plt.show()

# Print the forecasted values
print("Forecasted Sell-Out Values (Next 12 Months):")
print(future_pred_real)
