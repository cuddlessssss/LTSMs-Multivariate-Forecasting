# Multivariate LSTM Forecasting grouped by 'feature_1'

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load and prepare Excel data
df = pd.read_excel('your_data2.xlsx', sheet_name='Sheet1')
df = df.sort_values('date')

# Calculate % share of sell_out by feature_1 per date
df['total_sell_out_by_date'] = df.groupby('date')['sell_out'].transform('sum')
df['sell_out_share'] = df['sell_out'] / df['total_sell_out_by_date']

# Function to create sequences
def create_sequences(data, input_timesteps=24, output_timesteps=12):
    X, Y = [], []
    for i in range(len(data) - input_timesteps - output_timesteps + 1):
        x_seq = data[i: i + input_timesteps]
        y_seq = data[i + input_timesteps: i + input_timesteps + output_timesteps, 0]
        X.append(x_seq)
        Y.append(y_seq)
    return np.array(X), np.array(Y)

# Define model builder
def create_lstm_forecaster(input_timesteps, num_features, output_timesteps=12):
    inputs = layers.Input(shape=(input_timesteps, num_features))
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(output_timesteps)(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# Train and forecast for each group
input_timesteps = 24
output_timesteps = 12

for group in df['feature_1'].unique():
    print(f"\n=== Processing group: {group} ===")
    df_group = df[df['feature_1'] == group].copy()

    # Scale features
    features_to_scale = ['sell_out', 'sell_out_share']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_group[features_to_scale])

    df_scaled = pd.DataFrame(scaled, columns=features_to_scale)
    df_scaled['date'] = df_group['date'].values

    sequence_data = df_scaled[features_to_scale].values
    X, Y = create_sequences(sequence_data, input_timesteps, output_timesteps)

    if len(X) < 1:
        print("Not enough data to train. Skipping...")
        continue

    # Train-test split
    split_index = int(0.7 * len(X))
    X_train, X_val = X[:split_index], X[split_index:]
    Y_train, Y_val = Y[:split_index], Y[split_index:]

    # Build and train the model
    model = create_lstm_forecaster(input_timesteps, X.shape[2], output_timesteps)
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=50,
        batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )

    # Forecast
    latest_input = sequence_data[-input_timesteps:].reshape(1, input_timesteps, X.shape[2])
    future_pred_scaled = model.predict(latest_input).flatten()

    # Inverse scale only sell_out (first column)
    dummy_input = np.zeros((output_timesteps, len(features_to_scale)))
    dummy_input[:, 0] = future_pred_scaled
    future_pred_real = scaler.inverse_transform(dummy_input)[:, 0]

    # Plot
    plt.figure()
    plt.plot(future_pred_real, marker='o')
    plt.title(f"12-Month Forecast: {group}")
    plt.xlabel("Months Ahead")
    plt.ylabel("Sell-Out Value")
    plt.grid(True)
    plt.show()

    # Output values
    print("Forecasted Sell-Out:")
    print(future_pred_real)