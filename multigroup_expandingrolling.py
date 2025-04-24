# Rolling Forecast for 52 Weeks Ahead Using All Available Past Data

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# === Load and Prepare Data ===
df = pd.read_excel('your_data2.xlsx', sheet_name='Sheet1')
df = df.sort_values('date')

# Calculate % share of sell_out by feature_1 per date
df['total_sell_out_by_date'] = df.groupby('date')['sell_out'].transform('sum')
df['sell_out_share'] = df['sell_out'] / df['total_sell_out_by_date']

# === Create Sequences ===
def create_sequences(data, input_timesteps=24, output_timesteps=52):
    X, Y = [], []
    for i in range(len(data) - input_timesteps - output_timesteps + 1):
        x_seq = data[i: i + input_timesteps]
        y_seq = data[i + input_timesteps: i + input_timesteps + output_timesteps, 0]
        X.append(x_seq)
        Y.append(y_seq)
    return np.array(X), np.array(Y)

# === LSTM Model Builder ===
def create_lstm_forecaster(input_timesteps, num_features, output_timesteps=52):
    inputs = layers.Input(shape=(input_timesteps, num_features))
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(output_timesteps)(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# === Rolling Forecasting ===
input_timesteps = 24
forecast_horizon = 52
step_size = 1

for group in df['feature_1'].unique():
    print(f"\n=== Rolling Forecast for group: {group} ===")
    df_group = df[df['feature_1'] == group].copy()

    features_to_scale = ['sell_out', 'sell_out_share']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_group[features_to_scale])

    df_scaled = pd.DataFrame(scaled, columns=features_to_scale)
    df_scaled['date'] = df_group['date'].values

    sequence_data = df_scaled[features_to_scale].values
    total_weeks = len(sequence_data)

    start_idx = input_timesteps
    rolling_predictions = []
    rolling_dates = []

    while start_idx + forecast_horizon <= total_weeks:
        current_data = sequence_data[:start_idx]
        X_temp, Y_temp = create_sequences(current_data, input_timesteps, forecast_horizon)

        if len(X_temp) == 0:
            break

        X_input = X_temp[-1].reshape(1, input_timesteps, X_temp.shape[2])

        model = create_lstm_forecaster(input_timesteps, X_input.shape[2], forecast_horizon)
        model.fit(X_temp, Y_temp, epochs=30, batch_size=16, verbose=0)

        pred_scaled = model.predict(X_input).flatten()

        dummy_input = np.zeros((forecast_horizon, len(features_to_scale)))
        dummy_input[:, 0] = pred_scaled
        pred_real = scaler.inverse_transform(dummy_input)[:, 0]

        rolling_predictions.append(pred_real)
        rolling_dates.append(df_scaled['date'].iloc[start_idx])

        start_idx += step_size

    # Plotting the last forecast only (optional)
    if rolling_predictions:
        plt.plot(rolling_predictions[-1], marker='o')
        plt.title(f"52-Week Rolling Forecast: {group}")
        plt.xlabel("Weeks Ahead")
        plt.ylabel("Sell-Out Value")
        plt.grid(True)
        plt.show()

        print(f"Latest forecast for group {group} on date {rolling_dates[-1].date()}:")
        print(rolling_predictions[-1])