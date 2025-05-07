import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

# 1. 🗂️ Load Data
# Load past sales data
data = pd.read_excel('your_past_data.xlsx', sheet_name='updated')  # Columns: ['Date', 'Model Name', 'Account Name', 'Dealer Net Sell Out Qty']

# Clean column names (remove any spaces)
data.columns = data.columns.str.strip()

# Ensure 'Date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Create 'year_month' column (use the year and month of the 'Date')
data['year_month'] = data['Date'].dt.to_period('M')

# Print column names to verify
print("Data columns:", data.columns)

# 📅 Filter to only last 1 year (from most recent date)
latest_date = data['Date'].max()
cutoff_date = latest_date - pd.DateOffset(years=3)  # Change from 4 years to 1 year
data = data[data['Date'] >= cutoff_date]

# Load future total sell-out targets
future_total_sell_outs = pd.read_excel('future_total_sell_outs.xlsx')  # Columns: ['year_month', 'total_sell_out']

# 2. ⚙️ Preprocessing
# Label Encoding
f1_encoder = LabelEncoder()
f2_encoder = LabelEncoder()
data['Model Name_enc'] = f1_encoder.fit_transform(data['Model Name'])
data['Account Name_enc'] = f2_encoder.fit_transform(data['Account Name'])

# Aggregate by month and feature
grouped = data.groupby(['year_month', 'Model Name_enc', 'Account Name_enc']).agg({'Dealer Net Sell Out Qty': 'sum'}).reset_index()

# Pivot to wide format
pivot = grouped.pivot_table(index='year_month', columns=['Model Name_enc', 'Account Name_enc'], values='Dealer Net Sell Out Qty', fill_value=0)
pivot = pivot.sort_index(axis=1)

# Normalize: make each month's total = 1
pivot = pivot.div(pivot.sum(axis=1), axis=0)

# Prepare sequences for LSTM
input_sequence_length = 12  # Change from 36 months to 12 months for lookback
X, y = [], []

# Ensure there's enough data to create sequences
if len(pivot) > input_sequence_length:
    for i in range(input_sequence_length, len(pivot)):
        X.append(pivot.iloc[i-input_sequence_length:i].values)  # Use last 24 months
        y.append(pivot.iloc[i].values)

    X = np.array(X)
    y = np.array(y)
else: 
    raise ValueError("Not enough data to create sequences for training. Increase your dataset size or reduce sequence length.")

# 3. 🧠 Build Attention Model (removed attention mechanism)
# Build model
input_layer = Input(shape=(input_sequence_length, X.shape[2]))
lstm_out = LSTM(128)(input_layer)
output = Dense(X.shape[2], activation='softmax')(lstm_out)

model = Model(inputs=input_layer, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))

# Train model
model.fit(X, y, epochs=100, batch_size=16)

# 4. 📈 Forecasting Future 12 Months
last_sequence = pivot.iloc[-input_sequence_length:].values
future_preds = []

for i in range(12):
    pred = model.predict(last_sequence[np.newaxis, :, :])
    pred = pred[0]
    future_preds.append(pred)
    last_sequence = np.vstack([last_sequence[1:], pred])

# Rescale prediction based on future monthly totals
final_predictions = []

for i, pred in enumerate(future_preds):
    month_total = future_total_sell_outs.iloc[i]['total_sell_out']
    split_pred = pred * month_total
    final_predictions.append(split_pred)

final_predictions = np.array(final_predictions)

# 5. 📊 Save Predictions into Excel
# Prepare result dataframe
column_index = pivot.columns
result_df = []

for month_idx in range(12):
    for col_idx, value in enumerate(final_predictions[month_idx]):
        feature_1_enc, feature_2_enc = column_index[col_idx]
        result_df.append({
            'year_month': future_total_sell_outs.iloc[month_idx]['year_month'],
            'Model Name': f1_encoder.inverse_transform([feature_1_enc])[0],
            'Account Name': f2_encoder.inverse_transform([feature_2_enc])[0],
            'predicted_Dealer_Net_Sell_Out_Qty': value
        })

result_df = pd.DataFrame(result_df)

# Save to Excel
result_df.to_excel('predicted_split.xlsx', index=False)

print("✅ Done! Excel file saved with predictions!")
