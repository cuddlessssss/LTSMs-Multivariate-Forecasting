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
data = pd.read_excel('your_past_data.xlsx')  # Columns: ['date', 'feature_1', 'feature_2', 'sell_out']

# Clean column names (remove any spaces)
data.columns = data.columns.str.strip()

# Ensure 'date' column is in datetime format
data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Create 'year_month' column (use the year and month of the 'date')
data['year_month'] = data['date'].dt.to_period('M')

# Print column names to verify
print("Data columns:", data.columns)

# 📅 Filter to only last 1 year (from most recent date)
latest_date = data['date'].max()
cutoff_date = latest_date - pd.DateOffset(years=1)  # Change from 4 years to 1 year
data = data[data['date'] >= cutoff_date]

# Load future total sell-out targets
future_total_sell_outs = pd.read_excel('future_total_sell_outs.xlsx')  # Columns: ['year_month', 'total_sell_out']

# 2. ⚙️ Preprocessing
# Label Encoding
f1_encoder = LabelEncoder()
f2_encoder = LabelEncoder()
data['feature_1_enc'] = f1_encoder.fit_transform(data['feature_1'])
data['feature_2_enc'] = f2_encoder.fit_transform(data['feature_2'])

# Aggregate by month and feature
grouped = data.groupby(['year_month', 'feature_1_enc', 'feature_2_enc']).agg({'sell_out': 'sum'}).reset_index()

# Pivot to wide format
pivot = grouped.pivot_table(index='year_month', columns=['feature_1_enc', 'feature_2_enc'], values='sell_out', fill_value=0)
pivot = pivot.sort_index(axis=1)

# Normalize: make each month's total = 1
pivot = pivot.div(pivot.sum(axis=1), axis=0)

# Prepare sequences for LSTM
input_sequence_length = 6
X, y = [], []

# Ensure there's enough data to create sequences
if len(pivot) > input_sequence_length:
    for i in range(input_sequence_length, len(pivot)):
        X.append(pivot.iloc[i-input_sequence_length:i].values)
        y.append(pivot.iloc[i].values)

    X = np.array(X)
    y = np.array(y)
else:
    raise ValueError("Not enough data to create sequences for training. Increase your dataset size or reduce sequence length.")

# 3. 🧠 Build Attention Model
# Define Attention Layer
class AttentionWithContext(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], input_shape[-1]),
                                 initializer='random_normal', trainable=True)
        self.u = self.add_weight(name='attention_context_vector', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        super(AttentionWithContext, self).build(input_shape)

    def call(self, x):
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1))
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.squeeze(ait, -1)
        ait = tf.nn.softmax(ait)
        ait = tf.expand_dims(ait, -1)
        weighted_input = x * ait
        output = tf.reduce_sum(weighted_input, axis=1)
        return output, ait

# Build model
input_layer = Input(shape=(input_sequence_length, X.shape[2]))
lstm_out = LSTM(128, return_sequences=True)(input_layer)
attention_layer = AttentionWithContext()
attention_out, attention_weights = attention_layer(lstm_out)
output = Dense(X.shape[2], activation='softmax')(attention_out)

model = Model(inputs=input_layer, outputs=[output, attention_weights])
model.compile(loss=['categorical_crossentropy', None], optimizer=Adam(learning_rate=0.001))

# Train model
model.fit(X, [y, np.zeros_like(y)], epochs=100, batch_size=16)

# 4. 📈 Forecasting Future 12 Months
last_sequence = pivot.iloc[-input_sequence_length:].values
future_preds = []
future_attentions = []

for i in range(12):
    pred, attn = model.predict(last_sequence[np.newaxis, :, :])
    pred = pred[0]
    attn = attn[0, :, 0]
    future_preds.append(pred)
    future_attentions.append(attn)
    last_sequence = np.vstack([last_sequence[1:], pred])

# Rescale prediction based on future monthly totals
final_predictions = []

for i, pred in enumerate(future_preds):
    month_total = future_total_sell_outs.iloc[i]['total_sell_out']
    split_pred = pred * month_total
    final_predictions.append(split_pred)

final_predictions = np.array(final_predictions)

# 5. 📊 Save Predictions and Attention into Excel
# Prepare result dataframe
column_index = pivot.columns
result_df = []

for month_idx in range(12):
    for col_idx, value in enumerate(final_predictions[month_idx]):
        feature_1_enc, feature_2_enc = column_index[col_idx]
        result_df.append({
            'year_month': future_total_sell_outs.iloc[month_idx]['year_month'],
            'feature_1': f1_encoder.inverse_transform([feature_1_enc])[0],
            'feature_2': f2_encoder.inverse_transform([feature_2_enc])[0],
            'predicted_sell_out': value
        })

result_df = pd.DataFrame(result_df)

# Prepare attention weights dataframe
attention_data = []

for month_idx, attn in enumerate(future_attentions):
    for past_month_idx, attn_value in enumerate(attn):
        attention_data.append({
            'forecasted_year_month': future_total_sell_outs.iloc[month_idx]['year_month'],
            'past_month_number': past_month_idx + 1,
            'attention_weight': attn_value
        })

attention_df = pd.DataFrame(attention_data)

# Save to Excel
with pd.ExcelWriter('predicted_split_with_attention.xlsx', engine='openpyxl') as writer:
    result_df.to_excel(writer, sheet_name='Predictions', index=False)
    attention_df.to_excel(writer, sheet_name='Attention', index=False)

# Highlight strongest attention weight
wb = load_workbook('predicted_split_with_attention.xlsx')
ws = wb['Attention']

highlight_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")

attention_df_grouped = attention_df.groupby('forecasted_year_month')['attention_weight'].idxmax()

for idx in attention_df_grouped.values:
    excel_row = idx + 2  # Account for header row
    for col_letter in ['A', 'B', 'C']:
        ws[f'{col_letter}{excel_row}'].fill = highlight_fill

wb.save('predicted_split_with_attention.xlsx')

print("✅ Done! Excel file saved with predictions and highlighted attention weights!")
