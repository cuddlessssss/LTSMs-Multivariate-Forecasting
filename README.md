# LSTMs-Multivariate-Forecasting
Multivariate Time Series Forecasting with LSTMs (Long-Short-Term-Memory)

🧠 LSTM Basis of Predictions:
1. Patterns in Past Data
LSTM learns from sequences — so it looks at how the values change over time.

Example:
If sales increase every December and drop in January, the LSTM can learn that pattern and use it to forecast future Dec/Jan trends.

2. Input Sequence (a "window" of past values)
You provide the model with a time window (e.g., last 30 days of data), and it learns how to predict the next step.

This is like saying:

"Given that the last 30 days looked like this, what is the most likely value on day 31?"

3. Weights Learned During Training
LSTM contains weights (parameters) that get adjusted during training to minimize prediction error.

It tries to find the best internal "rules" for how past values influence future ones.

4. Temporal Relationships
LSTMs can capture both:

Short-term dependencies (e.g., sales yesterday influence today’s sales)

Long-term dependencies (e.g., yearly seasonality, like holidays or trends)

-----------------------------------------
My Project Goals:
 • Predict future values based on past values inputted
 • Use past values and weighted percentages per month (likely features influencing our variable of study)
 • The model should be incrementally updated as new data arrives
 
What the code provides
 • A trained LSTM model
 • A 12-month forecast
 • A plotted result

Workflow:
 1. Preprocess your data (normalize, handle missing values, encode categorical features)
 2. Design LSTM model with inputs:
 • Past sell-out values
 • Monthly weighted features (could be auxiliary inputs)
 3. Train on historical data
 4. Update periodically with new monthly data
 5. (Optional) Add attention layer to learn importance of each month
-------------------
Key Considerations:

Factor Your Need

Multivariate inputs Yes

Weighted influence of features Yes

Temporal modeling Yes (time series)

Adaptability to new data Yes (online / retraining)

Non-linear relationships Likely

-------------------
Model Recommendations:

1. LSTM (or LSTM with Attention) — Strong Fit
 • Can learn non-linear temporal dependencies
 • Handles multiple input features (e.g., monthly weights)
 • You can use attention mechanisms to let the model learn which months have more impact
 • Easily supports retraining with new data (e.g., fine-tune monthly or online learning variants)

2. Temporal Fusion Transformer (TFT) — Advanced Option
 • Built for time series forecasting
 • Can learn feature importance dynamically
 • Supports multi-horizon forecasting and interpretability
 • Great for complex feature sets, but more resource-intensive

⸻

3. VAR?
 • Not ideal here. It assumes linearity and stationarity, and doesn’t handle complex feature weighting easily.
 • Hard to adapt quickly to new data — retraining is slower and less flexible.
---------------------
12forecasts.py

allows 2 features (feature_2 subfeature of feature_1) 

Purpose: GIVEN the total forecast of each month (for 12 mths) -> breaking down to how much per combination of feature_1 and feature_2
then exports via excel

#📦 Files you need:

your_past_data.csv with columns: date, feature_1, feature_2, sell_out

future_total_sell_outs.csv with columns: year_month, total_sell_out

#Steps Overview:

Data Preprocessing:

Load Historical Data: The past sales data is loaded from an Excel file. The date is converted to a datetime object, and only the last 4 years of data is kept by filtering based on the most recent date.

Load Future Total Sell-Out Targets: This is the total sell-out value that needs to be distributed across feature_1 and feature_2 for each of the next 12 months.

Label Encoding: feature_1 and feature_2 are encoded into numerical values to prepare them for the model.

Data Aggregation:

The data is aggregated by month, feature_1, and feature_2 to compute the total sell_out values for each combination. This data is then pivoted to create a wide-format table where each combination of feature_1 and feature_2 has its own column, and the rows correspond to monthly data.

The table is then normalized so that the sum of each month's rows equals 1 (this ensures predictions are proportional).

Sequence Preparation:

To train the LSTM model, sequences of past data are prepared using a sliding window of input_sequence_length=6 (i.e., using the past 6 months' data to predict the next month's split).

The sequences are used to train the LSTM model.

Model Building:

The LSTM model is combined with an attention mechanism to give more weight to important past months when making predictions for the future months. The AttentionWithContext layer is custom-built to calculate attention scores, allowing the model to focus on key time points.

The model outputs the predicted sell_out proportions for each feature_1 and feature_2 combination.

Forecasting:

The trained model predicts the proportions of sell_out for each of the 12 future months.

These proportions are then scaled by the respective total sell_out value for each month (from the future_total_sell_outs.xlsx file).

Save Predictions:

The results (predicted sell_out values) are saved into an Excel file. For each month, the sell_out values are assigned to each feature_1 and feature_2 combination, and attention weights are stored for each historical month.

The attention weights are highlighted for the highest weight in each future month, which shows which past months had the greatest influence on the forecast.

Excel Output:

The Excel file contains two sheets:

Predictions: Contains the predicted sell_out values for each combination of feature_1 and feature_2 for each of the 12 future months.

Attention: Contains the attention weights for each month, indicating which past months were most influential.

---------------------
multivariate_forecasting.py
Original Base

----------------------
multigroup.py

Multi-Target Time Forecaster using by 1) sell_out + 2) sell_out% of total @ point in time, Grouped by feature_1 (non-numerical)

Goal - Predict future sell_out values for each group using an LSTM

x-axis - dates
y-axis | target variable - sell_out

Group By
*non-numerical* feature_1 (if want numerical, need to scale with MinMaxScaler())

Features:

1. Historical sell_out values (raw or scaled)

2. (By creation of NEW dataframe based on sell_out values @ point in time) Relative % of sell_out compared to total of other feature_1s on the same date
→ For example: on 2024-01-01, if feature_1 = COURTS sold 100 units and total on that day across all feature_1s is 500, then COURTS gets 100 / 500 = 20%
-----------------------
🔁 What’s the Problem?
When training models like RNNs on sequences (like time series), two main problems happen:

Vanishing gradients – the model forgets older information because the gradients become too small while training.

Exploding gradients – the model becomes unstable because gradients become too large and cause huge jumps in training.

🧠 What Makes LSTM Different?
LSTM (Long Short-Term Memory) is a special kind of RNN that solves both problems using gates and a cell state.

🧩 Key Ideas (in plain terms):
1. Memory Cell (Cell State)
Think of it like a conveyor belt that carries important information from the past into the future.

It moves along without being changed too much.

This helps the model remember things for longer.

2. Gates (like valves)
There are 3 gates in LSTM:

Forget gate: What old information should we forget?

Input gate: What new information should we add?

Output gate: What do we show as output?

Each gate controls the flow using values between 0 and 1 (like a dimmer switch).

✋ How LSTM Stops Vanishing Gradients
In normal RNNs, the signal fades (vanishes) as it moves backward in time.

In LSTM:

The forget gate can choose to keep information by setting its value close to 1.

Instead of multiplying again and again, LSTM adds information in the cell state.

Adding keeps the information strong, so the model remembers older patterns.

✅ Result: Gradients don’t vanish → LSTM can learn long-term patterns.

🚫 How LSTM Stops Exploding Gradients
All gates in LSTM use sigmoid or tanh, which squash values between -1 and 1.

This means no huge numbers can come out.

Also, optimizer tricks like gradient clipping can stop exploding values.

✅ Result: Training is more stable → LSTM doesn’t explode.

🎯 Final Summary
Problem	What happens?	How LSTM fixes it
Vanishing gradient	Model forgets old info	| Keeps memory with cell state + forget gate
Exploding gradient	Model gets unstable |	Uses gates + safe activations to limit size
