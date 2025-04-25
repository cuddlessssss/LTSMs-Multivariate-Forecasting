# LSTMs-Multivariate-Forecasting
Multivariate Time Series Forecasting with LTSMs

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
multigroup_expandingrolling.py (not functional yet)

Almost the same as multigroup.py except it includes an
-> Expanding window of ALL historical data
