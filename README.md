# LTSMs-Multivariate-Forecasting
Multivariate Time Series Forecasting with LTSMs

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
multigroup.py

Multi-Target Time Forecaster by 1) sell_out + 2) sell_out% of total @ point in time, Grouped by feature_1 (non-numerical)

Goal - Predict future sell_out values for each group using an LSTM

x-axis - dates
y-axis | target variable - sell_out

Group By
*non-numerical* feature_1 (if want numerical, need to scale with MinMaxScaler())

Features:

1. Historical sell_out values (raw or scaled)

2. (By creation of NEW dataframe based on sell_out values @ point in time) Relative % of sell_out compared to total of other feature_1s on the same date
→ For example: on 2024-01-01, if feature_1 = COURTS sold 100 units and total on that day across all feature_1s is 500, then COURTS gets 100 / 500 = 20%
