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
||Mathematics||

1. Neural Network Basics
A neural network consists of layers of neurons, where each layer processes input data through linear transformations followed by non-linear activations. For a fully connected neural network (like the LSTM model in your code), the typical operations can be described as:

Linear Transformation:

𝑧
=
𝑊
⋅
𝑥
+
𝑏
z=W⋅x+b

Where:

𝑊
W is the weight matrix (learned by the model),

𝑥
x is the input vector (features),

𝑏
b is the bias term.

Activation Function:

𝑦
=
𝜎
(
𝑧
)
y=σ(z)

Where 
𝜎
σ is an activation function (e.g., ReLU, Sigmoid, Softmax).

In the case of LSTMs, the "neurons" are a bit more complex, as they involve memory cells that can retain information over time.

2. LSTM Basics
The LSTM network is specifically designed to handle sequential data (time series, text, etc.) by learning long-term dependencies between time steps. The core mathematical operations inside an LSTM unit are designed to address the problem of vanishing gradients and exploding gradients, which typically arise in training traditional RNNs (Recurrent Neural Networks).

An LSTM consists of three gates and a cell state that control how information is retained or discarded at each time step:

Forget Gate: Decides what information to forget from the previous time step.

Input Gate: Decides what new information to store in the memory cell.

Output Gate: Determines what information should be outputted based on the current input and previous cell state.

The operations of these gates are mathematically described as follows:

3. Forget Gate (
𝑓
𝑡
f 
t
​
 )
The forget gate determines how much of the previous memory state should be carried forward to the next time step. It takes the current input (
𝑥
𝑡
x 
t
​
 ) and the previous hidden state (
ℎ
𝑡
−
1
h 
t−1
​
 ):

𝑓
𝑡
=
𝜎
(
𝑊
𝑓
⋅
[
ℎ
𝑡
−
1
,
𝑥
𝑡
]
+
𝑏
𝑓
)
f 
t
​
 =σ(W 
f
​
 ⋅[h 
t−1
​
 ,x 
t
​
 ]+b 
f
​
 )
Where:

𝑓
𝑡
f 
t
​
  is the forget gate output (a value between 0 and 1, determined by the sigmoid activation function 
𝜎
σ),

𝑊
𝑓
W 
f
​
  is the weight matrix for the forget gate,

[
ℎ
𝑡
−
1
,
𝑥
𝑡
]
[h 
t−1
​
 ,x 
t
​
 ] is the concatenation of the previous hidden state and the current input,

𝑏
𝑓
b 
f
​
  is the bias term.

4. Input Gate (
𝑖
𝑡
i 
t
​
 ) and Candidate Cell State (
𝐶
~
𝑡
C
~
  
t
​
 )
The input gate decides what information to update in the memory cell, while the candidate cell state determines the new information to be stored.

𝑖
𝑡
=
𝜎
(
𝑊
𝑖
⋅
[
ℎ
𝑡
−
1
,
𝑥
𝑡
]
+
𝑏
𝑖
)
i 
t
​
 =σ(W 
i
​
 ⋅[h 
t−1
​
 ,x 
t
​
 ]+b 
i
​
 )
𝐶
~
𝑡
=
tanh
⁡
(
𝑊
𝐶
⋅
[
ℎ
𝑡
−
1
,
𝑥
𝑡
]
+
𝑏
𝐶
)
C
~
  
t
​
 =tanh(W 
C
​
 ⋅[h 
t−1
​
 ,x 
t
​
 ]+b 
C
​
 )
Where:

𝑖
𝑡
i 
t
​
  is the input gate output (sigmoid function),

𝐶
~
𝑡
C
~
  
t
​
  is the candidate cell state (tanh activation function, which ensures the values are in the range [-1, 1]),

𝑊
𝑖
W 
i
​
 , 
𝑊
𝐶
W 
C
​
  are weight matrices, and 
𝑏
𝑖
b 
i
​
 , 
𝑏
𝐶
b 
C
​
  are bias terms.

5. Update the Cell State (
𝐶
𝑡
C 
t
​
 )
Now, the cell state is updated by combining the previous cell state and the new information from the input gate and candidate cell state:

𝐶
𝑡
=
𝑓
𝑡
⋅
𝐶
𝑡
−
1
+
𝑖
𝑡
⋅
𝐶
~
𝑡
C 
t
​
 =f 
t
​
 ⋅C 
t−1
​
 +i 
t
​
 ⋅ 
C
~
  
t
​
 
Where:

𝐶
𝑡
−
1
C 
t−1
​
  is the previous memory cell state,

𝐶
𝑡
C 
t
​
  is the updated memory cell state at time step 
𝑡
t,

𝑓
𝑡
f 
t
​
  determines how much of the old memory to keep,

𝑖
𝑡
i 
t
​
  determines how much of the new information to store in the memory.

6. Output Gate (
𝑜
𝑡
o 
t
​
 ) and Hidden State (
ℎ
𝑡
h 
t
​
 )
Finally, the output gate controls what part of the memory cell state is passed on as the output of the LSTM at each time step.

𝑜
𝑡
=
𝜎
(
𝑊
𝑜
⋅
[
ℎ
𝑡
−
1
,
𝑥
𝑡
]
+
𝑏
𝑜
)
o 
t
​
 =σ(W 
o
​
 ⋅[h 
t−1
​
 ,x 
t
​
 ]+b 
o
​
 )
ℎ
𝑡
=
𝑜
𝑡
⋅
tanh
⁡
(
𝐶
𝑡
)
h 
t
​
 =o 
t
​
 ⋅tanh(C 
t
​
 )
Where:

𝑜
𝑡
o 
t
​
  is the output gate,

ℎ
𝑡
h 
t
​
  is the hidden state (which is used as the input to the next time step and also as the output of the LSTM cell at time step 
𝑡
t).

7. Loss Function (Categorical Cross-Entropy)
For training the LSTM model, we use a categorical cross-entropy loss function. This is commonly used for multi-class classification tasks (like predicting the sales distribution across different products). It compares the model's predicted probabilities with the true labels.

Loss
=
−
∑
𝑖
𝑦
𝑖
⋅
log
⁡
(
𝑝
𝑖
)
Loss=− 
i
∑
​
 y 
i
​
 ⋅log(p 
i
​
 )
Where:

𝑦
𝑖
y 
i
​
  is the true probability distribution (usually one-hot encoded),

𝑝
𝑖
p 
i
​
  is the predicted probability distribution (output of the softmax activation in the output layer).

8. Optimization (Backpropagation Through Time)
Once the loss is computed, backpropagation through time (BPTT) is used to adjust the weights of the LSTM model. This involves:

Calculating the gradients of the loss with respect to the weights.

Using the Adam optimizer (or any gradient descent algorithm) to update the weights iteratively.

For each weight 
𝑊
W in the model, the update rule is:

𝑊
=
𝑊
−
𝜂
⋅
∂
Loss
∂
𝑊
W=W−η⋅ 
∂W
∂Loss
​
 
Where:

𝜂
η is the learning rate,

∂
Loss
∂
𝑊
∂W
∂Loss
​
  is the gradient of the loss with respect to the weight 
𝑊
W.

This process is repeated for each training batch until the model converges (i.e., the loss stabilizes).

Summary of Key Mathematical Concepts:
LSTM Gates: The forget, input, and output gates determine how the model retains and discards information at each time step. These gates are controlled by sigmoid activations, and the memory is updated using a combination of past information and new input.

Cell State: The cell state acts as the "memory" of the LSTM, storing relevant long-term dependencies across time steps.

Activation Functions: Sigmoid and tanh functions are used to introduce non-linearities, enabling the model to learn complex patterns.

Loss Function: The categorical cross-entropy function is used to compare the predicted probability distribution with the true distribution.

Backpropagation Through Time (BPTT): This method is used to train the LSTM by updating the weights based on the gradients of the loss.

By combining these components, the LSTM network is capable of learning long-term dependencies in sequential data, making it a powerful tool for time series forecasting, such as predicting future sales distributions in your case.






You said:
explain how it address vanishing gradiant and exploding gradiant


ChatGPT said:
The vanishing gradient and exploding gradient problems are major challenges that arise when training deep neural networks, particularly Recurrent Neural Networks (RNNs), such as LSTMs. These problems occur during backpropagation (the process of updating the weights of the network) and can prevent the model from effectively learning long-term dependencies in sequential data.

1. Vanishing Gradient Problem
The vanishing gradient problem occurs when the gradients of the loss function become very small as they are propagated backward through the network. As a result, the model's weights are updated very little, making it difficult for the model to learn. This is especially problematic for deep networks or long sequences, where the gradients can shrink to nearly zero.

Why does it happen?
In an RNN, the hidden states at each time step are dependent on the previous hidden states, and gradients are propagated backward through each time step during training. If the activation functions used in the network (like sigmoid or tanh) have very small derivatives, the gradients shrink exponentially as they move backward through time.

For example:

Sigmoid and tanh derivatives are bounded in the range [0, 1], so they compress the gradients.

During backpropagation, the product of many small values (derivatives) can lead to an exponentially smaller gradient, causing the weights to update very slowly.

This leads to vanishing gradients, where the network cannot effectively learn long-term dependencies.

2. Exploding Gradient Problem
On the other hand, the exploding gradient problem occurs when the gradients grow exponentially large as they are propagated backward through the network. This leads to excessively large weight updates, causing the model to become unstable and fail to converge.

Why does it happen?
Exploding gradients typically arise when the network has very large weights or when the derivatives of the activations are large, causing gradients to grow rapidly during backpropagation. As a result, the gradients can become very large, leading to instability during training.

3. How LSTM Addresses These Problems
LSTM networks are specifically designed to overcome both vanishing gradients and exploding gradients by introducing several mechanisms to control how information flows through the network. These mechanisms help maintain a stable flow of gradients during training.

Key Features of LSTM that Address Vanishing and Exploding Gradients:
a. Cell State (Memory)
The cell state 
𝐶
𝑡
C 
t
​
  in an LSTM is a key component that helps mitigate both the vanishing and exploding gradient problems. The cell state is a long-term memory that runs through the entire sequence and is only updated by a gate mechanism (forget, input, and output gates). This allows the LSTM to retain information over long sequences without suffering from vanishing gradients.

Preserving information: Unlike traditional RNNs, the LSTM's cell state is not subject to multiplication by small or large values at each time step. The forget gate allows the LSTM to decide how much of the past information should be kept, and the input gate controls how much new information should be added. This way, information can be retained for long sequences without being diluted (vanishing gradients) or blown out of proportion (exploding gradients).

The cell state is updated using a simple addition operation:

𝐶
𝑡
=
𝑓
𝑡
⋅
𝐶
𝑡
−
1
+
𝑖
𝑡
⋅
𝐶
~
𝑡
C 
t
​
 =f 
t
​
 ⋅C 
t−1
​
 +i 
t
​
 ⋅ 
C
~
  
t
​
 
The multiplication by 
𝑓
𝑡
f 
t
​
  and 
𝑖
𝑡
i 
t
​
  ensures that the gradient flow is much more stable than in traditional RNNs.

b. Gradients Flowing Through Gates
The gates in an LSTM are controlled by sigmoid and tanh activation functions, but the key is how the gates are designed to prevent gradients from vanishing or exploding:

Forget Gate: The forget gate 
𝑓
𝑡
f 
t
​
  controls how much of the previous cell state is passed along to the next time step. It uses the sigmoid function, which outputs values between 0 and 1. This helps in controlling the amount of information to forget, preventing the gradient from becoming too small.

Input Gate and Candidate Cell State: The input gate 
𝑖
𝑡
i 
t
​
  and candidate cell state 
𝐶
~
𝑡
C
~
  
t
​
  also control the flow of information into the cell state. The input gate uses a sigmoid function to decide how much new information to store, and the candidate cell state is passed through a tanh activation, which ensures that the updates to the cell state are bounded within a reasonable range. This prevents the exploding gradient problem by ensuring that the cell state updates remain stable.

Output Gate: The output gate 
𝑜
𝑡
o 
t
​
  is responsible for outputting the hidden state 
ℎ
𝑡
h 
t
​
 . The output gate modulates how much of the cell state should be exposed to the next time step and to the output layer. This prevents the network from producing excessively large outputs, which helps mitigate exploding gradients.

c. Gating Mechanisms Prevent Overflows
The gates (forget, input, and output) in the LSTM network control the flow of information, ensuring that only relevant information is passed through while irrelevant data is discarded or “forgotten.” The gates are designed to prevent the cell state from becoming too large or too small, thus avoiding exploding or vanishing gradients.

Forget Gate: It multiplies the previous memory 
𝐶
𝑡
−
1
C 
t−1
​
  by a value between 0 and 1. This allows the LSTM to forget irrelevant information without allowing the gradient to vanish or explode.

Input Gate and Candidate State: The input gate controls the information added to the memory and helps limit the amount of new information that enters the system, avoiding the potential for large weight updates that could lead to exploding gradients.

d. Stable Gradient Flow
LSTMs address the vanishing gradient problem by allowing gradients to flow through time with minimal attenuation. The multiplicative structure of the LSTM (using gates that are based on activations like sigmoid) helps in stabilizing the gradient as it flows back through the network. The use of the cell state enables the LSTM to retain information over long sequences, without diminishing or over-amplifying the gradient.

Mathematical Explanation of Stable Gradient Flow:
Gradient of the Cell State:
When backpropagating through time in an LSTM, the gradients with respect to the cell state 
𝐶
𝑡
C 
t
​
  are less likely to vanish because the cell state gradient is not multiplied by the weights at each step. Instead, it is modulated by the forget and input gates, which provide a controlled flow of gradients.

∂
𝐿
∂
𝐶
𝑡
=
∂
𝐿
∂
𝐶
𝑡
⋅
𝑓
𝑡
∂C 
t
​
 
∂L
​
 = 
∂C 
t
​
 
∂L
​
 ⋅f 
t
​
 
The term 
𝑓
𝑡
f 
t
​
  (forget gate) controls how much of the previous cell state to keep, while 
𝑖
𝑡
i 
t
​
  (input gate) controls how much new information is stored. This prevents gradients from either decaying to zero (vanishing) or growing too large (exploding).

Summary of How LSTMs Overcome These Problems
Vanishing Gradients: LSTMs prevent the vanishing gradient problem by maintaining a long-term memory cell state (
𝐶
𝑡
C 
t
​
 ) that is updated gradually through the gates (forget, input). The gates allow gradients to flow through the network without shrinking exponentially.

Exploding Gradients: The gating mechanisms in the LSTM control how much information is retained or updated, preventing the gradients from growing excessively large. The use of sigmoid and tanh activations further helps to limit the size of the gradients.

Overall, the LSTM’s gating mechanisms provide a stable flow of information, allowing it to learn long-term dependencies in sequences without being affected by vanishing or exploding gradients.
