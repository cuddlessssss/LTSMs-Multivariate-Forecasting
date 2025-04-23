#source tf_env/bin/activate
#python test_packages.py IN TERMINAL

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
import matplotlib
import matplotlib.pyplot as plt


# Test versions
print("✅ pandas:", pd.__version__)
print("✅ numpy:", np.__version__)
print("✅ tensorflow:", tf.__version__)
print("✅ scikit-learn:", LinearRegression)
print("✅ matplotlib:", matplotlib.__version__) 
print("✅ GPU available in TensorFlow:", tf.config.list_physical_devices('GPU'))

# Simple computation test
df = pd.DataFrame({
    "x": np.arange(10),
    "y": np.arange(10) * 2 + 1
})

model = LinearRegression()
model.fit(df[["x"]], df["y"])
predicted = model.predict(df[["x"]])

plt.plot(df["x"], df["y"], label="Actual")
plt.plot(df["x"], predicted, label="Predicted", linestyle="--")
plt.legend()
plt.title("Test Linear Regression")
plt.show()


#pip install --force-reinstall <package-name>