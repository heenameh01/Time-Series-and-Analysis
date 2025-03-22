#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Load the dataset
file_path = "E:/221501043/daily-minimum-temperatures-in-me.csv"
df = pd.read_csv(file_path)

# Convert 'Date' column to datetime format and set it as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Rename the temperature column for easier access
df.rename(columns={'Daily minimum temperatures': 'Temperature'}, inplace=True)

# Convert Temperature column to numeric, forcing errors to NaN and then dropping them
df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
df.dropna(inplace=True)

# Perform Augmented Dickey-Fuller test
adf_test = adfuller(df['Temperature'])
adf_result = {
    "ADF Statistic": adf_test[0],
    "p-value": adf_test[1],
    "Critical Values": adf_test[4]
}

# Print ADF test results
print("ADF Statistic:", adf_result["ADF Statistic"])
print("p-value:", adf_result["p-value"])
for key, value in adf_result["Critical Values"].items():
    print(f"Critical Value ({key}): {value}")

# Plot scatter graph of time series data
plt.figure(figsize=(10, 5))
plt.scatter(df.index, df['Temperature'], s=10, alpha=0.5)
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Scatter Plot of Daily Minimum Temperatures')
plt.xticks(rotation=45)
plt.show()


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import numpy as np

# Load the dataset
file_path = "E:/221501043/daily-minimum-temperatures-in-me.csv"
df = pd.read_csv(file_path)

# Convert 'Date' column to datetime format and set it as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Rename the temperature column for easier access
df.rename(columns={'Daily minimum temperatures': 'Temperature'}, inplace=True)

# Convert Temperature column to numeric, forcing errors to NaN and then dropping them
df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
df.dropna(inplace=True)

# Perform Augmented Dickey-Fuller test
adf_test = adfuller(df['Temperature'])
adf_result = {
    "ADF Statistic": adf_test[0],
    "p-value": adf_test[1],
    "Critical Values": adf_test[4]
}

# Print ADF test results
print("ADF Statistic:", adf_result["ADF Statistic"])
print("p-value:", adf_result["p-value"])
for key, value in adf_result["Critical Values"].items():
    print(f"Critical Value ({key}): {value}")

# Identify stationary points (where changes are minimal)
df['Diff'] = df['Temperature'].diff().abs()
threshold = df['Diff'].quantile(0.1)  # Define threshold as lower 10% of changes
stationary_points = df[df['Diff'] < threshold]

# Plot scatter graph of time series data
plt.figure(figsize=(10, 5))
plt.scatter(df.index, df['Temperature'], s=10, alpha=0.5, label='Temperature')
plt.scatter(stationary_points.index, stationary_points['Temperature'], color='red', s=15, label='Stationary Points')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Scatter Plot of Daily Minimum Temperatures with Stationary Points')
plt.xticks(rotation=45)
plt.legend()
plt.show()


# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import numpy as np

# Load the dataset
file_path = "E:/221501043/daily-minimum-temperatures-in-me.csv"
df = pd.read_csv(file_path)

# Convert 'Date' column to datetime format and set it as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Rename the temperature column for easier access
df.rename(columns={'Daily minimum temperatures': 'Temperature'}, inplace=True)

# Convert Temperature column to numeric, forcing errors to NaN and then dropping them
df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
df.dropna(inplace=True)

# Perform Augmented Dickey-Fuller test
adf_test = adfuller(df['Temperature'])
adf_result = {
    "ADF Statistic": adf_test[0],
    "p-value": adf_test[1],
    "Critical Values": adf_test[4]
}

# Print ADF test results
print("ADF Statistic:", adf_result["ADF Statistic"])
print("p-value:", adf_result["p-value"])
for key, value in adf_result["Critical Values"].items():
    print(f"Critical Value ({key}): {value}")

# Identify stationary points (where changes are minimal)
df['Diff'] = df['Temperature'].diff().abs()
threshold = df['Diff'].quantile(0.1)  # Define threshold as lower 10% of changes
stationary_points = df[df['Diff'] < threshold]

# Estimate and eliminate trend - Aggregation using rolling mean
df['Rolling Mean'] = df['Temperature'].rolling(window=30).mean()

# Estimate and eliminate trend - Smoothing using exponential weighted moving average (EWMA)
df['EWMA'] = df['Temperature'].ewm(span=30, adjust=False).mean()

# Plot original data with trend estimations
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Temperature'], label='Original Data', alpha=0.5)
plt.plot(df.index, df['Rolling Mean'], label='Rolling Mean (30 days)', color='red')
plt.plot(df.index, df['EWMA'], label='EWMA (30 days)', color='green')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Trend Estimation and Smoothing in Time Series')
plt.legend()
plt.show()


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the dataset
file_path = "E:/221501043/daily-minimum-temperatures-in-me.csv"
df = pd.read_csv(file_path)

# Convert 'Date' column to datetime format and set it as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Rename the temperature column for easier access
df.rename(columns={'Daily minimum temperatures': 'Temperature'}, inplace=True)

# Convert Temperature column to numeric, forcing errors to NaN and then dropping them
df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
df.dropna(inplace=True)

# Perform Augmented Dickey-Fuller test
adf_test = adfuller(df['Temperature'])
adf_result = {
    "ADF Statistic": adf_test[0],
    "p-value": adf_test[1],
    "Critical Values": adf_test[4]
}

# Print ADF test results
print("ADF Statistic:", adf_result["ADF Statistic"])
print("p-value:", adf_result["p-value"])
for key, value in adf_result["Critical Values"].items():
    print(f"Critical Value ({key}): {value}")

# Identify stationary points (where changes are minimal)
df['Diff'] = df['Temperature'].diff().abs()
threshold = df['Diff'].quantile(0.1)  # Define threshold as lower 10% of changes
stationary_points = df[df['Diff'] < threshold]

# Estimate and eliminate trend - Aggregation using rolling mean
df['Rolling Mean'] = df['Temperature'].rolling(window=30).mean()

# Estimate and eliminate trend - Smoothing using exponential weighted moving average (EWMA)
df['EWMA'] = df['Temperature'].ewm(span=30, adjust=False).mean()

# Apply Linear Regression
X = np.array((df.index - df.index.min()).days).reshape(-1, 1)  # Convert dates to numerical values
y = df['Temperature'].values
model = LinearRegression()
model.fit(X, y)
df['Trend'] = model.predict(X)

# Plot original data, trend estimations, and linear regression
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Temperature'], label='Original Data', alpha=0.5)
plt.plot(df.index, df['Rolling Mean'], label='Rolling Mean (30 days)', color='red')
plt.plot(df.index, df['EWMA'], label='EWMA (30 days)', color='green')
plt.plot(df.index, df['Trend'], label='Linear Regression Trend', color='blue', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Trend Estimation, Smoothing, and Linear Regression in Time Series')
plt.legend()
plt.show()


# In[ ]:




