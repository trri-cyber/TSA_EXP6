# Ex.No: 6               HOLT WINTERS METHOD
### Date: 13-05-2026



### AIM:

Implement the program based on the Holt Winters Method model.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

data = pd.read_csv("/content/global_inflation_post_covid.csv")

data = data[data['country'] == 'USA']

data['date'] = pd.to_datetime(data['date'])

data = data.groupby('date')['food_price_index'].mean().reset_index()

data.set_index('date', inplace=True)

print(data.head())

data_monthly = data.resample('MS').mean()

print(data_monthly.head())

data_monthly.plot(figsize=(10,5))
plt.title("USA Food Price Index")
plt.xlabel("Date")
plt.ylabel("Food Price Index")
plt.show()

scaler = MinMaxScaler()

scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1,1)).flatten(),
    index=data_monthly.index
)

scaled_data.plot(figsize=(10,5))
plt.title("Scaled Data")
plt.show()

decomposition = seasonal_decompose(data_monthly, model='additive')

decomposition.plot()
plt.show()

scaled_data = scaled_data + 1

train_data = scaled_data[:int(len(scaled_data) * 0.8)]

test_data = scaled_data[int(len(scaled_data) * 0.8):]

model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

test_predictions_add = model_add.forecast(steps=len(test_data))

ax = train_data.plot(figsize=(10,5))

test_predictions_add.plot(ax=ax)

test_data.plot(ax=ax)

ax.legend(["train_data", "test_predictions_add", "test_data"])

ax.set_title("Visual Evaluation")

plt.show()

rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))

print("RMSE :", rmse)

print("Variance :", np.sqrt(scaled_data.var()))

print("Mean :", scaled_data.mean())

final_model = ExponentialSmoothing(
    scaled_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

final_predictions = final_model.forecast(
    steps=int(len(data_monthly) / 4)
)

ax = scaled_data.plot(figsize=(10,5))

final_predictions.plot(ax=ax)

ax.legend(["Original Data", "Future Predictions"])

ax.set_xlabel("Months")

ax.set_ylabel("Food Price Index")

ax.set_title("Future Prediction")

plt.show()
```
### OUTPUT:

<img width="848" height="467" alt="image" src="https://github.com/user-attachments/assets/88df0365-8816-40c4-9813-3bb465ebf126" />

TEST_PREDICTION

<img width="824" height="451" alt="image" src="https://github.com/user-attachments/assets/202ef1d8-2d00-4447-8001-8ee9a0271695" />

FINAL_PREDICTION

<img width="839" height="462" alt="image" src="https://github.com/user-attachments/assets/a1f6fd63-c09d-45af-9026-e5323bd1571e" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
