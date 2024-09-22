### Developed By : KANISHKAR M
### Register No. : 212222240044
### Date : 

# Ex.No: 6               HOLT WINTERS METHOD


### AIM:

To create and implement Holt Winter's Method Model using python for AirTemp (C) in Water Quality.



### ALGORITHM:

1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'Date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them
7. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-Winters model to the entire dataset and make future predictions
8. You plot the original sales data and the predictions

### PROGRAM:

#### Import Neccesary Libraries
```py
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
```

#### Load the dataset

```py


# Load the dataset
data = pd.read_csv('/content/waterquality.csv', index_col='Date', parse_dates=True)


```

#### Resample the data to a monthly frequency (beginning of the month)

```py
data = data['AirTemp (C)'].resample('MS').mean()


```
#### Scaling the Data using MinMaxScaler 


```py
scaler = MinMaxScaler()
data_scaled = pd.Series(scaler.fit_transform(data.values.reshape(-1, 1)).flatten(), index=data.index)
```

#### Split into training and testing sets (80% train, 20% test)

```py
train_data = data_scaled[:int(len(data_scaled) * 0.8)]
test_data = data_scaled[int(len(data_scaled) * 0.8):]
```

#### Fitting the model
```py

fitted_model_add = ExponentialSmoothing(
    train_data, trend='add', seasonal='add', seasonal_periods=12
).fit()


```

#### Forecast and evaluate

```py

# Forecast and evaluate
test_predictions_add = fitted_model_add.forecast(len(test_data))
# Ensure fillna is applied to the entire Series to handle NaNs, not just the forecast
test_predictions_add = test_predictions_add.fillna(method='ffill')
# Convert the Series to numeric, coercing NaNs to NaNs
test_predictions_add = pd.to_numeric(test_predictions_add, errors='coerce')
```



#### Plot predictions


```py

# Plot predictions
plt.figure(figsize=(12, 8))
plt.plot(train_data, label='TRAIN', color='red')
plt.plot(test_data, label='TEST', color='yellow')
plt.plot(test_predictions_add, label='PREDICTION', color='black')
plt.title('Train, Test, and Additive Holt-Winters Predictions')
plt.legend(loc='best')
plt.show()
```

#### Forecast future values

```py
final_model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=12).fit()

forecast_predictions = final_model.forecast(steps=12)
```

```py
data.plot(figsize=(12, 8), legend=True, label='AirTemp (C)')
forecast_predictions.plot(legend=True, label='Forecasted AirTemp (C)')
plt.title('AirTemp (C) Forecast')
plt.show()
```



### OUTPUT:

#### TEST PREDICTION

![image](https://github.com/user-attachments/assets/214a789a-ccab-47ec-a3e6-50ac5391dc33)


#### FINAL PREDICTION

![image](https://github.com/user-attachments/assets/49097f3b-cce1-41ca-b579-ca6b3329cf1e)

### RESULT:

#### Thus the program run successfully based on the Holt Winters Method model.
