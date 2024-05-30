# Import Libraries
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
from prophet import Prophet

# Function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Function to print evaluation metrics
def print_evaluation_metrics(y_true, y_pred):
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"RMSE: {rmse(y_true, y_pred):.4f}")
    print(f"MAPE: {mean_absolute_percentage_error(y_true, y_pred):.4f}")

# Load and Preprocess the Data
# Load dataset
df = pd.read_csv('energy_consumption.csv', parse_dates=['Date'], index_col='Date')

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Normalize the data
df['Energy'] = (df['Energy'] - df['Energy'].min()) / (df['Energy'].max() - df['Energy'].min())

# Train-test split
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Prepare data for LSTM
def create_lstm_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back):
        X.append(data[i:(i+look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 5
train_lstm, test_lstm = train.values, test.values
X_train_lstm, y_train_lstm = create_lstm_dataset(train_lstm, look_back)
X_test_lstm, y_test_lstm = create_lstm_dataset(test_lstm, look_back)

# Reshape input to be [samples, time steps, features]
X_train_lstm = np.reshape(X_train_lstm, (X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], 1))

# ARIMA Model
# Fit ARIMA model
arima_model = ARIMA(train['Energy'], order=(5,1,0))
arima_model_fit = arima_model.fit(disp=0)

# Forecast
arima_forecast = arima_model_fit.forecast(steps=len(test))[0]

# Evaluate ARIMA model
print("ARIMA Model Evaluation:")
print_evaluation_metrics(test['Energy'], arima_forecast)

# LSTM Model
# Build LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train LSTM model
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=1, verbose=2)

# Forecast
lstm_forecast = lstm_model.predict(X_test_lstm)

# Evaluate LSTM model
print("LSTM Model Evaluation:")
print_evaluation_metrics(y_test_lstm, lstm_forecast)

# Prophet Model
# Prepare data for Prophet
prophet_df = df.reset_index().rename(columns={'Date': 'ds', 'Energy': 'y'})

# Fit Prophet model
prophet_model = Prophet()
prophet_model.fit(prophet_df[:train_size])

# Forecast
future = prophet_model.make_future_dataframe(periods=len(test))
prophet_forecast = prophet_model.predict(future)
prophet_forecast = prophet_forecast[['ds', 'yhat']].set_index('ds').iloc[-len(test):]

# Evaluate Prophet model
print("Prophet Model Evaluation:")
print_evaluation_metrics(test['Energy'], prophet_forecast['yhat'])

# Plot Results
# Plot actual vs forecast
plt.figure(figsize=(15, 8))
plt.plot(df.index, df['Energy'], label='Actual')
plt.plot(test.index, arima_forecast, label='ARIMA Forecast')
plt.plot(test.index, lstm_forecast, label='LSTM Forecast')
plt.plot(prophet_forecast.index, prophet_forecast['yhat'], label='Prophet Forecast')
plt.legend()
plt.show()
