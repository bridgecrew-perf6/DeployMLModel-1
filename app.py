from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import flask
import matplotlib.pyplot as plt

app = Flask(__name__)

"""Calculate the moving average of the series"""
def moving_average_forecast(series, window_size):
  forecast = []
  for time in range(len(series) - window_size):
    forecast.append(series[time:time + window_size].mean(axis = 0))
  return np.array(forecast)


def predict_sales(dataset, seasonality=12, window_size=12, duration=1):
    dataset_length = len(dataset)
    if(dataset_length <= seasonality):
        meanValue = dataset.mean(axis = 0)
        return np.full((duration, 1), meanValue, dtype=int)

    last_season_semi_window_size = 3
    index_from = dataset_length - duration - seasonality - (last_season_semi_window_size)
    index_to = -seasonality + (last_season_semi_window_size)
    diff_series = (dataset[seasonality:] - dataset[:-seasonality])
    diff_moving_avg = moving_average_forecast(diff_series, window_size)[dataset_length - seasonality - window_size-duration:]
    prediction = moving_average_forecast(dataset[index_from: index_to], 2*last_season_semi_window_size)  + diff_moving_avg
    prediction = np.clip(prediction, 0, max_value).astype(int)
    return prediction


dataset = pd.read_csv('PreprocessedOrders.csv', index_col='Period')
dataset = dataset.drop(['Unnamed: 0'], axis = 1)
rows = len(dataset)
dataset = dataset.to_numpy()
time = np.arange(rows, dtype="float32")

#Splitting the data for traing and validation
split_time = int(rows * 0.7)
time_train = time[:split_time]
dataset_train = dataset[:split_time]
time_validate = time[split_time:]
dataset_valid = dataset[split_time:]
window_size = 12
max_value = 100000

@app.route('/')
def predict():
    products = 6
    stores = 1
    predictions = predict_sales(dataset)

    output = [{"pKey":"Product1", "productName": "Diet Pepsi", "quantity": predictions[0][0]},{"pKey":"Product2", "productName": "Frito Lays", "quantity": predictions[0][1]},{"pKey":"Product3", "productName": "Quaker Oats", "quantity": predictions[0][2]},{"pKey":"Product4", "productName": "Ruffles", "quantity": predictions[0][3]}, {"pKey":"Product5", "productName": "Tropican Orange", "quantity": predictions[0][4]},{"pKey":"Product6", "productName": "Mountain Dew", "quantity": predictions[0][5]}]
    print(output)
    return str(output)


if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='localhost', port=8081)