import requests
import os
import pandas as pd
import numpy as np
import datetime
import dateutil.relativedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from keras.models import Sequential
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error

headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36"}

step = 12  # amount of hours of previous data used for prediction


def get_currency_pairs_list():
    result = requests.get(
        'https://api.binance.com/api/v1/exchangeInfo', headers=headers)
    result = result.json()
    currency_pairs_list = []
    for symbol in result['symbols']:
        currency_pairs_list.append(symbol['symbol'])
    return np.unique(currency_pairs_list)


def get_kline(currency_pair, limit=1000):
    start = datetime.datetime.now()
    interval = '1h'
    start = start - dateutil.relativedelta.relativedelta(months=1, days=10)
    # https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data
    columns = ['open_time', 'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'close_time',
               'quote_asset_vol', 'num_trades', 'taker_buy_base_vol', 'taker_buy_quote_vol', 'unused_field']
    start = int(datetime.datetime.timestamp(pd.to_datetime(start))*1000)
    url = f'https://www.binance.com/api/v3/klines?symbol={currency_pair}&interval={interval}&limit={limit}&startTime={start}'
    data = pd.DataFrame(requests.get(
        url, headers=headers).json(), columns=columns, dtype=np.float32)
    data.open_time = [pd.to_datetime(x, unit='ms').strftime(
        '%Y-%m-%d %H:%M:%S') for x in data.open_time]
    data['pair'] = currency_pair
    return data


def download_dataset():
    currency_pairs_list = get_currency_pairs_list()
    df = get_kline(currency_pairs_list[0])
    for pair in currency_pairs_list[1:]:
        res = get_kline(pair)
        df = pd.concat([df, res])
        df.drop_duplicates(inplace=True)
    print(df)
    df.to_csv('./data/dataset.csv')


def get_LTSM_model(input_shape):
    model = Sequential([
        layers.LSTM(units=50, activation='relu',  return_sequences=True,
                    input_shape=input_shape),
        layers.Dropout(0.3),
        layers.LSTM(units=50, return_sequences=True, activation='relu'),
        layers.Dropout(0.3),
        layers.LSTM(units=50, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(units=1),
    ])
    return model


def to_supervised_dataset(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


def date_parser(x):
    return datetime.datetime.strptime(x,
                                      '%Y-%m-%d %H:%M:%S')


def scale_data(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    arr = np.array(new_row)
    arr = arr.reshape(1, len(arr))
    inverted = scaler.inverse_transform(arr)
    return inverted[0, -1]


def predict(model, X):
    X = X.reshape(1, 1, step)
    prediction = model.predict(X)
    return prediction[0, 0]


def predict_unscaled(model, X):
    X_scaled = scaler.transform(np.append(X, [None]).reshape(1, step + 1))
    X_scaled = X_scaled[0, 0:-1]
    X_pred = X_scaled.reshape(1, 1, step)
    prediction = predict(model, X_pred)
    prediction = invert_scale(scaler, X_scaled, prediction)
    return prediction


def predict_unscaled_batch(model, batch):
    batch = batch.reshape(1, step + 1)
    batch_scaled = scaler.transform(batch)
    X_scaled = batch_scaled[0, 0:-1]
    X_pred = X_scaled.reshape(1, 1, step)
    prediction = predict(model, X_pred)
    prediction = invert_scale(scaler, X_scaled, prediction)
    return prediction


if __name__ == '__main__':
    default_pair = 'ETHBTC'
    print(
        f'Please write the desired currency pair. \nThe default pair is {default_pair}. For the list of available pairs please visit https://www.binance.com .')
    pair = str(input())
    pair = pair.capitalize()
    if (not pair in get_currency_pairs_list()):
        print(
            f'Incorrect currency pair. \nExample of correct input: {default_pair}. \n Using the default pair: {default_pair}. \nPlease press Enter to continue...')
        pair = default_pair
        input()

    model = get_LTSM_model((1, step))
    model.summary()
    model.compile(
        optimizer='Adam',
        loss='mean_squared_error',
    )

    if (not os.path.exists('./data/dataset.csv')):
        download_dataset()

    df = pd.read_csv('./data/dataset.csv', header=0,
                     parse_dates=[1], index_col=0, date_parser=date_parser)
    pair_groups = df.groupby('pair')
    group = pair_groups.get_group(pair)
    group.index = group.open_time
    group = group.drop(columns='open_time')
    columns = ['close_price']
    group = group[columns]
    raw_values = group.values
    supervised_ds = to_supervised_dataset(raw_values, step)
    reduced_supervised_ds = supervised_ds.loc[(
        supervised_ds != 0.0).all(axis=1)]
    train, test = train_test_split(
        reduced_supervised_ds.values, random_state=42)
    scaler, train_scaled, test_scaled = scale_data(train, test)
    X, y = train_scaled[:, 0:-1], train_scaled[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])

    earlystopping = EarlyStopping(
        monitor='loss', patience=7, min_delta=0.0001)
    checkpoint = ModelCheckpoint(
        filepath='./tmp/checkpoint.hdf5',
        verbose=2,
        save_best_only=True,
        monitor='loss',
    )
    history = model.fit(
        X,
        y,
        epochs=100,
        batch_size=32,
        callbacks=[earlystopping, checkpoint],
    )

    predictions = list()
    expected = list()
    for i in range(len(test_scaled)):
        X = test_scaled[i, 0:-1]
        y = test[i, -1]
        expected.append(y)
        prediction = predict(model, X)
        predictions.append(prediction)

    mse = mean_squared_error(expected, predictions)

    predictions = list()
    expected = list()
    for i in range(step, len(raw_values) - 1):
        batch = raw_values[i-step:i + 1]
        prediction = predict_unscaled_batch(model, batch)
        predictions.append(prediction)
        y = raw_values[i + 1]
        expected.append(y)
    X = raw_values[-step:]
    prediction = predict_unscaled(model, X)
    future_prediction = prediction

    print('Test mean squared error: ', mse)
    print('Prediction for the next hour: ', future_prediction)
    future_date = group[-1:].index[0] + \
        dateutil.relativedelta.relativedelta(hours=1)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.plot(group[step + 1:].index, expected, label='real data')
    plt.plot(group[step + 1:].index, predictions, label='predictions')
    plt.scatter(future_date, future_prediction,
                label='prediction for the next hour', color='red', marker='x')
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.title(pair)
    plt.show()
