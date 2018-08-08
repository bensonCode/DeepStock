from sklearn import preprocessing
import numpy as np
import pandas as pd


def prepare_data(df, test_data_rate):
    print(len(df.index))
    test_data_size = int(len(df.index) * test_data_rate)

    x_train, x_test = df[0:test_data_size], df[test_data_size:]
    print(x_train.shape)
    print(x_test.shape)

    # x_train = df[:, 0:test_data_size]
    print("test_data_size", test_data_size, df.shape)
    print(x_train.tail())
    print(x_test.head())
    return x_train, x_test


def normalize_data(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1, 1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1, 1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1, 1))
    df['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1, 1))
    df['fluctuation'] = min_max_scaler.fit_transform(df['fluctuation'].values.reshape(-1, 1))
    df_dict = {
                "open": df['open'],
                "high": df['high'],
                "low": df['low'],
                "volume": df['volume'],
                "fluctuation": df['fluctuation']
    }
    dframe = pd.DataFrame(df_dict)
    return dframe


# Build Training Data
# 輸入X_train: 利用前30天的Open, High, Low, Close, Adj Close, Volume, month, year, date, day作為Features，shape為(30, 10)
# 輸出Y_train: 利用未來5天的Adj Close作為Features，shape為(5,1)
def build_train(train, pastDay=30, futureDay=5):
    x_train, y_train = [], []
    for i in range(train.shape[0] - futureDay - pastDay):
        x_train.append(np.array(train.iloc[i:i + pastDay]))
        y_train.append(np.array(train.iloc[i + pastDay:i + pastDay + futureDay]["fluctuation"]))
    return np.array(x_train), np.array(y_train)


# 將Training Data取一部份當作Validation Data
def split_data(X, Y, rate):
    X_train = X[int(X.shape[0] * rate):]
    Y_train = Y[int(Y.shape[0] * rate):]
    X_val = X[:int(X.shape[0] * rate)]
    Y_val = Y[:int(Y.shape[0] * rate)]
    return X_train, Y_train, X_val, Y_val
