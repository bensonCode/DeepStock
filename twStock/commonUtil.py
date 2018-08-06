from sklearn import preprocessing


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
    return df
