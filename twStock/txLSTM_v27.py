# from twStock.commonUtil import testfun
# import twStock.commonUtil
# from keras.layers.normalization import BatchNormalization
# from keras.optimizers import Adam
import os

import matplotlib.pyplot as plt2
import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM
from keras.models import Sequential

from twStock import commonUtil

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# %matplotlib inline


def buildOneToOneModel(shape):
    print("shape", shape[1], shape[2])
    model = Sequential()
    # model.add(Embedding(20, 15, input_length=300))
    model.add(LSTM(10, input_shape=(shape[1], shape[2])))
    # output shape: (1, 1)
    # model.add(TimeDistributed(Dense(3)))
    # or use model.add(Dense(1))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.summary()
    return model


# 讀csv進來變成dataframe格式
# df = pd.DataFrame()
# df = pd.read_csv("./data/tx/2017tx.csv", sep=r'\s*,\s*', encoding="utf-8")
df = pd.read_csv("./data/calData1417.csv", sep=r'\s*,\s*', encoding="utf-8")

print(df.columns.tolist())
# X_org, Y_org = commonUtil.build_train(df, 1, 1)
# X_org, Y_org, X_val_org, Y_val_org = commonUtil.split_data(X_org, Y_org, 0.2)

# 轉換df所需欄位為-1到1的格式，最佳化時計算所需

df = commonUtil.normalize_data(df)
df = df.replace("-", np.nan)
print(df.tail())

# 訓練資料比例70%、測試資料30%
# test_data_rate = 0.7
# x_train, x_test = commonUtil.prepare_data(df, test_data_rate)
# build Data, use last 30 days to predict next 5 days
# X_train, Y_train = commonUtil.build_train(df, 1, 1)
X_train, Y_train = commonUtil.build_train(df, 20, 1)

# split training data and validation data
X_train, Y_train, X_val, Y_val = commonUtil.split_data(X_train, Y_train, 0.2)
print(X_train.shape)

model = buildOneToOneModel(X_train.shape)
# callback = EarlyStopping(monitor="loss", patience=30, verbose=1, mode="auto")
# model.fit(X_train, Y_train, epochs=100, batch_size=1, validation_data=(X_val, Y_val), callbacks=[callback])
model.fit(X_train, Y_train, epochs=100, batch_size=16, validation_data=(X_val, Y_val), shuffle=False)

# 預測
# trainPredict = model.predict(X_train)
testPredict = model.predict(X_val)
#
# print("predict", trainPredict)
# print("predict222", testPredict)


# commonUtil.model_score(model, X_train, Y_train, X_val, Y_val)

# Test the model after training
# test_results = model.evaluate(X_train, y_test, verbose=False)
# print(f'Test results - Loss: {test_results[0]} - Accuracy: {100*test_results[1]}%')
#

plt2.plot(testPredict, color='red', label='Prediction')
plt2.plot(Y_val, color='blue', label='Actual')
plt2.legend(loc='best')
plt2.show()
plt2.savefig("tx.png")
