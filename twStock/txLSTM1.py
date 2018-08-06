#from twStock.commonUtil import testfun
#import twStock.commonUtil
import pandas as pd

from twStock import commonUtil
#讀csv進來變成dataframe格式
df = pd.read_csv("./data/tx/2017tx.csv", encoding="utf-8")
#轉換df所需欄位為-1到1的格式，最佳化時計算所需
df = commonUtil.normalize_data(df)

#訓練資料比例70%、測試資料30%
test_data_rate = 0.7
x_train, x_test = commonUtil.prepare_data(df, test_data_rate)



