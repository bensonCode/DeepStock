import pandas as pd

# 讀csv進來變成dataframe格式

df1 = pd.DataFrame()
df1 = pd.read_csv("./calData1415.csv", encoding="utf-8")

df2 = pd.DataFrame()
df2 = pd.read_csv("./calData1415.csv", encoding="utf-8")

# df3 = pd.DataFrame()
# df3 = pd.read_csv("./tx/2016tx.csv", encoding="utf-8")
#
# df4 = pd.DataFrame()
# df4 = pd.read_csv("./tx/2017tx.csv", encoding="utf-8")

df_merge1 = pd.DataFrame()
df_merge2 = pd.DataFrame()
df_merge3 = pd.DataFrame()
df_merge1 = pd.concat([df1, df2], axis=0,  ignore_index=True)
# df_merge2 = pd.concat([df3, df4],  ignore_index=True)
print(df_merge1.tail())
# print(df_merge2.tail())

# df_merge3 = pd.concat([df_merge1, df_merge2], axis=0, ignore_index=True)
# print(df_merge3.tail())

df_merge1.to_csv("calData1417.csv")

print(df_merge1.head(250))
