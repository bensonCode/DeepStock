import pandas as pd

#交易日期	契約	到期月份(週別)	開盤價	最高價	最低價	收盤價	漲跌價	漲跌%	成交量	結算價	未沖銷契約數	最後最佳買價	最後最佳賣價	歷史最高價	歷史最低價
#date	contract	contract_m	open	high	low	close	fluctuation	fluctuation_percent	volume	settlement	open_interest	last_best_buy	last_bset_sell	h_high	h_low

df = pd.read_csv("./data/2017_fut.csv", encoding="Big5",
                 dtype={"settlement": object, "open_interest": object})

# df.rename(columns={"契約":'date'}, inplace=True)

df["date"]=pd.to_datetime(df.date)

tx = df.loc[df["contract"] == "TX"]
tx = tx.sort_values(by=['date', 'contract_m'])
tx = tx.groupby("date").head(1)


print(tx.head(100))



#df.info()
#print(df.size)

#print(df.head(5))

#print(tx.size)
#print(tx.head())
