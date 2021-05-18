# monte carlo를 위한 종목데이터 추출 함수

import pandas as pd
import numpy as np
import pandas_datareader.data as web

start = '20200101'
end = '20210430'

TICKER = ['AAPL', 'TSLA', 'MSFT', 'AMZN']
dfs = web.DataReader(TICKER[0], 'yahoo', start, end)
dfs.reset_index(inplace=True)
dfs.set_index("Date", inplace=True)
dfs['Return'] = (dfs['Close'] / dfs['Close'].shift(1)) - 1
dfs['Return(cum)'] = (1 + dfs['Return']).cumprod()
dfs = dfs.dropna()
dfs.loc[:, 'TICKER'] = TICKER[0]
df = dfs
for i in range(1, len(TICKER)):
    dfs = web.DataReader(TICKER[i], 'yahoo', start, end)
    dfs.reset_index(inplace=True)
    dfs.set_index("Date", inplace=True)
    dfs['Return'] = (dfs['Close'] / dfs['Close'].shift(1)) - 1
    dfs['Return(cum)'] = (1 + dfs['Return']).cumprod()
    dfs = dfs.dropna()
    dfs.loc[:, 'TICKER'] = TICKER[i]
    df = df.append(dfs)

df = df[['TICKER', 'Return']]
df = df.reset_index().rename(columns={"index": "id"})
dff= pd.pivot_table(df, index = 'Date', columns = 'TICKER', values = 'Return')


rtn_hat = dff.mean() * 252
cov_hat = dff.cov() * 252


pf_rtn = []
pf_risk = []
pf_wgt = []


for _ in range(3) :
    wgt = np.random.random(4)
    wgt /= np.sum(wgt)
    wrtn = np.dot(wgt, rtn_hat)
    wrisk = np.sqrt(wgt.T,np.dot(cov_hat,wgt))
    pf_rtn.append(wrtn)
    pf_risk.append(wrisk)
    pf_wgt.append(wgt)
pf = {'Returns':pf_rtn, 'Risk':pf_risk}

for i, s in enumerate(TICKER):
    pf[s] = [wgt[i] for wgt in pf_wgt]

df = pd.DataFrame(pf)