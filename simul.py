

import pandas_datareader.data as web
import dash_bootstrap_components as dbc
import dash_table

import pandas as pd
import numpy as np
import pandas_datareader.data as web

start = '20200101'
end = '20210430'

def get_montsimul_result(start,end):
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

    # PF 리스크 산출을 위한 리턴, 공분산을 산출한다.
    rtn_hat = dff.mean() * 252
    cov_hat = dff.cov() * 252

    # PF 리스크를 100번 시뮬레이션하여 산출한다.
    pf_wgt = []
    pf_rtn = []
    pf_risk = []

    for _ in range(100) :
        wgt = np.random.random(len(TICKER))
        wgt /= np.sum(wgt)
        wrtn = np.dot(wgt, rtn_hat)
        wrisk = np.sqrt(np.dot(wgt.T,np.dot(cov_hat,wgt)))
        pf_rtn.append(wrtn)
        pf_risk.append(wrisk)
        pf_wgt.append(wgt)

    pf = {'Returns':pf_rtn, 'Risk':pf_risk}

    #  딕셔너리 형태의 pf를 dataframe 형태로 변환한다.
    for i,s in enumerate(TICKER):
        pf[s] = [wgt[i] for wgt in pf_wgt]
    df = pd.DataFrame(pf)
    df.iloc[0:len(df)] = round(df.iloc[0:len(df)],2)
    return df


