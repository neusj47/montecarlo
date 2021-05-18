# monte carlo 시뮬레이션을 통한 포트폴리오 리턴, 리스크 산출함수
# 기간을 입력한다. (1차)
# 종목 데이터를 추출하여 일별 Rtn 값을 구한다. (2차)
# 연간 Rtn, 연간 Cov 값을 산출한다. (3차)
# 임의의 포트폴리오에서의 wgt, wgt_rtn, wgt_risk를 산출한다. (4차)
# 포트폴리오의 wgt_rtn, wgt_risk 그래프를 산출한다. (5차)


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
from datetime import date
import datetime
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import dash_bootstrap_components as dbc

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

df = get_montsimul_result(start,end)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H3("기간별 monte carlo simuation"),
    html.Br(),
    dcc.DatePickerRange(
        id="my-date-picker-range",
        min_date_allowed=date(2015, 1, 1),
        start_date_placeholder_text='2020-01-01',
        end_date_placeholder_text='2020-12-31',
        display_format='YYYYMMDD'
    ),
    html.Br(),
    dcc.Graph(
        style={'height': 600},
        id='my-graph'
    )
])



@app.callback(
    Output('my-graph', 'figure'),
    [Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date')])
def update_graph(start_date, end_date):
    def get_montsimul_result(start, end):
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
        dff = pd.pivot_table(df, index='Date', columns='TICKER', values='Return')

        # PF 리스크 산출을 위한 리턴, 공분산을 산출한다.
        rtn_hat = dff.mean() * 252
        cov_hat = dff.cov() * 252

        # PF 리스크를 100번 시뮬레이션하여 산출한다.
        pf_wgt = []
        pf_rtn = []
        pf_risk = []

        for _ in range(100):
            wgt = np.random.random(len(TICKER))
            wgt /= np.sum(wgt)
            wrtn = np.dot(wgt, rtn_hat)
            wrisk = np.sqrt(np.dot(wgt.T, np.dot(cov_hat, wgt)))
            pf_rtn.append(wrtn)
            pf_risk.append(wrisk)
            pf_wgt.append(wgt)

        pf = {'Return': pf_rtn, 'Risk': pf_risk}

        #  딕셔너리 형태의 pf를 dataframe 형태로 변환한다.
        for i, s in enumerate(TICKER):
            pf[s] = [wgt[i] for wgt in pf_wgt]
        df = pd.DataFrame(pf)
        df.iloc[0:len(df)] = round(df.iloc[0:len(df)], 2)
        return df
    df= get_montsimul_result(start_date, end_date)
    fig = px.scatter(data_frame=df, x='Risk', y='Return', title='MonteCarlo Simulation')
    return fig

if __name__ == "__main__":
    app.run_server(debug=True, port=8060)