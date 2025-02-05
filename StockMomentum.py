#!/usr/bin/env python
# coding: utf-8

# In[142]:


import pandas as pd


# In[144]:


import numpy as np


# In[145]:


import matplotlib.pyplot as plt


# In[146]:


import matplotlib.dates as mdates


# In[147]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[148]:


import yfinance as yf


# In[149]:


import datetime as dt


# In[150]:


import time


# In[151]:


import os


# In[152]:


import cufflinks as cf


# In[153]:


import plotly.express as px


# In[154]:


import plotly.graph_objects as go


# In[155]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

cf.go_offline()

from plotly.subplots import make_subplots


# In[156]:


from os import listdir
from os.path import isfile, join


# In[157]:


import warnings
warnings.simplefilter("ignore")


# In[158]:


PATH = "D:/Python For Finance/Wilshire/"

S_Date = "2017-02-01"
E_Date = "2022-12-06"
S_Date_DT = pd.to_datetime(S_Date)
E_Date_DT = pd.to_datetime(E_Date)


# # GET COLUMN DATA FROM CSV

# In[160]:


def get_column_from_csv(file, col_name):
    try:
        df=pd.read_csv(file)
    except FileNotFoundError:
        print("File Does Not Exist")
    else:
        return df[col_name]


# # GET STOCK TICKERS

# In[162]:


ticker = get_column_from_csv("D:/Python For Finance/Wilshire_Stocks.csv", "Ticker")
print(len(ticker))


# # SAVE STOCK DATA TO CSV

# In[164]:


# def save_to_csv_from_yahoo(folder, ticker):
#     stock = yf.Ticker(ticker)
#     try:
#         print("Get Data for : ", ticker)
#         df=stock.history(period ="5y")
#         time.sleep(2)
#         the_file=folder + ticker.replace(".", "_") + '.csv'
#         print(the_file, "Saved")
#         df.to_csv(the_file)
#     except Exception as ex:
#         print("Couldnt Get Data for :", ticker)


# # DOWNLOAD ALL STOCKS

# In[166]:


#for x in range(0,3481):
    #save_to_csv_from_yahoo(PATH, tickers[x])
    #print("FINISHED")


# # GET DATAFRAME FROM CSV

# In[168]:


def get_stock_df_from_csv(ticker):
    try:
        df = pd.read_csv(PATH + ticker + '.csv', index_col=0)
    except FileNotFoundError:
        print("File Does Not Exist")
        return None  # It's good practice to return None or handle the error appropriately.
    else:
        return df


# # GET LIST OF DOWNLOADED STOCKS

# In[170]:


# files = [x for x in listdir(PATH) if isfile(join(PATH,x))]
# tickers=[os.path.splitext(x)[0] for x in files]
# tickers.sort()
# print(len(tickers))


# # FETCH DAILY RETURNS

# In[172]:


# def add_daily_return_to_df(df):
#     df['Daily_Return']= (df['Close']/df['Close'].shift(1))-1
#     return df


# # FETCH CUMULATIVE RETURNS

# In[174]:


# def add_cum_return_to_df(df):
#     df['cum_return'] = (1+df['Daily_Return']).cumprod()
#     return df


# # ADD BOLLINGER BANDS DATA

# In[176]:


# def add_bollinger_bands(df):
#     df['Middle_Band']=df['Close'].rolling(window=20).mean()
#     df['Upper_Band']=df['Middle_Band'] + 1.96 * df['Close'].rolling(window=20).std()
#     df['Lower_Band']=df['Middle_Band'] - 1.96 * df['Close'].rolling(window=20).std()
#     return df


# # ADD ICHIMOKU DATA

# In[178]:


def add_Ichimoku(df):
    #conversion line
    hi_val=df['High'].rolling(window=9).max()
    low_val=df['Low'].rolling(window=9).min()
    df['Conversion']= (hi_val + low_val)/2
    #base line
    hi_val2 = df['High'].rolling(window=26).max()
    low_val2=df['Low'].rolling(window=26).min()
    df['Baseline']=(hi_val2 + low_val2)/2
    #Span A
    df['SpanA']=((df['Conversion']+df['Baseline'])/2)
    #Span B
    hi_val3 = df['High'].rolling(window=52).max()
    low_val3 = df['Low'].rolling(window=52).min()
    df['SpanB'] = ((hi_val3 + low_val3)/2).shift(26)
    #lagging
    df['Lagging'] = df['Close'].shift(-26)
    return df


# # TEST ON ANY STOCK

# In[180]:


# try:
#     print("Working on:","A")
#     new_df=get_stock_df_from_csv("A")
#     new_df=add_daily_return_to_df(new_df)
#     new_df=add_cum_return_to_df(new_df)
#     new_df=add_bollinger_bands(new_df)
#     new_df=add_Ichimoku(new_df)
#     new_df.to_csv(PATH + "A" + ".csv")
# except Exception as ex:
#     print(ex)


# # TO RUN FUNCTION ON ALL THE STOCKS

# In[182]:


# for x in ticker:
#     try:
#         print("Working on:", x)
#         new_df = get_stock_df_from_csv(x)
#         new_df = add_daily_return_to_df(new_df)
#         new_df = add_cum_return_to_df(new_df)
#         new_df = add_bollinger_bands(new_df)
#         new_df = add_Ichimoku(new_df)
#         new_df.to_csv(PATH + x + ".csv")
#     except Exception as ex:
#         print(ex)


# # PLOT BOLLINGER BANDS

# In[184]:


# def plot_with_boll_bands(df,ticker):
#     fig=go.Figure()
#     candle=go.Candlestick(x=df.index, open=df['Open'],
#     high=df['High'],low=df['Low'],
#     close=df['Close'], name='Candlestick')
    
#     upper_line=go.Scatter(x=df.index, y=df['Upper_Band'],line=dict(color='rgba(250,0,0,0.75)', width=1), name='Upper Band')
    
#     mid_line=go.Scatter(x=df.index, y=df['Middle_Band'],line=dict(color='rgba(0,0,250,0.75)', width=0.7), name='Middle Band')

#     lower_line=go.Scatter(x=df.index, y=df['Lower_Band'],line=dict(color='rgba(0,250,0,0.75)',width=1), name='Lower Band')
                          
#     fig.add_trace(candle)
#     fig.add_trace(upper_line)
#     fig.add_trace(mid_line)
#     fig.add_trace(lower_line)

#     fig.update_xaxes(title='Date', rangeslider_visible=True)
#     fig.update_yaxes(title='Price')
#     fig.update_layout(title= ticker + ' Bollinger Bands', height=1200, width=1800, showlegend=True)

#     fig.show()


# In[185]:


# test_df=get_stock_df_from_csv('AMD')
# plot_with_boll_bands(test_df, 'AMD')


# # PLOT ICHIMOKU DATA

# In[187]:


def get_fill_color(label):
    if label>=1:
        return 'regba(0,20,0,0.4)'
    else:
        return 'rgba(250,0,0,0.4)'


# In[191]:


import numpy as np
import pandas as pd
import plotly.graph_objects as go

def get_fill_color(label):
    if label >= 1:
        return 'rgba(0,250,0,0.4)'
    else:
        return 'rgba(250,0,0,0.4)'

def get_Ichimoku(df):
    # Create the candlestick chart
    candle = go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick')
    
    df1 = df.copy()  # Make a copy of df for later use
    fig = go.Figure()  # Initialize the figure
    
    # Calculate labels and groups
    df['label'] = np.where(df['SpanA'] > df['SpanB'], 1, 0)
    df['group'] = df['label'].ne(df['label'].shift()).cumsum()
    
    # Group by 'group' column but keep the original DataFrame accessible
    grouped_df = df.groupby('group')

    # Iterate over the grouped DataFrames
    for name, data in grouped_df:
        fig.add_traces(go.Scatter(x=data.index, y=data['SpanA'], line=dict(color='rgba(0,0,0,0)')))
        fig.add_traces(go.Scatter(x=data.index, y=data['SpanB'], line=dict(color='rgba(0,0,0,0)'), 
                                   fill='tonexty', fillcolor=get_fill_color(data['label'].iloc[0])))

    # Define additional traces using df1
    baseline = go.Scatter(x=df1.index, y=df1['Baseline'], line=dict(color='pink', width=2), name='Baseline')
    conversion = go.Scatter(x=df1.index, y=df1['Conversion'], line=dict(color='black', width=1), name='Conversion')
    lagging = go.Scatter(x=df1.index, y=df1['Lagging'], line=dict(color='purple', width=2), name='Lagging')

    span_a = go.Scatter(x=df1.index, y=df1['SpanA'], line=dict(color='green', width=2, dash='dot'), name='Span A')
    span_b = go.Scatter(x=df1.index, y=df1['SpanB'], line=dict(color='red', width=1, dash='dot'), name='Span B')

    # Add all traces to the figure
    fig.add_trace(candle)
    fig.add_trace(baseline)
    fig.add_trace(conversion)
    fig.add_trace(lagging)
    fig.add_trace(span_a)
    fig.add_trace(span_b)

    # Update layout and show the figure
    fig.update_layout(height=1200, width=1800, showlegend=True)
    fig.show()

# Example usage (make sure to provide a valid DataFrame 'df' with necessary columns)
# get_Ichimoku(df)


# In[193]:


test_df=get_stock_df_from_csv("AMD")
get_Ichimoku(test_df)


# In[ ]:




