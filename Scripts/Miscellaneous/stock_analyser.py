# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 01:59:17 2020

@author: Clayton
"""

import pandas as pd
import yfinance as yf
from yahoofinancials import *
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

#style.use('ggplot')
#start=dt.datetime(2000,1,1)
#end=dt.datetime(2016,12,31)
#df=web.DataReader('TSLA','yahoo',start,end)
#df.to_csv('tsla.csv')
#
#df1=pd.read_csv('tsla.csv',parse_dates=True, index_col=0)
#df.plot()

stockname='XOM'
tsla_df = yf.download(stockname, 
                      start='2015-01-01', 
                      end='2020-12-31', 
                      progress=False)
tsla_df.head()


ticker = yf.Ticker(stockname)

tsla_df = ticker.history(period="1hour")

tsla_df['Close'].plot(title=stockname+" stock price")
#
#assets = ['TSLA', 'MSFT', 'FB']
#
#yahoo_financials = YahooFinancials(assets)
#
#data = yahoo_financials.get_historical_price_data(start_date='2019-01-01', 
#                                                  end_date='2019-12-31', 
#                                                  time_interval='weekly')
#
#prices_df = pd.DataFrame({
#    a: {x['formatted_date']: x['adjclose'] for x in data[a]['prices']} for a in assets

## Computing Volatility

# Load the required modules and packages
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

# Pull NIFTY data from Yahoo finance 
NIFTY = yf.download(stockname,start='2020-3-23', end='2020-4-21')

# Compute the logarithmic returns using the Closing price 
NIFTY['Log_Ret'] = np.log(NIFTY['Close'] / NIFTY['Close'].shift(1))

# Compute Volatility using the pandas rolling standard deviation function
#NIFTY['Volatility'] = NIFTY['Log_Ret'].rolling(window=252).std() * np.sqrt(252)
#print(NIFTY.tail(15))

# Plot the NIFTY Price series and the Volatility
#NIFTY[['Volatility']].plot(subplots=True, xlim=['2020-4-1', '2020-4-20'], color='blue',figsize=(8, 6))
#})
fig= plt.figure()
ax=fig.add_subplot(111)
#ax.plot(x_data,y_data)

ax.plot(NIFTY['Close'])
#ax.set_xlim(xmin='2020-4-1', xmax='2020-4-20')
df=NIFTY.copy()