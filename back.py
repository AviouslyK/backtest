import pandas as pd
import talib
import datetime
import csv
import yfinance as yf
import os
import sys

def main(stop_loss_percentage, rsi_sell, macd_min,start,end,minVol,minPrice):
        
    # ----------------------------------------------------------------------- #
    # ---------------------- Function Def and Setup  ------------------------ #
    # ----------------------------------------------------------------------- #

    # Redirect stdout to os.devnull
    #sys.stdout = open(os.devnull, 'w')

    # Prevent a "string indices must be integers" TypeError 
    yf.pdr_override() 

    # Define a function to calculate RSI for a given column (stock ticker)
    def calculate_rsi(column):
        delta = column.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Define a function to calculate the second derivative of the MACD for a given column (stock ticker)
    def calculate_macd_second_derivative(column):
        # Calculate the 12-day and 26-day Exponential Moving Averages (EMA)
        ema_12 = column.ewm(span=12, adjust=False).mean()
        ema_26 = column.ewm(span=26, adjust=False).mean()

        # Calculate the MACD line (12-day EMA - 26-day EMA)
        macd_line = ema_12 - ema_26

        # Calculate the second derivative of the MACD line
        second_derivative = macd_line.diff().diff()

        return second_derivative

    # Define a function to calculate the MACD for a given column (stock ticker)
    def calculate_macd(column):
        # Calculate the 12-day and 26-day Exponential Moving Averages (EMA)
        ema_12 = column.ewm(span=12, adjust=False).mean()
        ema_26 = column.ewm(span=26, adjust=False).mean()

        # Calculate the MACD line (12-day EMA - 26-day EMA)
        macd_line = ema_12 - ema_26

        return macd_line



    # ----------------------------------------------------------------------- #
    # ------------------------   Data Cleaning  ----------------------------- #
    # ----------------------------------------------------------------------- #

    # Specify the list of tickers to use
    tickers = []
    with open("ticker_list.csv", 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            ticker = row['Symbol']
            last_sale = float(row['Last Sale'].replace('$', ''))
            if (last_sale > minPrice 
                and "^" not in ticker 
                and "/" not in ticker 
                and len(ticker) < 6
            ):
                tickers.append(ticker)


    # Fetch the historical pricing data for each ticker from Yahoo Finance
    data = yf.download(tickers= tickers, start=start, end=end)

    # Calculate the average values in each column
    column_means = data['Volume'].mean()

    # Filter the columns that have an average value below the threshold
    columns_below_threshold = column_means[column_means < minVol]

    # Now each Column is the ticker, and each row is the date, each entry is the closing price
    data = data['Adj Close']
    data = data.drop(columns=columns_below_threshold.index)


    # Calculate RSI for each column (stock ticker) in the DataFrame
    rsi_data = data.apply(calculate_rsi)

    # Calculate the MACD and signal line for each column (stock ticker) in the DataFrame
    macd_data = data.apply(calculate_macd)
    signal_data = macd_data.ewm(span=9, adjust=False).mean()
   

    # Calculate the second derivative of MACD for each column (stock ticker) in the DataFrame
    macd_deriv_data = data.apply(calculate_macd_second_derivative)


    # ----------------------------------------------------------------------- #
    # ------------------------   Trading Strategy  -------------------------- #
    # ----------------------------------------------------------------------- #
    '''
    # Create a DataFrame to store trading signals
    signals = pd.DataFrame(index=data.index, columns=data.columns)

    # Define the trading strategy
    for ticker in data.columns:
        signals[ticker] = 0  # Initialize all signals as 0
        holding = False
        buy_price = 0   

        for i in range(1, len(data)):
            # Buy signal:
            if (rsi_data.at[rsi_data.index[i], ticker] < 30 # RSI < 30
                and macd_deriv_data.at[macd_deriv_data.index[i], ticker] > 0 # MACD curving upwards
                and macd_data.at[macd_data.index[i], ticker] <= -macd_min # MACD < threshold
                and macd_data.at[macd_data.index[i], ticker] <= signal_data.at[signal_data.index[i], ticker] # MACD line < signal line
                and not holding # don't buy if you already own
            ):
                signals.at[data.index[i], ticker] = 1  # 1 represents buy signal
                holding = True
                buy_price = data.at[data.index[i], ticker]

            # Sell signal: RSI > rsi_sell
            elif (rsi_data.at[rsi_data.index[i], ticker] > rsi_sell # high RSI
                  or (holding and (data.at[data.index[i], ticker] <= buy_price * (1 - stop_loss_percentage))) # stop loss
                  or macd_deriv_data.at[macd_deriv_data.index[i], ticker] < 0 # MACD curving downwards
            ): 
                signals.at[data.index[i], ticker] = -1  # -1 represents sell signal
                holding = False
    '''
    # Define the trading strategy
    
    # Initialize signals DataFrame to store buy and sell signals
    signals = pd.DataFrame(0, index=data.index, columns=data.columns)
    # Dictionary for if we own each ticker
    holding = {ticker: False for ticker in data.columns.tolist()}
    # Dictionary containing each tickers original purchase price - for stop loss calculation
    buy_price = {ticker: 0 for ticker in data.columns.tolist()}

    max_holding = 1000 # don't want to own more than x stocks on any given day 

    for i in range(1, len(data)): # Date loop
        # How many stocks do we own right now?
        num_holding = sum(value for value in holding.values() if value)
        for ticker in data.columns: # ticker loop
            # Buy signal:
            if (rsi_data.at[rsi_data.index[i], ticker] < 30 # RSI < 30
                and macd_deriv_data.at[macd_deriv_data.index[i], ticker] > 0 # MACD curving upwards
                and macd_data.at[macd_data.index[i], ticker] <= -macd_min # MACD < threshold
                and macd_data.at[macd_data.index[i], ticker] <= signal_data.at[signal_data.index[i], ticker] # MACD line < signal line
                and not holding[ticker] # don't buy if you already own
                and num_holding < max_holding # If we already own the max, can't buy another
            ):
                if holding[ticker] == False: # Don't buy if we already own
                    signals.at[data.index[i], ticker] = 1  # 1 represents buy signal
                    holding[ticker] = True
                    buy_price[ticker] = data.at[data.index[i], ticker]

            # Sell signal: RSI > rsi_sell
            elif ((rsi_data.at[rsi_data.index[i], ticker] > rsi_sell # high RSI
                  and macd_deriv_data.at[macd_deriv_data.index[i], ticker] < 0) # MACD curving downwards
                  or (holding[ticker] and (data.at[data.index[i], ticker] <= buy_price[ticker] * (1 - stop_loss_percentage))) # stop loss
                  #or macd_deriv_data.at[macd_deriv_data.index[i], ticker] < 0 # MACD curving downwards
            ): 
                if holding[ticker] == True: # can't sel if we don't own
                    signals.at[data.index[i], ticker] = -1  # -1 represents sell signal
                    holding[ticker] = False

    # ----------------------------------------------------------------------- #
    # -----------------------   Calculate Returns  -------------------------- #
    # ----------------------------------------------------------------------- #
    
    # Define start_date and end_date
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
   
    # Filter signals and data based on the start and end dates
    signals = signals.loc[start_date:end_date]
    data = data.loc[start_date:end_date]

    # Monitoring:
    #monsig = signals.drop(columns=signals.columns[signals.eq(0).all()]).copy # Drop columns with all 0s
    #print(monsig)
    
    # Define the initial investment amount and initialize the returns DataFrame
    initial_investment = 10000
    # Holds the dollar value in your portfolio at any given day
    returns = pd.DataFrame(index=signals.index, columns=signals.columns)
    returns.iloc[0] = initial_investment

    # Calculate returns based on signals
    for ticker in signals.columns:
        purchase_price = 0
        holding = False
        for i in range(1, len(signals)): # Date loop

            if  signals.at[signals.index[i], ticker] == 1:  # Buy signal
                purchase_price = data.at[data.index[i], ticker] # keep track of price you bought at
                holding = True
                returns.at[returns.index[i], ticker] = returns.at[returns.index[i - 1], ticker] # previous day

            elif signals.at[signals.index[i], ticker] == -1:  # Sell signal
                # Current value is initial investment - purchase price + current price
                returns.at[returns.index[i], ticker] = initial_investment - purchase_price + data.at[data.index[i], ticker]
                holding = False
           
            elif holding: # Hold signal
                # Current value is initial investment - purchase price + current price
                returns.at[returns.index[i], ticker] = initial_investment - purchase_price + data.at[data.index[i], ticker]

            else:  # Just take value from previous day
                returns.at[returns.index[i], ticker] = returns.at[returns.index[i - 1], ticker]   

    # Weight by number of tickers, i.o.w. equal allocation
    #returns = returns.dropna(axis=1) # Drop columns with NaN values
    returns = returns / len(returns.columns)
    
    # Download historical price data for SPY and VTI using yfinance
    etf_tickers = ['SPY', 'VTI']
    etf_data = yf.download(etf_tickers, start=start_date, end=end_date)['Adj Close']

    # Say you bought SPY/VTI on start date, and sold on end date
    # Relative return = final/initial * 100
    spy_return = 100*(etf_data.at[etf_data.index[-1],"SPY"]/etf_data.at[etf_data.index[0],"SPY"] - 1)
    vti_return = 100*(etf_data.at[etf_data.index[-1],"VTI"]/etf_data.at[etf_data.index[0],"VTI"] - 1)

    # Restore stdout
    #sys.stdout = sys.__stdout__

    # Calculate the total portfolio return
    
    # Monitoring
    #unique_value_counts = returns.apply(lambda col: col.nunique())
    #monret = returns[unique_value_counts[unique_value_counts > 1].index].copy() # Drop the columns with return = 0
    #print(monret)

    returns['Total_Return'] = returns.sum(axis=1)
    investment = returns['Total_Return'].iloc[-1]
    #print(returns['Total_Return'])

    # Printing the cumulative returns DataFrame, total return of the strategy, and ETF returns
    strategy_return = 100*(investment/initial_investment - 1)
    alpha = strategy_return - (spy_return + vti_return)/2
    print("\n\nStart = %s, End = %s" % (start,end))
    print("stop_loss_percentage = %.2f, rsi_sell = %d, macd_min = %.2f" % (stop_loss_percentage,rsi_sell,macd_min))
    print("Strategy Return = %.3f%%" % strategy_return)
    print("SPY Return = %.3f%%" % spy_return)
    print("VTI Return = %.3f%%" % vti_return)
    print("alpha = %.3f%%" % alpha)

    #return alpha

# Optimize config variables: rsi_sell, macd_min, etc.
#stop_losses = [0.01,0.07,0.13]
#rsi_sells = [50,60,70]
#macd_mins = [0.75,1.5,2.25]
#min_vols = [100000,500000,1000000]
#min_prices = [50,150,250]

#best_alpha =

#for stop_loss in stop_losses:
#    for rsi_sell in rsi_sells:
#        for macd_min in macd_mins:  
main(0.15,80,1,'2009-01-01','2010-01-01',200000,200)


'''
For calculating alpha:
all jan-01 to jan-01

strat[1]:
stop_loss_percentage = 0.20, rsi_sell = 55, macd_min = 1.50
minVol = 200,000, minPrice = 20

strat[2]:
stop_loss_percentage = 0.10, rsi_sell = 55, macd_min = 1.50
minVol = 200,000, minPrice = 100

strat[3]:
stop_loss_percentage = 0.10, rsi_sell = 55, macd_min = 1.50
minVol = 200,000, minPrice = 200
__________________________________________________________________________________________
         |  2022-2023 ||  2021-2022 || 2020-2021 || 2019-2020 || 2018-2019 || 2017-2018 ||
SPY      |   -19.95   ||    29.63   ||   17.24   ||   31.09   ||   -5.25   ||   20.78   ||
VTI      |   -21.31   ||    26.64   ||   20.08   ||   30.57   ||   -5.90   ||   20.30   ||
strat[1] |     0.30   ||     0.52   ||    0.33   ||    0.39   ||    0.17   ||    0.35   || ....
strat[2] |     1.23   ||     1.17   ||    1.38   ||    1.14   ||    0.42   ||    0.85   ||
strat[3] |     3.31   ||     2.82   ||    2.87   ||    2.01   ||    0.48   ||    1.28   ||
__________________________________________________________________________________________

      __________________________________________________________________________________________
               |  2016-2017 ||  2015-2016 || 2014-2015 || 2013-2014 || 2012-2013 || 2011-2012 ||
      SPY      |    13.59   ||     1.29   ||   14.56   ||   29.00   ||   14.17   ||    0.85   ||
      VTI      |    14.53   ||     0.43   ||   13.54   ||   30.15   ||   14.83   ||   -0.06   ||
....  strat[1] |     ----   ||     ----   ||    ----   ||    ----   ||    ----   ||    ----   ||
      strat[2] |     ----   ||     ----   ||    ----   ||    ----   ||    ----   ||    ----   ||
      strat[3] |     0.73   ||     0.73   ||    0.64   ||    0.72   ||    0.39   ||    0.23   ||
      __________________________________________________________________________________________

      __________________________________________________________________________________________
               |  2010-2011 ||  2009-2010 || 2008-2009 || 2007-2008 || 2006-2007 || 2005-2006 ||
      SPY      |    13.14   ||    22.66   ||   -36.24  ||    5.33   ||    13.84  ||    5.33   ||
      VTI      |    15.50   ||    25.29   ||   -36.23  ||    5.57   ||    14.01  ||    7.08   ||
....  strat[1] |     ----   ||     ----   ||    ----   ||    ----   ||    ----   ||    ----   ||
      strat[2] |     ----   ||     ----   ||    ----   ||    ----   ||    ----   ||    ----   ||
      strat[3] |     0.34   ||     0.19   ||    0.08   ||    0.27   ||    0.20   ||    0.19   ||
      __________________________________________________________________________________________


TODO: add actual alpha calculation. something like:

import statsmodels.api as sm

strategy_returns = pd.Series([0.02, 0.01, -0.01, 0.03, 0.02])
market_returns = pd.Series([0.01, 0.02, 0.03, 0.02, 0.01])

# Add a constant column for the regression intercept
X = sm.add_constant(market_returns)

# Perform linear regression
model = sm.OLS(strategy_returns, X).fit()

# Get the alpha and beta coefficients
alpha, beta = model.params[0], model.params[1]

print("Alpha:", alpha)
print("Beta:", beta)


'''