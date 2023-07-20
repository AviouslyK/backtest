import pandas as pd
import talib
import datetime
import csv
import yfinance as yf
import os
import sys

def main(stop_loss_percentage, rsi_sell, macd_min):
        
    # ----------------------------------------------------------------------- #
    # ---------------------- Function Def and Setup  ------------------------ #
    # ----------------------------------------------------------------------- #

    # Redirect stdout to os.devnull
    sys.stdout = open(os.devnull, 'w')

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

    # Set the Price and Volume thresholds
    minVol = 200000
    minPrice = 15

    # Bad Ticker List 
    # - These have given Failed download warnings which I'm tired of looking at
    bad_tickers = ['AMH','RCA','EAI','EMP','ZS','ETX','WFG','EDR','GMAB', 'PD',
                   'BFC','SEB','SFB','CALX','PFXNZ','TTC','UNMA','CRAI','IRT',
                   'SHOO', 'KE', 'HSBC', 'GIB', 'CSWCZ', 'BLCO','GRMN','CNA',
                   'HTFB','AQNB','LRCX','OI','FIX']

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
                and ticker not in bad_tickers
                and len(ticker) < 6
            ):
                tickers.append(ticker)


    # Fetch the historical pricing data for each ticker from Yahoo Finance
    data = yf.download(tickers= tickers, start='2023-06-01', end='2023-07-19')

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


    # Create a DataFrame to store trading signals
    signals = pd.DataFrame(index=data.index, columns=data.columns)


    # ----------------------------------------------------------------------- #
    # ------------------------   Trading Strategy  -------------------------- #
    # ----------------------------------------------------------------------- #

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
            ):
                signals.at[data.index[i], ticker] = 1  # 1 represents buy signal
                holding = True
                buy_price = data.at[data.index[i], ticker]

            # Sell signal: RSI > rsi_sell
            elif (rsi_data.at[rsi_data.index[i], ticker] > rsi_sell) or (holding and (data.at[data.index[i], ticker] <= buy_price * (1 - stop_loss_percentage))):
                signals.at[data.index[i], ticker] = -1  # -1 represents sell signal
                holding = False

    # ----------------------------------------------------------------------- #
    # -----------------------   Calculate Returns  -------------------------- #
    # ----------------------------------------------------------------------- #
    
    # Define start_date and end_date
    start_date = pd.to_datetime('2023-06-01')
    end_date = pd.to_datetime('2023-07-19')
   
    # Filter signals and data based on the start and end dates
    signals = signals.loc[start_date:end_date]
    data = data.loc[start_date:end_date]

    # Define the initial investment amount and initialize the returns DataFrame
    initial_investment = 10000
    # Holds the dollar value in your portfolio at any given day
    returns = pd.DataFrame(index=signals.index, columns=tickers)
    returns.iloc[0] = initial_investment

    # Calculate returns based on signals
    for ticker in signals.columns:
        purchase_price = 0
        holding = False
        for i in range(1, len(signals)): # Date loop

            if signals.at[signals.index[i], ticker] == 1:  # Buy signal
                purchase_price = data.at[data.index[i], ticker] # keep track of price you bought at
                holding = True

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
    returns = returns.dropna(axis=1) # Drop columns with NaN values
    returns = returns / len(signals.columns)
    
    # Download historical price data for SPY and VTI using yfinance
    etf_tickers = ['SPY', 'VTI']
    etf_data = yf.download(etf_tickers, start=start_date, end=end_date)['Adj Close']

    # Say you bought SPY/VTI on start date, and sold on end date
    spy_value = initial_investment - etf_data.at[etf_data.index[0],'SPY'] + etf_data.at[etf_data.index[-1],"SPY"]
    vti_value = initial_investment - etf_data.at[etf_data.index[0],'VTI'] + etf_data.at[etf_data.index[-1],"VTI"]
    spy_return = (100*(spy_value-initial_investment)/initial_investment)
    vti_return = (100*(vti_value-initial_investment)/initial_investment)

    # Restore stdout
    sys.stdout = sys.__stdout__

    # Calculate the total portfolio return
    print(returns)
    returns['Total_Return'] = returns.sum(axis=1)
    investment = returns['Total_Return'].iloc[-1]
    print(returns['Total_Return'])

    # Printing the cumulative returns DataFrame, total return of the strategy, and ETF returns
    strategy_return = (100*(investment-initial_investment)/initial_investment)
    print("\n\nstop_loss_percentage = %.2f, rsi_sell = %d, macd_min = %.2f" % (stop_loss_percentage,rsi_sell,macd_min))
    print("Strategy Return = %.3f%%" % strategy_return)
    print("SPY Return = %.3f%%" % spy_return)
    print("VTI Return = %.3f%%" % vti_return)
    print("alpha = ", (strategy_return - (spy_return + vti_return)/2))

main(0.10,55,1.5)
#main(0.10,55,1)


