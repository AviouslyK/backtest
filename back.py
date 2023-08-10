import pandas as pd
import talib
import datetime
import csv
import yfinance as yf
import os
import sys

def main(data, etf_data, stop_loss_percentage, rsi_buy, rsi_sell, macd_min, macd_diff, start, end, max_holding):
        
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

    # Define the trading strategy

    # Initialize signals DataFrame to store buy and sell signals
    signals = pd.DataFrame(0, index=data.index, columns=data.columns)
    # Dictionary for if we own each ticker
    holding = {ticker: False for ticker in data.columns.tolist()}
    # Dictionary containing each tickers original purchase price - for stop loss calculation
    buy_price = {ticker: 0 for ticker in data.columns.tolist()}
    
    for i in range(1, len(data)): # Date loop
        for ticker in data.columns: # ticker loop

            # How many stocks do we own right now?
            num_holding = sum(value for value in holding.values() if value)

            # Buy signal:
            if (rsi_data.at[rsi_data.index[i], ticker] < rsi_buy # RSI below threshold
                and macd_deriv_data.at[macd_deriv_data.index[i], ticker] > 0 # MACD curving upwards
                and macd_data.at[macd_data.index[i], ticker] <= macd_min # MACD < threshold
                and macd_data.at[macd_data.index[i], ticker] <= signal_data.at[signal_data.index[i], ticker] # MACD line <= signal line
                and abs(macd_data.at[macd_data.index[i], ticker] - signal_data.at[signal_data.index[i], ticker]) < macd_diff # |MACD line - signal line| < macd_min
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
                if holding[ticker] == True: # can't sell if we don't own
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
  
    # Define the initial investment amount and initialize the portfolio DataFrame
    initial_investment = 0
    # Holds the dollar value of your portfolio at any given day
    portfolio = pd.DataFrame(index=signals.index, columns=signals.columns)
    portfolio.iloc[0] = initial_investment

    # Calculate portfolio based on signals
    for ticker in signals.columns:
        purchase_price = 0
        day_before_purchase = 0
        holding = False
        for i in range(1, len(signals)): # Date loop

            if  signals.at[signals.index[i], ticker] == 1:  # Buy signal
                purchase_price = data.at[data.index[i], ticker] # keep track of price you bought at
                holding = True
                initial_investment = initial_investment + purchase_price
                # Invest price of stock into portfolio to buy, so -
                # - Portfolio[this ticker] is worth current price of stock + how much you had the day before
                portfolio.at[portfolio.index[i], ticker] = purchase_price + portfolio.at[portfolio.index[i - 1], ticker]
                day_before_purchase = data.index[i-1]
                #print("initial_investment = %.2f, purchase_price = %.2f, ticker = %s, date = %s" % (initial_investment, purchase_price, ticker, signals.index[i]))

            elif signals.at[signals.index[i], ticker] == -1:  # Sell signal
                # Portfolio[this ticker] doesn't change
                portfolio.at[portfolio.index[i], ticker] = portfolio.at[portfolio.index[i - 1], ticker]  
                holding = False
                #print("purchase_price = %.2f, sold_at = %.2f, ticker = %s, date = %s" % (purchase_price, data.at[data.index[i], ticker] , ticker, signals.index[i]))
           
            elif holding: # Hold signal
                # Portfolio[this ticker] is worth current price + what you had in it the day before stock was bought 
                portfolio.at[portfolio.index[i], ticker] = data.at[data.index[i], ticker] + portfolio.at[day_before_purchase, ticker]

            else:  # Just take value from previous day, since you don't own the stock
                portfolio.at[portfolio.index[i], ticker] = portfolio.at[portfolio.index[i - 1], ticker]   


    #portfolio = portfolio.dropna(axis=1) # Drop columns with NaN values   
    
    # Say you bought SPY/VTI on start date, and sold on end date
    # Relative return = final/initial * 100
    spy_return = 100*(etf_data.at[etf_data.index[-1],"SPY"]/etf_data.at[etf_data.index[0],"SPY"] - 1)
    vti_return = 100*(etf_data.at[etf_data.index[-1],"VTI"]/etf_data.at[etf_data.index[0],"VTI"] - 1)

    # Restore stdout
    #sys.stdout = sys.__stdout__

    # Calculate the total portfolio return
    
    # Monitoring
    #unique_value_counts = portfolio.apply(lambda col: col.nunique())
    #monret = portfolio[unique_value_counts[unique_value_counts > 2].index].copy() # Drop the columns whose value didn't change
    #print(monret)

    portfolio['Portfolio Value'] = portfolio.sum(axis=1)
    investment = portfolio['Portfolio Value'].iloc[-1] 
    #print("initial investment = ", initial_investment)
    #print("final portfolio value = ", investment)
    #print(portfolio['Portfolio Value'])

    # Printing the cumulative portfolio DataFrame, total return of the strategy, and ETF returns
    strategy_return = 100*(investment/initial_investment - 1)
    alpha = strategy_return - (spy_return + vti_return)/2
    print("\n\nStart = %s, End = %s" % (start,end))
    print("stop_loss_percentage = %.2f, rsi_sell = %d, macd_min = %.2f" % (stop_loss_percentage,rsi_sell,macd_min))
    print("rsi_buy = %d, maxholding = %d, macd_diff = %.2f" % (rsi_buy,max_holding,macd_diff))
    print("Strategy Return = %.3f%%" % strategy_return)
    print("SPY Return = %.3f%%" % spy_return)
    print("VTI Return = %.3f%%" % vti_return)
    print("alpha = %.3f%%" % alpha)

    return alpha
# end main


# ----------------------------------------------------------------------- #
# -------------------------  Download Data   ---------------------------- #
# ----------------------------------------------------------------------- #

# Define Parameters
start = '2022-01-01'
end = '2023-07-01'
start_date = pd.to_datetime(start)
end_date = pd.to_datetime(end)
minPrice = 20
minVol = 1000000


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

# Download historical price data for SPY and VTI using yfinance
etf_tickers = ['SPY', 'VTI']
etf_data = yf.download(etf_tickers, start=start_date, end=end_date)['Adj Close']

# ----------------------------------------------------------------------- #
# -----------------------------  Run Main  ------------------------------ #
# ----------------------------------------------------------------------- #

# Run many iterations and find the best configuration of parameters
stop_losses = [0.04]
rsi_buys = [30]
rsi_sells = [50]
macd_mins = [0]
macd_diffs = [0.05,0.005]
max_holdings = [8,20]

best_alpha = float("-inf")
best_combination = None

for stop_loss in stop_losses:
    for rsi_buy in rsi_buys:
        for rsi_sell in rsi_sells:
            for macd_min in macd_mins:  
                for macd_diff in macd_diffs:
                    for max_holding in max_holdings:
                        # return alpha with this combo of parameters
                        alpha_value = main(data,etf_data,stop_loss,rsi_buy,rsi_sell,macd_min,macd_diff,start,end,max_holding)

                        if alpha_value > best_alpha:
                            best_alpha = alpha_value
                            best_combination = (stop_loss,rsi_buy,rsi_sell,macd_min,macd_diff,max_holding)

print("Best Parameters were: (stop_loss,rsi_buy,rsi_sell,macd_min,macd_diff,max_holding) =  ", best_combination)
print("With an alpha = ", best_alpha)


'''
Best Parameters were: (stop_loss,rsi_buy,rsi_sell,macd_min,max_holding) =   (0.1, 25, 75, 1, 8)
With an alpha =  -110.99988727893118

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