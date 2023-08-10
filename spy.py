import yfinance as yf
import matplotlib.pyplot as plt
import argparse
from matplotlib.widgets import Button

parser = argparse.ArgumentParser(description='Compare Trading Strategy to BuyAndHold')
parser.add_argument('--shortMA', type=int, default=50, help='Short moving average window size (default: 50)')
parser.add_argument('--longMA', type=int, default=200, help='Long moving average window size (default: 200)')
args = parser.parse_args()

short_MA = args.shortMA
long_MA = args.longMA

def get_signals(data):
    data[str(short_MA) + '-day MA'] = data['Close'].rolling(window=short_MA).mean()
    data[str(long_MA) + '-day MA'] = data['Close'].rolling(window=long_MA).mean()
    signals = []
    for i in range(1, len(data)):
        if data[str(short_MA) + '-day MA'][i] > data[str(long_MA) + '-day MA'][i] and data[str(short_MA) + '-day MA'][i-1] <= data[str(long_MA) + '-day MA'][i-1]:
            signals.append('Buy')
        elif data[str(short_MA) + '-day MA'][i] < data[str(long_MA) + '-day MA'][i] and data[str(short_MA) + '-day MA'][i-1] >= data[str(long_MA) + '-day MA'][i-1]:
            signals.append('Sell')
        else:
            signals.append('Hold')
    # Append a NaN value to match the length of the DataFrame
    signals = ['NaN'] + signals
    data['Signal'] = signals
    return data

def simulate_portfolio(data, initial_balance=1000.0):
    balance = initial_balance
    holdings = 0
    portfolio_worth = [balance]
    moves = []
    for i in range(1, len(data)):
        if data['Signal'][i] == 'Buy': # Buy
            # Calculate the maximum number of shares you can buy without overspending
            max_shares = balance // data['Adj Close'][i]
            holdings = max_shares
            balance -= holdings * data['Adj Close'][i]
            moves.append('Buy')
        elif data['Signal'][i] == 'Sell': # Sell
            # Sell all of your holdings when there's a sell signal
            balance += holdings * data['Adj Close'][i]
            holdings = 0
            moves.append("Sell")
        else: # Do nothing/Hold
            moves.append("Hold")
        
        portfolio_worth.append(balance + holdings * data['Adj Close'][i])

    # Append a NaN value to match the length of the DataFrame
    moves = ['NaN'] + moves
    data['Moves'] = moves
    data['Portfolio Worth'] = portfolio_worth
    return data

def simulate_buy_and_hold(data, initial_balance=1000.0):
    balance = initial_balance
    holdings = 0
    portfolio_worth = [balance]
    for i in range(1, len(data)):

        if i == 1: # Buy on day 0
            # Calculate the maximum number of shares you can buy without overspending
            max_shares = balance // data['Adj Close'][i]
            holdings = max_shares
            balance -= holdings * data['Adj Close'][i]
    
        # Update portfolio value
        portfolio_worth.append(balance + holdings * data['Adj Close'][i])

    data['Holding Worth'] = portfolio_worth
    return data

'''
# Plot the price of SPY overtime, as you buy and sell and below that
def plot_signals(data): 
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1)

    line_top = ax_top.plot(data.index, data['Close'], label='Close Price')
    line_top = ax_top.plot(data.index, data[str(short_MA) + '-day MA'], label=str(short_MA) + '-day MA', linestyle='dashed')
    line_top = ax_top.plot(data.index, data[str(long_MA) + '-day MA'], label=str(long_MA) + '-day MA', linestyle='dashed')
    buy_signals = data[data['Moves'] == 'Buy']
    ax_top.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='g', label='Buy Signal', s=150)
    sell_signals = data[data['Moves'] == 'Sell']
    ax_top.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='r', label='Sell Signal', s=150)
    ax_top.set_xlabel('Date')
    ax_top.set_ylabel('Stock Price')
    ax_top.set_title('SPY Stock Price with Moving Averages and Buy/Sell Signals')
    ax_top.legend()
  
    # Bottom Version 1: Portfolio Values
    line_bottom1, = ax_bottom.plot(data.index, data['Portfolio Worth'], label='Portfolio Worth', color='blue')
    line_bottom2, = ax_bottom.plot(data.index, data['Holding Worth'], label='Buy and Hold', color='red')
    ax_bottom.set_xlabel('Date')
    #ax_bottom.set_ylabel('Portfolio Worth ($)')
    #ax_bottom.set_title('Portfolio Worth Over Time')
    
    # Set the visibility of the bottom plot to False (initially hidden)
    line_bottom1.set_visible(False) 
    line_bottom2.set_visible(False) 

    # Bottom Version 2: Difference between portfolios
    difference = data['Portfolio Worth'] - data['Holding Worth']
    line_bottom3, = ax_bottom.plot(data.index, difference, label='Portfolio Worth - Holding Worth', color='purple')
    ax_bottom.set_ylabel('Difference ($)')
    ax_bottom.set_title('Difference between Portfolio Worth and Holding Worth Over Time')
    ax_bottom.legend()
    ax_bottom.legend()
    

    plt.tight_layout()
    plt.show(block=False)
'''

def main():
    spy_data = yf.download('SPY', start='1992-01-01', end='2023-01-01')
    spy_data = get_signals(spy_data)
    spy_data = simulate_portfolio(spy_data)
    spy_data = simulate_buy_and_hold(spy_data)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 8))

    line_top = ax_top.plot(spy_data.index, spy_data['Close'], label='Close Price')
    line_top = ax_top.plot(spy_data.index, spy_data[str(short_MA) + '-day MA'], label=str(short_MA) + '-day MA', linestyle='dashed')
    line_top = ax_top.plot(spy_data.index, spy_data[str(long_MA) + '-day MA'], label=str(long_MA) + '-day MA', linestyle='dashed')
    buy_signals = spy_data[spy_data['Moves'] == 'Buy']
    ax_top.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='g', label='Buy Signal', s=150)
    sell_signals = spy_data[spy_data['Moves'] == 'Sell']
    ax_top.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='r', label='Sell Signal', s=150)
    ax_top.set_xlabel('Date')
    ax_top.set_ylabel('Stock Price')
    ax_top.set_title('SPY Stock Price with Moving Averages and Buy/Sell Signals')
    ax_top.legend()
  
    # Bottom Version 1: Portfolio Values
    line_bottom1, = ax_bottom.plot(spy_data.index, spy_data['Portfolio Worth'], label='Strategy', color='blue')
    line_bottom2, = ax_bottom.plot(spy_data.index, spy_data['Holding Worth'], label='Buy and Hold', color='red')
    ax_bottom.set_xlabel('Date')
    #ax_bottom.set_ylabel('Portfolio Worth ($)')
    #ax_bottom.set_title('Portfolio Worth Over Time')
    
    # Set the visibility of the bottom plot to False (initially hidden)
    line_bottom1.set_visible(False) 
    line_bottom2.set_visible(False) 

    # Bottom Version 2: Difference between portfolios
    difference = spy_data['Portfolio Worth'] - spy_data['Holding Worth']
    line_bottom3, = ax_bottom.plot(spy_data.index, difference, label='Strategy - Buy and Hold', color='purple')
    ax_bottom.set_ylabel('Difference ($)')
    ax_bottom.set_title('Difference between Portfolio Worth and Holding Worth Over Time')

    handles, labels = ax_bottom.get_legend_handles_labels()
    visible_handles = [handle for handle, visible in zip(handles, [line_bottom1.get_visible(),line_bottom2.get_visible(),line_bottom3.get_visible()]) if visible]
    visible_labels = [label for label, visible in zip(labels, [line_bottom1.get_visible(),line_bottom2.get_visible(),line_bottom3.get_visible()]) if visible]
    ax_bottom.legend(visible_handles, visible_labels)
    
    plt.tight_layout()
    #plt.show(block=False)

    # Create a toggle function to switch between the bottom plots
    def toggle_plot(event):
        if toggle_plot.plot_type == 'Difference':
            line_bottom1.set_visible(True) 
            line_bottom2.set_visible(True) 
            line_bottom3.set_visible(False) 
            ax_bottom.set_ylabel('Portfolio Worth ($)')
            ax_bottom.set_title('Portfolio Worth Over Time')
            toggle_plot.plot_type = 'Value'
        else:
            line_bottom1.set_visible(False) 
            line_bottom2.set_visible(False) 
            line_bottom3.set_visible(True) 
            ax_bottom.set_ylabel('Difference ($)')
            ax_bottom.set_title('Difference between Portfolio Worth and Holding Worth Over Time')
            toggle_plot.plot_type = 'Difference'

        # Update the legend to show only the visible lines
        handles, labels = ax_bottom.get_legend_handles_labels()
        visible_handles = [handle for handle, visible in zip(handles, [line_bottom1.get_visible(),line_bottom2.get_visible(),line_bottom3.get_visible()]) if visible]
        visible_labels = [label for label, visible in zip(labels, [line_bottom1.get_visible(),line_bottom2.get_visible(),line_bottom3.get_visible()]) if visible]
        ax_bottom.legend(visible_handles, visible_labels)
        plt.draw()


    toggle_plot.plot_type = 'Difference'

    # Connect the toggle function to a key press event (in this example, we use the 't' key)
    fig.canvas.mpl_connect('key_press_event', lambda event: toggle_plot(event))

    # Show the plot
    plt.show()

if __name__ == '__main__':
    main()
