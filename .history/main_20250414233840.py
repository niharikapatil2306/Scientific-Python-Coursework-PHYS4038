'''
I, Niharika Patil, have read and understood the School's Academic Integrity Policy, as well as guidance relating to this 
module, and confirm that this submission complies with the policy. The content of this file is my own original work, with 
any significant material copied or adapted from other sources clearly indicated and attributed.
'''

# the env file I'm using is project_reducedversioning.yml
# First singup then directed to login 

import tkinter as tk
from tkinter import ttk
import yfinance as yf
import sqlite3
from tkinter import messagebox
import re
import hashlib
import random
from datetime import *
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

end_date = datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

# Fetching data for benchmark stock S&P500
benchmark_data = yf.download('^GSPC', start=start_date, end=end_date)

benchmark_data.columns = benchmark_data.columns.droplevel(1)

benchmark_data.columns.name = None
benchmark_data.reset_index(inplace=True)
benchmark_data.sort_values('Date', inplace=True)

benchmark_data['Daily Return'] = benchmark_data['Adj Close'].pct_change().fillna(0)
benchmark_data['Cumulative Return'] = ((1 + benchmark_data['Daily Return']).cumprod() - 1)*100

def fetch_stock_data(ticker, start_date=start_date, end_date=end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        # Handle empty DataFrame (no data found)
        if stock_data.empty:
            print(f"No data found for {ticker}.")
            return None

        stock_data.columns = stock_data.columns.droplevel(1)
        stock_data.columns.name = None
        stock_data.reset_index(inplace=True)
        stock_data.sort_values('Date', inplace=True)
        stock_data['Returns'] = stock_data['Adj Close'].pct_change().fillna(0)

        return stock_data

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        raise

class FinancialMetrics:
    def __init__(self, asset, data):
        # Initialize the asset and data, calculate ROI and ROI percentage
        self.data = data
        self.asset = asset

        # Calculate ROI and ROI percentage based on the asset's data
        self.data['ROI'] = self.calc_roi()
        self.data['ROI (%)'] = self.calc_roi_percent()

        # Retrieve the stock information using yfinance
        self.stock = yf.Ticker(self.asset['ticker'])

    def calc_std(self):
        # Calculate the standard deviation of the returns (scaled by 100)
        return (self.data['Returns']*100).std()
    
    def calc_var(self):
        # Calculate the variance of the returns (scaled by 100)
        return (self.data['Returns']*100).var()
    
    def get_beta(self):
        # Get the beta value for the asset (systematic risk relative to the market)
        # `benchmark_data` needs to be defined elsewhere in your code
        cov = self.data['Returns'], benchmark_data['Daily Return']
        beta = self.stock.info.get('beta')  # Get the stock's beta from Yahoo Finance
        return beta
    
    def avg_daily_returns(self):
        # Calculate the average daily return (scaled by 100)
        avg = (self.data['Returns']*100).mean()
        return avg
    
    def calc_var_risk(self, ci_level=0.95):
        # Calculate the Value at Risk (VaR) for the asset at a given confidence level (default 95%)
        mu = self.avg_daily_returns()
        sigma = self.calc_std()/100  # Standard deviation (scaled)
        z_score = stats.norm.ppf(1-(1-ci_level))  # Z-score for the confidence level
        var_percentage = mu - z_score * sigma  # VaR percentage
        return var_percentage
        
    def get_risk_free_rate(self):
        # Retrieve the current risk-free rate (based on 10-year Treasury yield)
        rate = yf.Ticker("^TNX").history(period="1d")['Close'].iloc[-1]
        return rate
    
    def sharpe_ratio(self):
        # Calculate the Sharpe ratio, which measures risk-adjusted return
        sharpe = (self.avg_daily_returns() - self.get_risk_free_rate()) / self.calc_std()
        return sharpe
    
    def get_roa(self):
        # Get the Return on Assets (ROA) for the asset from Yahoo Finance
        roa = self.stock.info.get('returnOnAssets')
        return roa
    
    def get_roe(self):
        # Get the Return on Equity (ROE) for the asset from Yahoo Finance
        roe = self.stock.info.get('returnOnEquity')
        return roe
    
    def calc_roi(self):
        # Calculate the return on investment (ROI) based on the asset's buy price
        roi = self.data['Close'] - self.asset['buy_price']
        return roi
    
    def calc_roi_percent(self):
        # Calculate the percentage return on investment (ROI%)
        roi_percent = (self.data['Close'] - self.asset['buy_price']) / self.asset['buy_price'] * 100
        return roi_percent
    
    def calc_bands(self, std_dev_multiplier=1):
        # Calculate the upper and lower bands for the returns based on standard deviation
        avg_return = self.avg_daily_returns()
        std_dev = self.calc_std()
        
        # Upper and lower bands calculated using standard deviation multiplier
        upper_band = avg_return + std_dev_multiplier * std_dev
        lower_band = avg_return - std_dev_multiplier * std_dev
        
        return upper_band, lower_band
    
class FinancePlots:
    def __init__(self):
        self.legend_handles = []
    
    def def_fig(self):
        return plt.figure(figsize=(10, 6))
    
    def show_labels_titles(self, x_label, y_label, title, legend_values):
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend(self.legend_handles, legend_values)
        self.legend_handles =[]
    
    def add_hls(self, hls):
        for i, hl in enumerate(hls):
            l = plt.axhline(hl['y'], color=hl.get('color', 'black'), linestyle=hl.get('linestyle', '--'), label=hl.get('label', ''))
            self.legend_handles.append(l)

    def add_plots(self, x_val, plots):
        for plot in plots:
            l, = plt.plot(x_val, plot['y'], color=plot.get('color', 'black'), linestyle=plot.get('linestyle', '--'), label=plot.get('label', ''), marker=plot.get('marker', None))
            self.legend_handles.append(l)
    
    def bar_plot(self, x_val, y_val, x_label, y_label, legend_values, title, hls, grid_val, current_stock=None, current_stock_color='salmon', other_color='lightcoral'):
        fig = self.def_fig()
        if current_stock is not None:
            colors = [current_stock_color if x == current_stock else other_color for x in x_val]
        else:
            colors = [other_color for _ in x_val]
        bar = plt.bar(x_val, y_val, color=colors)
        plt.bar_label(bar)
        if hls:
            self.add_hls(hls)

        plt.grid(visible=grid_val)
        self.show_labels_titles( x_label, y_label, title, legend_values)
        return fig 

    def line_plot(self, x_val, y_val, x_label, y_label, legend_values, title, hls, grid_val):
        fig = self.def_fig()
        plt.plot(x_val, y_val, c='cornflowerblue')
        if hls:
            self.add_hls(hls)
        plt.grid(visible=grid_val)
        self.show_labels_titles( x_label, y_label, title, legend_values)
        return fig 

    def multiple_plots(self, x_val, x_label, y_label, legend_values, title, plots):
        fig = self.def_fig()
        if plots:
            self.add_plots(x_val, plots)
        self.show_labels_titles( x_label, y_label, title, legend_values)
        return fig 
    
    def hist_plot(self, x_val, x_label, y_label, title, color):
        fig = self.def_fig()
        sns.histplot(x_val, kde=True, color=color)
        self.show_labels_titles( x_label, y_label, title,[])
        return fig 

    def band_plot(self, x_val, plots, hls, bands, x_label, y_label, title, label):
        fig = self.def_fig()
        self.add_plots(x_val, plots)
        self.add_hls(hls)
        band = plt.fill_between(x_val, bands[0], bands[1], color="salmon", alpha=0.2, label=label)
        self.legend_handles.append(band)
        all_labels = [plot['label'] for plot in plots] + [hl['label'] for hl in hls] + [label]
        self.show_labels_titles( x_label, y_label, title, all_labels)
        return fig 

class FinancialAnalysis:
    def __init__(self, asset, data):
        
        self.fp = FinancePlots()
        self.data = data
        self.asset = asset
        self.fm = FinancialMetrics(asset=self.asset, data=self.data) 

    def std_comparison(self, std, tickers):
        t = self.asset['ticker']
        fig = self.fp.bar_plot(tickers, std, "Tickers of shares",
                                          "Standard Deviation/Volatility (%)", [], "Standard Deviation/Volatility of Shares.", None, True,
                                          current_stock=t,
                                          current_stock_color='salmon',
                                          other_color='lightblue')
        return {
            "plot": fig
        }
    
    def comparison(self):
        t = self.asset['ticker']
        fig = self.fp.multiple_plots(self.data['Date'], "Date", "Percentages", [f"Daily Return on Investment {t}", "S&P 500 Returns"], "Comparison between ROI and Returns of S&P 500", [{'y': self.data['ROI (%)'], 'color': 'salmon', 'label': 'S&P 500 Returns'},{'y': benchmark_data['Cumulative Return'], 'color': 'black', 'label': 'S&P 500 Returns'}])
        return {
            "plot": fig
        }
    
    def returns_stock(self): 
        t = self.asset['ticker']
        fig = self.fp.line_plot(self.data['Date'], self.data['Returns']*100, 'Date', 'Daily Returns (%)', ['expected Return'], f"Returns of share {t}", [{'y': self.fm.avg_daily_returns(), 'color': 'black', 'label': 'Expected Return'}], False )
        return {
            "plot": fig
        }
    
    def return_dist(self):
        t = self.asset['ticker']
        fig = self.fp.hist_plot(self.data['Returns']*100, 'Daily Return (%)' , "Frequency", f"Daily Returns Distribution for {t}", color="salmon")
        return {
            "plot": fig
        }

    def std_analysis(self):
        t = self.asset['ticker']

        bands = self.fm.calc_bands(1)
        std = self.fm.calc_std() 

        fig=self.fp.band_plot(self.data['Date'],[{'y': self.data['Returns']*100, 'color': 'cornflowerblue', 'label': f'{t} Returns', 'marker':'o'}],[{'y': self.fm.avg_daily_returns(), 'color': 'black', 'label': f'{t} Average'}], bands, 'Date', "Returns (%)", f"Stock Returns with ±1 Standard Deviation Band for {t}", f"{t} ±1 Standard Deviation Band")

        return {
            "std_dev": std,
            "plot": fig
        }   
    
class StockRiskCalculator:
    def __init__(self, asset, data):
        self.fm = FinancialMetrics(asset=asset, data=data)
        self.asset = asset
        self.data = data

    def calculate_risk(self):
        weights = {
            'std': random.uniform(0.1, 0.3),  # Standard Deviation Weight
            'beta': random.uniform(0.1, 0.3),  # Beta Weight
            'sharpe': random.uniform(0.1, 0.3),  # Sharpe Ratio Weight
            'roi': random.uniform(0.1, 0.2)  # ROI Weight
        }

        std_risk = self.fm.calc_std()
        beta_risk = self.fm.get_beta()
        sharpe_risk = self.fm.sharpe_ratio()
        roi_risk = self.fm.calc_roi_percent().mean()

        # Calculate weighted risk score
        risk_score = (weights['std'] * std_risk) + (weights['beta'] * beta_risk) + (weights['sharpe'] * sharpe_risk) + (weights['roi'] * roi_risk)

        return risk_score

class PortfolioRiskCalculator:
    def __init__(self, portfolio, data):
        self.portfolio = portfolio
        self.data = data
        self.cov_matrix = None

    def calculate_risk(self):
        total_mv = 0
        for stock in self.portfolio:
            total_mv += stock.market_value

        if total_mv == 0:
            print("Warning: Total market value of the portfolio is zero, skipping risk calculation.")
            return 0
        
        weights = {}
        for stock in self.portfolio:
            if total_mv != 0: 
                stock.weight = stock.market_value / total_mv 
                weights[stock.ticker] = stock.weight

        weight_array = np.array(list(weights.values()))

        returns = np.array([self.data[stock.ticker]['Returns'].values for stock in self.portfolio]).T

        self.cov_matrix = np.cov(returns, rowvar=False)

        portfolio_variance = np.dot(weight_array, np.dot(self.cov_matrix, weight_array))

        portfolio_risk = np.sqrt(portfolio_variance)

        annualized_portfolio_risk = (portfolio_risk * np.sqrt(252)) * 100

        return annualized_portfolio_risk
class Stock:
    def __init__(self, ticker, quantity, buy_price):
        # Initialize the stock with a ticker symbol, quantity, and buy price
        self.ticker = ticker.strip().upper().replace("$", "")  # Ensure the ticker symbol is in uppercase
        print(self.ticker)
        self.quantity = quantity  # Set the quantity of shares owned
        self.buy_price = buy_price  # Set the buy price per share
        self.update_current_price()  # Call the method to update the current price of the stock

    def update_current_price(self):
        try:
            # Fetch the stock data from Yahoo Finance using the ticker symbol
            stock_data = yf.Ticker(self.ticker)
            # Get the latest closing price of the stock
            self.current_price = float(stock_data.history(period="1d")['Close'].iloc[-1])
            # Calculate the market value of the stock based on quantity and current price
            self.market_value = self.current_price * self.quantity
            # Get the company name for the stock
            self.company_name = stock_data.info['shortName']
        except:
            # In case of an error (e.g., invalid ticker), set values to None
            self.current_price = None
            self.market_value = None
            self.company_name = None

class Portfolio:
    def __init__(self):
        # Initialize an empty list to hold the stocks in the portfolio
        self.stocks = []

    def add_stock(self, ticker, quantity, buy_price):
        # Add a new stock to the portfolio by creating a Stock object
        new_stock = Stock(ticker, quantity, buy_price)
        # Update the stock's current price
        new_stock.update_current_price()
        # Append the new stock to the portfolio's list of stocks
        self.stocks.append(new_stock)

    def remove_stock(self, ticker):
        # Remove a stock from the portfolio based on the ticker symbol
        ticker = ticker.upper()  # Ensure the ticker symbol is in uppercase
        for i, stock in enumerate(self.stocks):
            if stock.ticker == ticker:
                # Delete the stock if the ticker matches
                del self.stocks[i]
                return True  # Return True to indicate successful removal
        return False  # Return False if the stock was not found

    def update_prices(self):
        # Update the current price of all stocks in the portfolio
        for stock in self.stocks:
            stock.update_current_price()

    def get_total_market_value(self):
        # Calculate and return the total market value of all stocks in the portfolio
        # Only include stocks that have a valid market value
        return sum(stock.market_value for stock in self.stocks if stock.market_value is not None)
    

class User:
    def __init__(self, user_id, username, email):
        # Initialize the user with an ID, username, and email
        self.id = user_id
        self.username = username
        self.email = email
        # Initialize a portfolio for the user
        self.portfolio = Portfolio()
class Dashboard(tk.Frame):
    def __init__(self, root, width, height, app):
        super().__init__(root)
        self.w = width  # Set the width of the dashboard
        self.h = height  # Set the height of the dashboard
        
        self.app = app  # Reference to the main app
        self.portfolio = Portfolio()  # Initialize a new portfolio

        self.create_widgets()  # Create the dashboard widgets
        self.update_portfolio_table()  # Update the portfolio table with current data

    def create_widgets(self):
        # Create the main canvas
        self.canvas = tk.Canvas(self, width=self.w, height=self.h)
        self.canvas.pack(fill="both", expand=True)

        # Create the navigation bar frame
        self.navbar = tk.Frame(self.canvas, bg="#007BFF", height=200)
        self.canvas.create_window(0, 0, window=self.navbar, anchor="nw", width=self.w)

        self.add_navbar_buttons()  # Add buttons to the navbar

        # Create a content area frame below the navbar
        self.content_area = tk.Frame(self.canvas, bg="white")
        self.canvas.create_window(0, 90, window=self.content_area, anchor="nw", width=self.w, height=self.h - 100)

        # Create the user input form in the content area
        self.create_user_input_form()

        # Create the portfolio table
        self.table_frame = tk.Frame(self.content_area)
        self.table_frame.pack(fill="both", pady=5, padx=10)

        # Create the Treeview for displaying the portfolio
        self.table = tk. ttk.Treeview(self.table_frame, columns=("Company Name", "Ticker", "Quantity", "Buy Price", "Current Price"), show="headings")
        self.table.heading("Company Name", text="Company Name")
        self.table.heading("Ticker", text="Ticker")
        self.table.heading("Quantity", text="Quantity")
        self.table.heading("Buy Price", text="Buy Price")
        self.table.heading("Current Price", text="Current Price")
        self.table.pack(fill="both", expand=True)

        # Delete button for removing a stock from the portfolio
        self.delete_button = tk.Button(self.table_frame, text="Delete", command=self.delete_selected_stock)
        self.delete_button.pack(pady=10)

    def sign_out(self):
        # Sign out the user and clear their portfolio data
        if self.app.current_user:
            self.app.save_portfolio()
            self.app.current_user.portfolio.stocks.clear()
        self.app.show_login_form()  # Show the login form

    def go_home(self):
        # Go back to the home screen
        self.app.clear_frame()
        self.app.show_dashboard()
        
    def fetch_portfolio_risk(self):
        # Fetch the portfolio risk using a risk calculator
        risk_calculator = PortfolioRiskCalculator(self.app.current_user.portfolio.stocks, self.app.global_stock_data)
        annualized_portfolio_risk = risk_calculator.calculate_risk()  
        return annualized_portfolio_risk

    def add_navbar_buttons(self):
        # Add navigation buttons to the navbar
        toggle_btn = tk.Button(self.navbar, text="☰", bg="#007BFF", fg="white", font=("Arial", 12))
        toggle_btn.pack(side="left", padx=10, pady=20)

        home_button = tk.Button(self.navbar, text="Home", fg="white", bg="#007BFF", font=("Arial", 14), command=self.go_home)
        home_button.pack(side="left", padx=10, pady=20)

        # Create a dropdown for selecting stocks
        self.stock_dropdown = ttk.Combobox(self.navbar, values=[stock.ticker for stock in self.app.current_user.portfolio.stocks])
        self.stock_dropdown.pack(side="left", padx=10, pady=10)
        self.stock_dropdown.bind("<<ComboboxSelected>>", lambda e: self.app.show_stock_details(e))

        # Sign out button
        signout_btn = tk.Button(self.navbar, text="Sign Out", bg="#007BFF", fg="white", font=("Arial", 12), command=self.sign_out)
        signout_btn.pack(side="right", padx=10, pady=20)

        # Portfolio risk label
        self.risk_label = tk.Label(self.navbar, text="Portfolio Risk: Calculating...", bg="#007BFF", font=("Arial", 12, "bold"), fg="red")
        self.risk_label.pack(side="right", padx=10, pady=20)

        self.update_risk_label()  # Update the risk label with calculated risk

    def update_risk_label(self):
        # Update the portfolio risk label with the calculated risk value
        annualized_portfolio_risk = self.fetch_portfolio_risk()
        self.risk_label.config(text=f"Portfolio Risk: {annualized_portfolio_risk:.2f}")

    def create_user_input_form(self):
        # Create the form for adding a new stock to the portfolio
        form_frame = tk.Frame(self.content_area, bg="white")
        form_frame.pack(pady=20)
        label_ticker = tk.Label(form_frame, text="Ticker:")
        label_ticker.pack(side="left", padx=5)
        self.entry_ticker = tk.Entry(form_frame)
        self.entry_ticker.pack(side="left", padx=5)

        label_quantity = tk.Label(form_frame, text="Quantity:")
        label_quantity.pack(side="left", padx=5)
        self.entry_quantity = tk.Entry(form_frame)
        self.entry_quantity.pack(side="left", padx=5)

        label_price = tk.Label(form_frame, text="Buy Price:")
        label_price.pack(side="left", padx=5)
        self.entry_price = tk.Entry(form_frame)
        self.entry_price.pack(side="left", padx=5)

        # Submit button to add stock to portfolio
        submit_button = tk.Button(form_frame, text="Add Stock", command=self.add_stock)
        submit_button.pack(pady=10)

    def add_stock(self):
        ticker = self.entry_ticker.get().strip().upper()
        quantity = self.entry_quantity.get()
        buy_price = self.entry_price.get()

        # Validate inputs
        if not quantity.isdigit() or not buy_price.replace('.', '', 1).isdigit():
            messagebox.showerror("Error", "Invalid quantity or price.")
            return

        quantity = int(quantity)
        buy_price = float(buy_price)

        # Check if ticker is valid
        new_stock = Stock(ticker, quantity, buy_price)
        if new_stock.current_price is None:
            messagebox.showerror("Error", f"Invalid ticker: {ticker} or data unavailable.")
            return

        # Add to portfolio
        self.portfolio.add_stock(ticker, quantity, buy_price)
        self.update_portfolio_table()
        self.app.save_portfolio()
        self.app.load_portfolio()
        self.update_stock_dropdown()
        self.update_risk_label()
    def update_stock_dropdown(self):
        # Update the stock dropdown list with the latest portfolio stocks
        self.stock_dropdown.delete(0, tk.END)
        self.stock_dropdown['values'] = [stock.ticker for stock in self.app.current_user.portfolio.stocks]

    def delete_selected_stock(self):
        # Delete the selected stock from the portfolio
        selected_items = self.table.selection()
        if not selected_items:
            messagebox.showerror("Error", "Please select a stock to delete.")
            return
        selected_item = selected_items[0]
        ticker = self.table.item(selected_item)['values'][1]
        if not self.portfolio.remove_stock(ticker):
            messagebox.showerror("Error", f"Failed to remove stock {ticker} from portfolio.")
            return
        try:
            # Delete the stock from the database
            self.app.cursor.execute(
                "DELETE FROM portfolios WHERE user_id = ? AND ticker = ?",
                (self.app.current_user.id, ticker)
            )
            self.app.conn.commit()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete stock {ticker} from database: {e}")
            return
        self.update_portfolio_table()  # Refresh the portfolio table
        self.update_stock_dropdown()  # Update the stock dropdown
        self.update_risk_label()  # Update the risk label

        messagebox.showinfo("Success", f"Stock {ticker} successfully deleted.")  # Show success message

    def update_portfolio_table(self):
        # Update the portfolio table with the latest stock data
        self.table.delete(*self.table.get_children())  # Clear existing data
        for stock in self.portfolio.stocks:
            # Insert the new stock data into the table
            # Handle None or invalid current_price
            current_price = stock.current_price
            display_price = "N/A" if current_price is None else round(current_price, 2)
            self.table.insert("", "end", values=(stock.company_name, stock.ticker, stock.quantity, stock.buy_price, display_price))
        self.portfolio.update_prices()  # Update the prices of all stocks in the portfolio

class StockPortfolioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.width = self.winfo_screenwidth()  # Get the screen width
        self.height = self.winfo_screenheight()  # Get the screen height
        self.title("Stock Portfolio App")  # Set the window title
        self.geometry("800x600")  # Set window size

        self.global_stock_data = {}  # Store global stock data for portfolio

        self.conn = sqlite3.connect('users.db')  # Connect to the SQLite database
        self.cursor = self.conn.cursor()  # Create a cursor for database queries
        self.create_tables()  # Initialize the database tables

        self.current_user = None  # Placeholder for current logged-in user
        self.show_login_form()  # Display the login form

    def clear_frame(self):
        for widget in self.winfo_children():
            widget.destroy()  # Destroys all widgets in the current frame

    def update_risk_information(self, asset, data):
        risk_calculator = StockRiskCalculator(asset, data)
        risk_value = risk_calculator.calculate_risk()  # Calculate the risk of the asset
        self.risk_value_label.config(text=f"{risk_value:.2f}%")  # Update the UI label with risk percentage

    def display_plot_in_canvas(self, content_area, plot_figure, analysis_text):
        for widget in content_area.winfo_children():
            widget.destroy()  # Clear previous content

        plot_and_text_frame = tk.Frame(content_area)  # Frame to hold plot and text
        plot_and_text_frame.pack(fill=tk.BOTH, expand=True)

        canvas = FigureCanvasTkAgg(plot_figure, master=plot_and_text_frame)  # Embed the plot on canvas
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        analysis_label = tk.Label(plot_and_text_frame, text=analysis_text, justify=tk.LEFT)  # Display analysis text
        analysis_label.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

    def show_stock_details(self, event):
        selected_stock = self.dashboard.stock_dropdown.get()  # Get the selected stock ticker
        asset = []
        tickers=[]
        std =[]  # Standard deviation values for stocks
        beta=[]  # Beta values for stocks

        # Clear previous content
        for widget in self.dashboard.content_area.winfo_children():
            widget.destroy()

        for stock in self.current_user.portfolio.stocks:
            asset1 = []
            tickers.append(stock.ticker)
            asset1.append({
                'ticker':stock.ticker,
                'buy_price': stock.buy_price,
                'company_name': stock.company_name,
                'current_price': stock.current_price,
                'market_value': stock.market_value,
                'quantity': stock.quantity,
            })
            fm = FinancialMetrics(asset1[0], self.global_stock_data[stock.ticker])
            std.append(fm.calc_std())  # Calculate standard deviation
            beta.append(fm.get_beta())  # Calculate beta

        for stock in self.current_user.portfolio.stocks:
            if stock.ticker == selected_stock:
                selected_stock_data = stock  # Select the stock data for the chosen stock
                asset.append({
                    'ticker':stock.ticker,
                    'buy_price': stock.buy_price,
                    'company_name': stock.company_name,
                    'current_price': stock.current_price,
                    'market_value': stock.market_value,
                    'quantity': stock.quantity,
                })
        
        # Display stock information on the UI (company name, current price, ROI, risk, etc.)
        stock_name_label = tk.Label(self.dashboard.content_area, text=selected_stock_data.company_name, font=("Arial", 16))
        stock_name_label.pack(pady=10)

        current_stock_data = yf.Ticker(selected_stock).info  # Fetch live stock data using yfinance
        current_price = current_stock_data['currentPrice']
        current_roi = ((current_price - selected_stock_data.buy_price) / selected_stock_data.buy_price) * 100  # Calculate ROI

        fm = FinancialMetrics(asset[0], self.global_stock_data[selected_stock])
        beta = fm.get_beta()  # Get the stock's beta

        # Create frames to display stock data
        stock_info_frame = tk.Frame(self.dashboard.content_area)
        stock_info_frame.pack(pady=10)

        ticker_frame = tk.Frame(stock_info_frame, padx=10)
        ticker_frame.pack(side="left", padx=10)

        ticker_label = ttk.Label(ticker_frame, text=f"{selected_stock} - ${current_price:.2f}", font=("Arial", 12))
        ticker_label.pack(side="left")

        roi_frame = tk.Frame(stock_info_frame, padx=10)
        roi_frame.pack(side="left", padx=10)

        roi_label = ttk.Label(roi_frame, text=f"ROI: {current_roi:.2f}%", font=("Arial", 12))
        roi_label.pack(side="left")

        risk_frame = tk.Frame(stock_info_frame, padx=10)
        risk_frame.pack(side="left", padx=10)

        risk_label = tk.Label(risk_frame, text="Risk:", font=("Arial", 12, "bold"), fg="red")
        risk_label.pack(side="left")

        self.risk_value_label = tk.Label(risk_frame, text="Calculating...", font=("Arial", 12, "bold"), fg="red")
        self.risk_value_label.pack(side="left")

        btns_frame = tk.Frame(self.dashboard.content_area)
        btns_frame.pack(pady=10)

        financial_analysis = FinancialAnalysis(asset[0], self.global_stock_data[selected_stock])
        std_analysis_results = financial_analysis.std_analysis()
        compare = financial_analysis.comparison()
        returns_plot = financial_analysis.returns_stock()
        returns_dist = financial_analysis.return_dist()
        std_compare = financial_analysis.std_comparison(tickers=tickers, std=std)

        stock_canvas_frame = tk.Frame(self.dashboard.content_area)
        stock_canvas_frame.pack(pady=10)

        std_dev_button = tk.Button( btns_frame, text=f"Standard Deviation: {std_analysis_results['std_dev']:.2f}%", command=lambda: self.display_plot_in_canvas(stock_canvas_frame, std_analysis_results['plot'], "This is the standard deviation analysis text."))

        std_compare_button = tk.Button( btns_frame, text=f"Standard Deviation of all Plots", command=lambda: self.display_plot_in_canvas(stock_canvas_frame, std_compare['plot'], "This is the standard deviation analysis text."))
       
        returns_button = tk.Button( btns_frame, text=f"Returns Plot", command=lambda: self.display_plot_in_canvas(stock_canvas_frame, returns_plot['plot'], "This is the standard deviation analysis text."))

        returns_dist_button = tk.Button( btns_frame, text=f"Daily Returns Distribution Plot", command=lambda: self.display_plot_in_canvas(stock_canvas_frame, returns_dist['plot'], "This is the standard deviation analysis text."))

        comparison_button = tk.Button( btns_frame, text=f"Comparison with Benchmark Stock (S&P 500)", command=lambda: self.display_plot_in_canvas(stock_canvas_frame, compare['plot'], "This is the comparison analysis text."))

        returns_dist_button.pack(side="left", anchor="w", padx=5, pady=5)
        returns_button.pack(side="left", anchor="w", padx=5, pady=5)
        std_dev_button.pack(side="left", anchor="w", padx=5, pady=5)
        std_compare_button.pack(side="left", anchor="w", padx=5, pady=5)
        comparison_button.pack(side="left", anchor="w", padx=5, pady=5)

        self.update_risk_information(asset[0], self.global_stock_data[selected_stock])


    def login(self):
        username = self.username_entry.get()  # Get the username
        password = self.password_entry.get()  # Get the password
        self.cursor.execute("SELECT id, email FROM users WHERE username = ? AND password = ?", 
                            (username, self.hash_password(password)))  # Validate user credentials
        row = self.cursor.fetchone()
        if row:
            self.current_user = User(row[0], username, row[1])  # Load the current user
            self.load_portfolio()  # Load the user's portfolio
            messagebox.showinfo("Success", "Login successful!")  # Show success message
            self.clear_frame()
            self.show_dashboard()  # Display dashboard after login
        else:
            messagebox.showerror("Error", "Invalid username or password.")  # Show error if login fails

    def signup(self):
        username = self.username_entry.get()  # Get the username
        email = self.email_entry.get()  # Get the email
        password = self.password_entry.get()  # Get the password
        
        # Validate the username, email, and password inputs
        if not self.validate_username(username):
            messagebox.showerror("Invalid Username", "Username can only contain alphanumeric characters and underscores.")
            return
        if not self.validate_email(email):
            messagebox.showerror("Invalid Email", "Email must be valid (e.g., example@domain.com).")
            return
        if not self.validate_password(password):
            messagebox.showerror("Invalid Password", "Password must be at least 8 characters long, include an uppercase letter, a number, and one special character (_, -).")
            return
        
        try:
            self.cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                                (username, email, self.hash_password(password)))  # Insert new user into the database
            self.conn.commit()
            messagebox.showinfo("Success", "Account created successfully!")
            self.show_login_form()  # Show the login form after successful signup
        except sqlite3.IntegrityError:
            messagebox.showerror("Error", "Username or email already exists.")  # Error if the username/email already exists

    def load_portfolio(self):
        if self.current_user:
            self.current_user.portfolio.stocks.clear()  # Clear existing portfolio data
            self.cursor.execute("SELECT ticker, quantity, buy_price FROM portfolios WHERE user_id = ?", (self.current_user.id,))  # Fetch user's portfolio from database
            for row in self.cursor.fetchall():
                # Check if the stock already exists in the portfolio
                if not any(stock.ticker == row[0] for stock in self.current_user.portfolio.stocks):
                    self.current_user.portfolio.add_stock(row[0], row[1], row[2])  # Add stock to the portfolio

        # Fetch data for each stock in the portfolio
        for stock in self.current_user.portfolio.stocks:
            if stock.ticker not in self.global_stock_data:
                stock_data = fetch_stock_data(stock.ticker)  # Fetch stock data from an external source
                self.global_stock_data[stock.ticker] = stock_data

    def save_portfolio(self):
        if self.current_user:
            self.cursor.execute("DELETE FROM portfolios WHERE user_id = ?", (self.current_user.id,))  # Clear the existing portfolio
            for stock in self.current_user.portfolio.stocks:
                self.cursor.execute("INSERT INTO portfolios (user_id, ticker, quantity, buy_price) VALUES (?, ?, ?, ?)",
                                    (self.current_user.id, stock.ticker, stock.quantity, stock.buy_price))  # Insert updated portfolio into the database
            self.conn.commit()

    def show_login_form(self):
        self.clear_frame()# Clear any previous frames
        # Create login form with entry fields and buttons

        login_label = tk.Label(self, text="Login", font=("Arial", 16))
        login_label.pack(pady=10)

        username_label = tk.Label(self, text="Username:")
        username_label.pack()
        self.username_entry = tk.Entry(self)
        self.username_entry.pack()

        password_label = tk.Label(self, text="Password:")
        password_label.pack()
        self.password_entry = tk.Entry(self, show="*")
        self.password_entry.pack()

        login_button = tk.Button(self, text="Login", command=self.login)
        login_button.pack(pady=10)

        signup_label = tk.Label(self, text="Don't have an account? Sign Up")
        signup_label.pack()
        signup_label.bind("<Button-1>", lambda e: self.show_signup_form())

    def show_signup_form(self):
        self.clear_frame()# Clear any previous frames
        # Create signup form with entry fields and buttons

        signup_label = tk.Label(self, text="Sign Up", font=("Arial", 16))
        signup_label.pack(pady=10)

        username_label = tk.Label(self, text="Username:")
        username_label.pack()
        self.username_entry = tk.Entry(self)
        self.username_entry.pack()

        email_label = tk.Label(self, text="Email:")
        email_label.pack()
        self.email_entry = tk.Entry(self)
        self.email_entry.pack()

        password_label = tk.Label(self, text="Password:")
        password_label.pack()
        self.password_entry = tk.Entry(self, show="*")
        self.password_entry.pack()

        signup_button = tk.Button(self, text="Sign Up", command=self.signup)
        signup_button.pack(pady=10)

        login_label = tk.Label(self, text="Already have an account? Login")
        login_label.pack()
        login_label.bind("<Button-1>", lambda e: self.show_login_form())

    def show_dashboard(self):
        # Create a new Dashboard instance with the current app size
        self.dashboard = Dashboard(self, self.width, self.height, self)  # Pass width and height here
        
        # Pack the dashboard to fill the available space
        self.dashboard.pack(fill="both", expand=True)

        # If a user is logged in, update the dashboard with their portfolio
        if self.current_user:
            self.dashboard.portfolio = self.current_user.portfolio
            # Update the portfolio table with the user's portfolio data
            self.dashboard.update_portfolio_table()
    def create_tables(self):
        # Create a 'users' table with columns for ID, username, email, and password
        self.cursor.execute(''' 
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                email TEXT UNIQUE,
                password TEXT
            )
        ''')
        # Create a 'portfolios' table with columns for ID, user ID, ticker symbol, quantity, and buy price
        self.cursor.execute(''' 
            CREATE TABLE IF NOT EXISTS portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id),
                ticker TEXT,
                quantity INTEGER,
                buy_price REAL
            )
        ''')
        # Save the changes made to the database
        self.conn.commit()

    def hash_password(self, password):
        # Hash the password using SHA-256 and return the hashed value
        return hashlib.sha256(password.encode()).hexdigest()

    def validate_username(self, username):
        # Check if the username contains only alphanumeric characters and underscores
        return re.match(r'^[a-zA-Z0-9_]+$', username) is not None

    def validate_email(self, email):
        # Check if the email matches a basic email format (e.g., name@example.com)
        return re.match(r'^[^@]+@[^@]+\.[^@]+$', email) is not None

    def validate_password(self, password):
        # Check if the password contains at least one uppercase letter, one number, one special character, and is at least 8 characters long
        return (re.search(r'[A-Z]', password) and
                re.search(r'[0-9]', password) and
                re.search(r'[_\-]', password) and
                len(password) >= 8)

if __name__ == "__main__":
    # Create an instance of the app and start the main event loop
    app = StockPortfolioApp()
    app.mainloop()