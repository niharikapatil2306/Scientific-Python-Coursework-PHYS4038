import unittest
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import pandas as pd

from unittest.mock import patch, MagicMock

from main import FinancialMetrics, StockRiskCalculator, PortfolioRiskCalculator, fetch_stock_data

class TestFetchStockData(unittest.TestCase):
    
    def test_valid_ticker(self):
        # Test for a valid ticker (Apple stock in this case)
        ticker = "AAPL"
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        
        result = fetch_stock_data(ticker, start_date, end_date)
        
        # Check if the returned DataFrame has the necessary columns and data
        self.assertIsNotNone(result)
        self.assertIn('Date', result.columns)
        self.assertIn('Adj Close', result.columns)
        self.assertIn('Returns', result.columns)
        self.assertGreater(len(result), 0)  # Ensure the DataFrame is not empty

    def test_no_data_found(self):
        # Test for a ticker that is unlikely to exist
        ticker = "INVALID"
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        
        result = fetch_stock_data(ticker, start_date, end_date)
        
        # The result should be None if no data is found
        self.assertIsNone(result)

    def test_empty_data_frame(self):
        # Simulate empty data (for example, a date range with no data)
        ticker = "AAPL"
        start_date = "2023-01-01"
        end_date = "2023-01-02"  # Short date range, likely to have no stock data
        
        result = fetch_stock_data(ticker, start_date, end_date)
        
        # The result should be None or an empty DataFrame
        self.assertIsNone(result)

    def test_default_dates(self):
        # Test the function with default dates
        ticker = "AAPL"
        result = fetch_stock_data(ticker)  # Uses default start_date and end_date

        # Check if data is returned and has columns
        self.assertIsNotNone(result)
        self.assertIn('Date', result.columns)
        self.assertIn('Adj Close', result.columns)
        self.assertIn('Returns', result.columns)

class TestFinancialMetrics(unittest.TestCase):

    @patch('yfinance.Ticker')
    def setUp(self, MockTicker):
        self.asset = {
            'ticker': 'AAPL',
            'buy_price': 150
        }
        self.data = pd.DataFrame({
            'Close': [155, 158, 160, 162, 164],
            'Returns': [0.03, 0.02, 0.01, 0.015, 0.025]
        })
        self.mock_stock = MagicMock()
        self.mock_stock.info = {'beta': 1.2, 'returnOnAssets': 0.05, 'returnOnEquity': 0.1}
        MockTicker.return_value = self.mock_stock
        self.fm = FinancialMetrics(self.asset, self.data)

    def test_calc_roi(self):
        expected_roi = [155 - 150, 158 - 150, 160 - 150, 162 - 150, 164 - 150]
        self.assertListEqual(self.fm.data['ROI'].tolist(), expected_roi)

    def test_calc_roi_percent(self):
        expected_roi_percent = [(155 - 150) / 150 * 100, (158 - 150) / 150 * 100, 
                                (160 - 150) / 150 * 100, (162 - 150) / 150 * 100, 
                                (164 - 150) / 150 * 100]
        self.assertListEqual(np.round(self.fm.data['ROI (%)'].tolist(), 2).tolist(),
                             np.round(expected_roi_percent, 2).tolist())

    def test_calc_std(self):
        expected_std = (self.data['Returns'] * 100).std()
        self.assertEqual(self.fm.calc_std(), expected_std)

    def test_calc_var(self):
        expected_var = (self.data['Returns'] * 100).var()
        self.assertEqual(self.fm.calc_var(), expected_var)

    def test_get_beta(self):
        self.assertEqual(self.fm.get_beta(), 1.2)

    def test_avg_daily_returns(self):
        expected_avg = (self.data['Returns'] * 100).mean()
        self.assertEqual(self.fm.avg_daily_returns(), expected_avg)

    def test_calc_var_risk(self):
        mu = self.fm.avg_daily_returns()
        sigma = self.fm.calc_std() / 100
        z_score = stats.norm.ppf(1 - (1 - 0.95))
        expected_var_risk = mu - z_score * sigma
        self.assertEqual(self.fm.calc_var_risk(), expected_var_risk)

    def test_get_risk_free_rate(self):
        with patch('yfinance.Ticker') as MockTicker:
            mock_tnx = MagicMock()
            mock_tnx.history.return_value = pd.DataFrame({'Close': [3.5]})
            MockTicker.return_value = mock_tnx
            self.assertEqual(self.fm.get_risk_free_rate(), 3.5)

    def test_sharpe_ratio(self):
        sharpe = (self.fm.avg_daily_returns() - self.fm.get_risk_free_rate()) / self.fm.calc_std()
        self.assertEqual(self.fm.sharpe_ratio(), sharpe)

    def test_calc_bands(self):
        avg_return = self.fm.avg_daily_returns()
        std_dev = self.fm.calc_std()
        upper_band = avg_return + 1 * std_dev
        lower_band = avg_return - 1 * std_dev
        self.assertEqual(self.fm.calc_bands(), (upper_band, lower_band))


class TestStockRiskCalculator(unittest.TestCase):

    def setUp(self):
        self.asset = {
            'ticker': 'AAPL',
            'buy_price': 150
        }
        self.data = pd.DataFrame({
            'Close': [155, 158, 160, 162, 164],
            'Returns': [0.03, 0.02, 0.01, 0.015, 0.025]
        })
        self.fm_mock = MagicMock()
        self.fm_mock.calc_std.return_value = 2.5
        self.fm_mock.get_beta.return_value = 1.2
        self.fm_mock.sharpe_ratio.return_value = 1.5
        self.fm_mock.calc_roi_percent.return_value = self.data['Returns'] * 100
        self.fm = StockRiskCalculator(self.asset, self.data)

    def test_calculate_risk(self):
        risk_score = self.fm.calculate_risk()

        self.assertIsInstance(risk_score, float)
        self.assertGreaterEqual(risk_score, 0)

    def test_calculate_risk_with_mocked_values(self):
        self.fm_mock.calc_std.return_value = 3.0
        self.fm_mock.get_beta.return_value = 1.1
        self.fm_mock.sharpe_ratio.return_value = 1.8
        self.fm_mock.calc_roi_percent.return_value = self.data['Returns'] * 110

        risk_score = self.fm.calculate_risk()

        self.assertIsInstance(risk_score, float)
        self.assertGreaterEqual(risk_score, 0)

class TestPortfolioRiskCalculator(unittest.TestCase):

    def setUp(self):
        self.data = {
            'AAPL': pd.DataFrame({'Returns': [0.03, 0.02, 0.01, 0.015, 0.025]}),
            'GOOG': pd.DataFrame({'Returns': [0.02, 0.015, 0.01, 0.02, 0.03]}),
        }

        self.portfolio = [
            MagicMock(ticker='AAPL', market_value=1000000),
            MagicMock(ticker='GOOG', market_value=1500000)
        ]

        for stock in self.portfolio:
            stock.weight = None  # Initialize weight attribute

        self.pfm = PortfolioRiskCalculator(self.portfolio, self.data)

    def test_calculate_risk(self):
        self.pfm.calculate_risk()

        for stock in self.portfolio:
            self.assertIsNotNone(stock.weight)

    def test_calculate_risk_with_cov_matrix(self):
        annualized_risk = self.pfm.calculate_risk()

        self.assertIsInstance(annualized_risk, float)
        self.assertGreaterEqual(annualized_risk, 0)

if __name__ == "__main__":
    unittest.main()
