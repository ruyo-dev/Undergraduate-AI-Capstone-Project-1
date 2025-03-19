import yfinance as yf
import pandas as pd
import numpy as np

stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'NVDA', 'META', 'V', 'IBM', 'GS', 'NFLX', 'DIS', 'SPY', 'BA', 'PYPL', 'GOOG', 'UNH', 'JNJ', 'PG', 'JPM', \
          'MA', 'CVX' ,'ABBV', 'KO', 'TMO', 'MRK', 'CSCO','CMCSA', 'T', 'VZ', 'NKE', 'ORCL', 'LLY', 'HON', 'MCD', 'ABT','DHR', 'MDT', 'COST', 'AMGN', \
            'TXN', 'UPS', 'RTX', 'BMY', 'LIN', 'UNP', 'CAT', 'MS', 'BLK', 'SPGI']
tests = ['MMM', 'WMT', 'PFE', 'AMD', 'COIN']
# Storage for features and targets
features = []
targets = []

def collect_stock_data(ticker):
    print(f"Processing {ticker}...")

    # Download 2024 stock data (training data)
    data_2024 = yf.download(ticker, start="2024-10-01", end="2025-01-01")['Close']

    all_days_2024 = pd.date_range(start="2024-10-01", end="2024-12-31")
    data_2024 = data_2024.reindex(all_days_2024)

    # Handle missing values: forward-fill, backward-fill, fallback to mean
    data_2024.ffill(inplace=True)  # Forward-fill
    data_2024.bfill(inplace=True)  # Backward-fill
    data_2024.fillna(data_2024.mean(), inplace=True)  # Mean for edge cases

    # Download Q1 2025 stock data (target data)
    data_2025 = yf.download(ticker, start="2025-01-01", end="2025-04-01")['Close']
    avg_target = float(data_2025.mean())  # Calculate average closing price for Q1 2025

    # Find the first and last non-NaN values for year_trend
    first_valid = data_2024.first_valid_index()
    last_valid = data_2024.last_valid_index()

    if pd.isna(first_valid) or pd.isna(last_valid):
        return

    first_close = data_2024.loc[first_valid]
    last_close = data_2024.loc[last_valid]

    # Calculate year_trend: (Last - First) / First
    growth = (last_close - first_close) / first_close

    # Store features (366 days + year_trend) and targets
    features.append(np.append(data_2024.values, growth))
    targets.append([ticker, avg_target])

    print(f"✔️ {ticker} processed successfully (Target: {avg_target:.2f})")

    # Process all stocks
for stock in stocks:
    collect_stock_data(stock)

date_columns = [str(date.date()) for date in pd.date_range("2024-10-01", "2024-12-31")]
feature_columns = date_columns + ['growth']

# Save features and targets to CSV
features_df = pd.DataFrame(features, columns=feature_columns, index=stocks)
targets_df = pd.DataFrame(targets, columns=["Stock", "Avg_Close_Q1_2025"])

features_df.to_csv('stock_features.csv')
targets_df.to_csv('stock_targets.csv', index=False)

print("\n✅ Dataset creation complete!")
print("Features saved to 'stock_features.csv'")
print("Targets saved to 'stock_targets.csv'")


features = []
targets = []
for test in tests:
    collect_stock_data(test)

date_columns = [str(date.date()) for date in pd.date_range("2024-10-01", "2024-12-31")]
feature_columns = date_columns + ['growth']

# Save features and targets to CSV
features_df = pd.DataFrame(features, columns=feature_columns, index=tests)
targets_df = pd.DataFrame(targets, columns=["Stock", "Avg_Close_Q1_2025"])

features_df.to_csv('test_features.csv')
targets_df.to_csv('test_targets.csv', index=False)

print("\n✅ Testcase creation complete!")
print("Features saved to 'test_features.csv'")
print("Targets saved to 'test_targets.csv'")