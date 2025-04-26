# Indian Stock Market Analyzer

A comprehensive web application for analyzing Indian stock market data, including historical prices, technical indicators, market sentiment, and price predictions.

## Features

- Historical price data visualization with candlestick charts
- Technical indicators (RSI, MACD)
- Volume analysis
- Market sentiment analysis
- Price prediction capabilities
- Interactive date range selection
- Real-time stock data updates

## Setup Instructions

1. Create and activate a Conda environment:
```bash
conda create -n stock_analysis python=3.9
conda activate stock_analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Enter a stock symbol in the sidebar (e.g., RELIANCE.NS for Reliance Industries)
2. Select your desired date range
3. View various analyses including:
   - Current price and daily changes
   - Candlestick charts
   - Technical indicators
   - Volume analysis
   - Market sentiment
   - Price predictions

## Stock Symbol Format

For Indian stocks, append `.NS` to the stock symbol. Examples:
- RELIANCE.NS (Reliance Industries)
- TCS.NS (Tata Consultancy Services)
- INFY.NS (Infosys)
- HDFCBANK.NS (HDFC Bank)

## Note

This application uses the Yahoo Finance API for stock data. Please ensure you have a stable internet connection for optimal performance. 