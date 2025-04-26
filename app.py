import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import ta
from textblob import TextBlob
import requests
import json
from ml_model import StockPredictor
import time

# Set page config
st.set_page_config(
    page_title="Indian Stock Market Analyzer",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Indian Stock Market Analyzer")
st.markdown("""
This application provides real-time analysis of Indian stocks including:
- Live price data and candlestick charts
- Technical indicators
- Hourly price predictions
- Prediction accuracy tracking
""")

# Sidebar for user inputs
st.sidebar.header("Stock Selection")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., RELIANCE.NS)", "RELIANCE.NS")

# Date range selection
end_date = datetime.now()
start_date = end_date - timedelta(days=30)  # Reduced to 30 days for more recent data
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(start_date, end_date),
    max_value=end_date
)

# Initialize session state for predictor
if 'predictor' not in st.session_state:
    st.session_state.predictor = StockPredictor()

# Fetch stock data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date, interval='1h')
    return df

try:
    df = load_stock_data(stock_symbol, date_range[0], date_range[1])
    
    # Display basic stock information
    st.header(f"Stock Information for {stock_symbol}")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Price", f"₹{df['Close'].iloc[-1]:.2f}")
    with col2:
        daily_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        st.metric("Hourly Change", f"{daily_change:.2f}%")
    with col3:
        st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")

    # Candlestick Chart
    st.subheader("Price Chart")
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Close'])])
    fig.update_layout(title=f"{stock_symbol} Price Chart",
                     yaxis_title="Price (₹)",
                     xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)

    # Technical Indicators
    st.subheader("Technical Indicators")
    
    # Calculate RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    
    # Calculate MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    # Plot technical indicators
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='OHLC'),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'),
                  row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(height=800, title_text="Price and Technical Indicators")
    st.plotly_chart(fig, use_container_width=True)

    # Real-time Price Prediction
    st.subheader("Next Hour Price Prediction")
    
    with st.spinner("Generating predictions..."):
        predictions = st.session_state.predictor.predict_next_hour(df)
        
        # Display prediction metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("LSTM Predicted Price", f"₹{predictions['LSTM_Prediction'].iloc[0]:.2f}")
        with col2:
            st.metric("Prophet Predicted Price", f"₹{predictions['Prophet_Prediction'].iloc[0]:.2f}")
        with col3:
            st.metric("Ensemble Predicted Price", f"₹{predictions['Ensemble_Prediction'].iloc[0]:.2f}")
        
        # Plot prediction history
        accuracy_data = st.session_state.predictor.get_prediction_accuracy()
        if accuracy_data:
            st.subheader("Prediction Accuracy")
            st.metric("Mean Absolute Percentage Error (MAPE)", f"{accuracy_data['mape']:.2f}%")
            
            # Plot prediction history
            history_df = pd.DataFrame(accuracy_data['history'])
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['actual'],
                name='Actual Price',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['lstm_pred'],
                name='LSTM Prediction',
                line=dict(color='red', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['prophet_pred'],
                name='Prophet Prediction',
                line=dict(color='green', dash='dash')
            ))
            
            fig.update_layout(
                title="Prediction History",
                xaxis_title="Time",
                yaxis_title="Price (₹)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.info("Please check if the stock symbol is correct and try again.") 