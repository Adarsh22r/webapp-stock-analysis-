import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from prophet import Prophet
import tensorflow as tf
from datetime import datetime, timedelta

class StockPredictor:
    def __init__(self):
        self.lstm_model = None
        self.prophet_model = None
        self.scaler = MinMaxScaler()
        self.prediction_history = []
        self.last_training_time = None
        
    def prepare_data(self, df, lookback=24):  # Changed to 24 hours
        # Prepare data for LSTM
        data = df['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
            
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split data into train and test
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, X_test, y_train, y_test
    
    def build_lstm_model(self, lookback=24):  # Changed to 24 hours
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train_lstm(self, df, lookback=24):  # Changed to 24 hours
        X_train, X_test, y_train, y_test = self.prepare_data(df, lookback)
        self.lstm_model = self.build_lstm_model(lookback)
        
        # Train the model
        self.lstm_model.fit(
            X_train, y_train,
            epochs=20,  # Reduced epochs for faster training
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )
        
        # Make predictions
        lstm_predictions = self.lstm_model.predict(X_test)
        lstm_predictions = self.scaler.inverse_transform(lstm_predictions)
        y_test_original = self.scaler.inverse_transform([y_test])
        
        return lstm_predictions, y_test_original.T
    
    def train_prophet(self, df):
        # Prepare data for Prophet
        prophet_df = df.reset_index()[['Date', 'Close']]
        prophet_df.columns = ['ds', 'y']
        
        # Remove timezone information if present
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds']).dt.tz_localize(None)
        
        # Train Prophet model
        self.prophet_model = Prophet(daily_seasonality=True, hourly_seasonality=True)
        self.prophet_model.fit(prophet_df)
        
        # Make future predictions
        future = self.prophet_model.make_future_dataframe(periods=24)  # Predict next 24 hours
        prophet_predictions = self.prophet_model.predict(future)
        
        return prophet_predictions
    
    def predict_next_hour(self, df):
        current_time = datetime.now()
        
        # Check if we need to retrain (every 24 hours)
        if self.last_training_time is None or (current_time - self.last_training_time).total_seconds() > 24*3600:
            # Get LSTM predictions
            lstm_pred, _ = self.train_lstm(df)
            
            # Get Prophet predictions
            prophet_pred = self.train_prophet(df)
            
            self.last_training_time = current_time
            
            # Store the predictions
            self.prediction_history.append({
                'timestamp': current_time,
                'lstm_pred': lstm_pred[-1][0],
                'prophet_pred': prophet_pred['yhat'].iloc[-1],
                'actual': df['Close'].iloc[-1]
            })
            
            # Keep only last 24 predictions
            if len(self.prediction_history) > 24:
                self.prediction_history = self.prediction_history[-24:]
        
        # Calculate next hour prediction
        next_hour = current_time + timedelta(hours=1)
        
        # Prepare prediction data
        prediction_data = pd.DataFrame({
            'Date': [next_hour],
            'LSTM_Prediction': [self.prediction_history[-1]['lstm_pred']],
            'Prophet_Prediction': [self.prediction_history[-1]['prophet_pred']]
        })
        
        # Calculate ensemble prediction
        prediction_data['Ensemble_Prediction'] = prediction_data[['LSTM_Prediction', 'Prophet_Prediction']].mean(axis=1)
        
        return prediction_data
    
    def get_prediction_accuracy(self):
        if len(self.prediction_history) < 2:
            return None
        
        # Calculate accuracy metrics
        actual_values = [p['actual'] for p in self.prediction_history]
        predicted_values = [p['lstm_pred'] for p in self.prediction_history]
        
        # Calculate Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((np.array(actual_values) - np.array(predicted_values)) / np.array(actual_values))) * 100
        
        return {
            'mape': mape,
            'history': self.prediction_history
        } 