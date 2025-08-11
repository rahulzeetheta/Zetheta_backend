import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

def load_lstm_model(model_path: str = "model/lstm_market_model.h5"):
    return load_model(model_path)

# ✅ Corrected: Fetch all required features
def fetch_data(ticker: str, start="2022-01-01", end="2024-12-31"):
    df = yf.download(ticker, start=start, end=end, interval='1d', auto_adjust=False)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df.values  # shape: (n_days, 5 features)

# ✅ Corrected: Preprocess to produce 3D inputs with 5 features
def preprocess(data, time_step: int = 100):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X = []
    y = []

    for i in range(len(data_scaled) - time_step - 1):
        X.append(data_scaled[i:(i + time_step)])
        y.append(data_scaled[i + time_step, 3])  # predict 'Close'

    X = np.array(X)  # shape: (samples, time_step, 5)
    y = np.array(y)

    return X, scaler

# Predict using LSTM model
def predict_next(model, X_input):
    prediction = model.predict(X_input)
    return prediction