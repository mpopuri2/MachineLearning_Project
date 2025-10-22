from __future__ import annotations
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    
    if not all(col in out.columns for col in ["close", "high", "low"]):
        raise ValueError("DataFrame must contain columns: close, high, low")

    out["ret1"] = out["close"].pct_change()

    # --- RSI ---
    try:
        rsi = RSIIndicator(out["close"].squeeze(), window=14)
        out["rsi"] = rsi.rsi()
    except Exception as e:
        print("RSI failed:", e)
        out["rsi"] = np.nan

    # --- MACD ---
    try:
        macd = MACD(out["close"].squeeze())
        out["macd"] = macd.macd()
        out["macd_signal"] = macd.macd_signal()
        out["macd_hist"] = macd.macd_diff()
    except Exception as e:
        print("MACD failed:", e)
        out["macd"], out["macd_signal"], out["macd_hist"] = np.nan, np.nan, np.nan

    # --- Stochastic Oscillator ---
    try:
        stoch = StochasticOscillator(out["high"].squeeze(), out["low"].squeeze(), out["close"].squeeze(), window=14, smooth_window=3)
        out["stoch_k"] = stoch.stoch()
        out["stoch_d"] = stoch.stoch_signal()
    except Exception as e:
        print("Warning: Stochastic oscillator failed:", e)
        out["stoch_k"], out["stoch_d"] = np.nan, np.nan

    # --- Bollinger Bands ---
    try:
        bb = BollingerBands(out["close"].squeeze(), window=20, window_dev=2)
        out["bb_high"] = bb.bollinger_hband()
        out["bb_low"] = bb.bollinger_lband()
        
        out["bb_pct"] = (out["close"].squeeze() - out["bb_low"]) / (out["bb_high"] - out["bb_low"] + 1e-8)
    except Exception as e:
        print("Bollinger Bands failed:", e)
        out["bb_high"], out["bb_low"], out["bb_pct"] = np.nan, np.nan, np.nan

     # --- On-Balance Volume (OBV) ---
    try:
        obv = (np.sign(out["close"].diff()) * out["volume"]).fillna(0).cumsum()
        out["obv"] = obv
    except Exception as e:
        print("OBV failed:", e)
        out["obv"] = np.nan

    # --- Exponential Moving Average (EMA) ---
    try:
        out["ema_20"] = out["close"].ewm(span=20, adjust=False).mean()
        out["ema_50"] = out["close"].ewm(span=50, adjust=False).mean()
    except Exception as e:
        print("EMA failed:", e)
        out["ema_20"], out["ema_50"] = np.nan, np.nan

    # --- Average True Range (ATR) ---
    try:
        high_low = out["high"] - out["low"]
        high_close = np.abs(out["high"] - out["close"].shift())
        low_close = np.abs(out["low"] - out["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        out["atr"] = tr.rolling(window=14).mean()
    except Exception as e:
        print("ATR failed:", e)
        out["atr"] = np.nan
    out = out.dropna().copy()
    return out


def make_state_matrix(df: pd.DataFrame, cols: list[str], window: int, normalize: bool=True) -> np.ndarray:
    if not all(col in df.columns for col in cols):
        missing_cols = [col for col in cols if col not in df.columns]
        raise ValueError(f"DataFrame is missing required feature columns: {missing_cols}")
        
    if len(df) < window:
        raise ValueError(f"Insufficient data: len(df)={len(df)} is less than window={window}")
        
    X = []
    for i in range(window, len(df) + 1):
        window_df = df.iloc[i-window:i]
        arr = window_df[cols].values.astype(np.float32)
        
        if normalize:
            arr_mean = arr.mean(axis=0, keepdims=True)
            arr_std = arr.std(axis=0, keepdims=True)
            arr = (arr - arr_mean) / (arr_std + 1e-8)
            
        X.append(arr)
        
    if len(X) == 0:
        raise ValueError(f"No valid feature windows: len(df)={len(df)}, window={window}")
        
    return np.stack(X)