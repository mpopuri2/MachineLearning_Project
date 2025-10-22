from __future__ import annotations
import pandas as pd
import yfinance as yf
from typing import List


def download_ohlcv(symbol: str, start: str, end: str, interval: str = "1h") -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {symbol} {start}..{end} interval={interval}")
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    df = df.dropna().copy()
    df.index = pd.to_datetime(df.index)
    return df


def concat_symbols(symbols: List[str], start: str, end: str, interval: str) -> pd.DataFrame:
    frames = [download_ohlcv(s, start, end, interval).assign(symbol=s) for s in symbols]
    return pd.concat(frames).sort_index()