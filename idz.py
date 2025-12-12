import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Iterator, Iterable
import math
import plotly.express as px

def download_market_data(start: str = "2015-01-01", end: str = "2025-04-13") -> pd.DataFrame:
    tickers = {
        "IBM": "IBM",
        "S&P500": "^GSPC",
        "Dow30": "^DJI",
        "Gold": "GC=F",
        "CrudeOil": "CL=F"
    }

    df = pd.DataFrame()

    for name, ticker in tickers.items():
        data = yf.download(ticker, start=start, end=end, progress=False)["Close"]
        df[name] = data

    return df

# Очищення
def clear_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna().copy()

# Нормалізація
def minmax_normalize(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    scaled = (numeric - numeric.min()) / (numeric.max() - numeric.min())
    return scaled.astype(float).copy()

# Описова статистика
def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe().T.copy()

# Створює univariate X, y
def create_univariate_dataset(series: Iterable[float], window_size: int, forecast_horizon: int):
    series = list(series)
    X, y = [], []

    for i in range(len(series) - window_size - forecast_horizon + 1):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size : i + window_size + forecast_horizon])

    return np.array(X), np.array(y)

# Multivariate датасет
def create_multivariate_dataset(df: pd.DataFrame, target_col: str,
                                window_size: int, forecast_horizon: int):
    data = df.values
    target_idx = df.columns.get_loc(target_col)

    X, y = [], []

    for i in range(len(df) - window_size - forecast_horizon + 1):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size : i + window_size + forecast_horizon, target_idx])

    return np.array(X), np.array(y)

@dataclass
class PricePoint:
    asset: str
    date: any
    close: float

@dataclass
class ReturnPoint:
    asset: str
    date: any
    log_return: float

def iter_market_data(start="2015-01-01", end="2025-04-13"):
    tickers = {
        "IBM": "IBM",
        "S&P500": "^GSPC",
        "Dow30": "^DJI",
        "Gold": "GC=F",
        "CrudeOil": "CL=F",
    }

    for asset, ticker in tickers.items():
        df = yf.download(ticker, start=start, end=end, progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            if ("Close", ticker) not in df.columns:
                print(f"No Close column for {ticker}")
                continue
            close = df[("Close", ticker)].dropna()

        # Старий формат — звичайна колонка Close
        else:
            if "Close" not in df:
                print(f"No Close column for {ticker}")
                continue
            close = df["Close"].dropna()

        for date, price in close.items():
            if hasattr(price, "iloc"):
                price = float(price.iloc[0])
            else:
                price = float(price)

            yield PricePoint(asset=asset, date=date, close=price)

def iter_log_returns(points):
    by_asset = {}

    # Групування по активу
    for p in points:
        by_asset.setdefault(p.asset, []).append(p)

    # Функція вищого порядку. Обчислення доходів між парами
    def apply_to_pairs(plist, func):
        for i in range(1, len(plist)):
            yield func(plist[i - 1], plist[i])

    # log returns
    for asset, plist in by_asset.items():
        plist.sort(key=lambda x: x.date)

        for ret_point in apply_to_pairs(plist, lambda prev, curr: ReturnPoint(
            asset=curr.asset,
            date=curr.date,
            log_return=math.log(curr.close / prev.close) if curr.close / prev.close > 0 else 0
        )):
            if ret_point.log_return != 0:
                yield ret_point

def returns_to_dataframe(points):
    rows = [{
        "Asset": p.asset,
        "Date": p.date,
        "Return": p.log_return
    } for p in points]

    return pd.DataFrame(rows)

if __name__ == "__main__":

    # Завантаження
    df = download_market_data()

    px.line(df, x=df.index, y=df.columns).show()

    # Очищення
    df_clean = clear_data(df)

    # Нормалізація
    df_norm = minmax_normalize(df_clean)

    # Статистика
    stats = get_descriptive_stats(df_clean)

    # Univariate
    X_uni, y_uni = create_univariate_dataset(df_clean["IBM"], 20, 1)

    # Multivariate
    X_multi, y_multi = create_multivariate_dataset(df_clean, "IBM", 20, 3)

    # Lazy лог-прибутки
    ret_df = returns_to_dataframe(iter_log_returns(iter_market_data()))

    print("Оригінальні дані:")
    print(df_clean.head())

    print("\nНормалізовані дані:")
    print(df_norm.head())

    print("\nОписова статистика:")
    print(stats)

    print("\nUnivariate X shape:", X_uni.shape, "y shape:", y_uni.shape)
    print("Multivariate X shape:", X_multi.shape, "y shape:", y_multi.shape)

    print("\nLazy Log Returns:")
    print(ret_df.head())
