# analysis/technical_indicators.py
"""
Librería de indicadores técnicos
Funciones puras para cálculo de indicadores estándar
"""
import pandas as pd
import numpy as np


def calculate_rsi(prices, periodo=14):
    """
    Calcula el Relative Strength Index (RSI)
    
    Parámetros:
    - prices: pd.Series con precios de cierre
    - periodo: int (default 14)
    
    Retorna:
    - pd.Series con valores de RSI (0-100)
    """
    # Calcular cambios de precio
    delta = prices.diff()
    
    # Separar ganancias y pérdidas
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calcular promedio de ganancias y pérdidas
    avg_gain = gain.rolling(window=periodo, min_periods=periodo).mean()
    avg_loss = loss.rolling(window=periodo, min_periods=periodo).mean()
    
    # Calcular RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Calcular RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_ema(prices, periodo=20):
    """
    Calcula la Exponential Moving Average (EMA)
    
    Parámetros:
    - prices: pd.Series con precios
    - periodo: int (default 20)
    
    Retorna:
    - pd.Series con valores de EMA
    """
    return prices.ewm(span=periodo, adjust=False).mean()


def calculate_sma(prices, periodo=20):
    """
    Calcula la Simple Moving Average (SMA)
    
    Parámetros:
    - prices: pd.Series con precios
    - periodo: int (default 20)
    
    Retorna:
    - pd.Series con valores de SMA
    """
    return prices.rolling(window=periodo).mean()


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calcula el Moving Average Convergence Divergence (MACD)
    
    Parámetros:
    - prices: pd.Series con precios de cierre
    - fast: int (periodo EMA rápida, default 12)
    - slow: int (periodo EMA lenta, default 26)
    - signal: int (periodo señal, default 9)
    
    Retorna:
    - dict con 'macd', 'signal', 'histogram'
    """
    # Calcular EMAs
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Signal line
    signal_line = calculate_ema(macd_line, signal)
    
    # Histogram
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def calculate_bollinger_bands(prices, periodo=20, std_dev=2):
    """
    Calcula las Bandas de Bollinger
    
    Parámetros:
    - prices: pd.Series con precios
    - periodo: int (default 20)
    - std_dev: float (desviaciones estándar, default 2)
    
    Retorna:
    - dict con 'upper', 'middle', 'lower'
    """
    middle = calculate_sma(prices, periodo)
    std = prices.rolling(window=periodo).std()
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower
    }


def calculate_atr(high, low, close, periodo=14):
    """
    Calcula el Average True Range (ATR)
    
    Parámetros:
    - high: pd.Series con precios máximos
    - low: pd.Series con precios mínimos
    - close: pd.Series con precios de cierre
    - periodo: int (default 14)
    
    Retorna:
    - pd.Series con valores de ATR
    """
    # Calcular True Range
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calcular ATR
    atr = true_range.rolling(window=periodo).mean()
    
    return atr


def calculate_stochastic(high, low, close, k_periodo=14, d_periodo=3):
    """
    Calcula el Oscilador Estocástico
    
    Parámetros:
    - high: pd.Series con precios máximos
    - low: pd.Series con precios mínimos
    - close: pd.Series con precios de cierre
    - k_periodo: int (default 14)
    - d_periodo: int (default 3)
    
    Retorna:
    - dict con '%K' y '%D'
    """
    # Calcular %K
    lowest_low = low.rolling(window=k_periodo).min()
    highest_high = high.rolling(window=k_periodo).max()
    
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    
    # Calcular %D (SMA de %K)
    d = k.rolling(window=d_periodo).mean()
    
    return {
        'k': k,
        'd': d
    }


def calculate_obv(close, volume):
    """
    Calcula el On-Balance Volume (OBV)
    
    Parámetros:
    - close: pd.Series con precios de cierre
    - volume: pd.Series con volumen
    
    Retorna:
    - pd.Series con valores de OBV
    """
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


def calculate_vwap(high, low, close, volume):
    """
    Calcula el Volume Weighted Average Price (VWAP)
    
    Parámetros:
    - high: pd.Series con precios máximos
    - low: pd.Series con precios mínimos
    - close: pd.Series con precios de cierre
    - volume: pd.Series con volumen
    
    Retorna:
    - pd.Series con valores de VWAP
    """
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap


# Alias para mantener compatibilidad con código anterior
class rsi:
    """Clase wrapper para mantener compatibilidad con import anterior"""
    @staticmethod
    def get(prices, periodo=14):
        """Wrapper para calculate_rsi"""
        return calculate_rsi(prices, periodo)