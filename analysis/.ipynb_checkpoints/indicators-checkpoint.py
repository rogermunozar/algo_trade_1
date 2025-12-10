# analysis/indicators.py
"""
Módulo agnóstico para cálculo de indicadores técnicos y análisis
Funciona con cualquier DataFrame OHLCV independientemente del broker
"""
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from analysis.technical_indicators import (
    calculate_rsi,
    calculate_ema,
    calculate_sma,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr
)


def calcular_indicadores(symbol, fecha_str, interval, df, comision=None, display=0):
    """
    Función que realiza todos los cálculos de indicadores técnicos.
    
    Parámetros:
    - symbol: str (ej: 'BTCUSDT')
    - fecha_str: str (fecha para el nombre del archivo)
    - interval: str (intervalo temporal)
    - df: DataFrame con columnas OHLCV (open, high, low, close, volume)
    - comision: float (opcional, comisión en decimal ej: 0.002 para 0.2%)
    - display: int (0 o 1, si 1 guarda CSV)
    
    Returns:
        dict: Diccionario con todos los datos calculados necesarios para graficar
    """
    # Comisión por defecto si no se especifica
    if comision is None:
        comision = 0.002  # 0.2% (0.1% * 2 para entrada y salida)
    
    # Eliminar filas con NaN
    dfX = df.dropna()
    
    # Calcular indicadores técnicos
    dfX['RSI'] = calculate_rsi(dfX['close'], periodo=14)
    dfX['RSI_30'] = 30
    dfX['RSI_70'] = 70
    
    # Calcular medias móviles
    dfX['SMA_20'] = calculate_sma(dfX['close'], periodo=20)
    dfX['SMA_50'] = calculate_sma(dfX['close'], periodo=50)
    dfX['EMA_12'] = calculate_ema(dfX['close'], periodo=12)
    dfX['EMA_26'] = calculate_ema(dfX['close'], periodo=26)
    
    # Calcular MACD
    macd_data = calculate_macd(dfX['close'], fast=12, slow=26, signal=9)
    dfX['MACD'] = macd_data['macd']
    dfX['MACD_signal'] = macd_data['signal']
    dfX['MACD_histogram'] = macd_data['histogram']
    
    # Calcular Bandas de Bollinger
    bb_data = calculate_bollinger_bands(dfX['close'], periodo=20, std_dev=2)
    dfX['BB_upper'] = bb_data['upper']
    dfX['BB_middle'] = bb_data['middle']
    dfX['BB_lower'] = bb_data['lower']
    
    # Calcular ATR
    dfX['ATR'] = calculate_atr(dfX['high'], dfX['low'], dfX['close'], periodo=14)
    
    highs = dfX['high']
    lows = dfX['low']
    
    # Parámetros para detección de picos
    min_distance_between_peaks = 20
    min_prominence_of_peaks = 2.0
    
    # Encontrar índices de picos (resistencias)
    high_peaks_indices, high_peak_properties = find_peaks(
        highs,
        distance=min_distance_between_peaks,
        prominence=min_prominence_of_peaks
    )
    
    # Encontrar índices de valles (soportes)
    low_peaks_indices, low_peak_properties = find_peaks(
        -lows,
        distance=min_distance_between_peaks,
        prominence=min_prominence_of_peaks
    )
    
    # Extraer niveles de precios
    resistance_levels = highs.iloc[high_peaks_indices].tolist()
    support_levels = lows.iloc[low_peaks_indices].tolist()
    
    # Crear arrays para marcar en el gráfico
    resistance_prices = np.full(len(dfX), np.nan)
    resistance_prices[high_peaks_indices] = dfX['high'].iloc[high_peaks_indices]
    
    support_prices = np.full(len(dfX), np.nan)
    support_prices[low_peaks_indices] = dfX['low'].iloc[low_peaks_indices]
    
    # Calcular pendientes de líneas de tendencia
    slope_resistance = None
    slope_support = None
    coeffs_resistance = None
    coeffs_support = None
    
    if len(high_peaks_indices) >= 2:
        x_resistance = high_peaks_indices
        y_resistance = dfX['high'].iloc[high_peaks_indices].values
        coeffs_resistance = np.polyfit(x_resistance, y_resistance, 1)
        slope_resistance = coeffs_resistance[0]
    
    if len(low_peaks_indices) >= 2:
        x_support = low_peaks_indices
        y_support = dfX['low'].iloc[low_peaks_indices].values
        coeffs_support = np.polyfit(x_support, y_support, 1)
        slope_support = coeffs_support[0]
    
    # Guardar CSV si display=1
    if display == 1:
        df.to_csv('DATA_' + symbol + '_' + fecha_str + '-' + interval + '.csv')
        print("Resistance levels:", resistance_levels)
        print("Support levels:", support_levels)
    
    # Retornar todos los datos calculados
    return {
        'dfX': dfX,
        'symbol': symbol,
        'fecha_str': fecha_str,
        'interval': interval,
        'comision': comision,
        'resistance_prices': resistance_prices,
        'support_prices': support_prices,
        'high_peaks_indices': high_peaks_indices,
        'low_peaks_indices': low_peaks_indices,
        'resistance_levels': resistance_levels,
        'support_levels': support_levels,
        'slope_resistance': slope_resistance,
        'slope_support': slope_support,
        'coeffs_resistance': coeffs_resistance,
        'coeffs_support': coeffs_support
    }


def calcular_comision_binance(symbol):
    """
    Calcula la comisión específica de Binance
    (Función helper para mantener compatibilidad)
    """
    binance_comision_base = 0.1  # 0.1%
    comision = binance_comision_base / 100 * 2
    if symbol == "BNBUSDT":
        comision = comision * 0.75
    return comision