# analysis/indicators.py
"""
Módulo agnóstico mejorado para cálculo de indicadores técnicos y análisis
Incluye: más indicadores, mejor detección de patrones, y señales de trading
"""
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from typing import Dict, Optional, Tuple, List
import logging

from analysis.technical_indicators import (
    calculate_rsi,
    calculate_ema,
    calculate_sma,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr
)

logger = logging.getLogger(__name__)


def detectar_divergencias_rsi(df: pd.DataFrame, lookback: int = 14) -> Dict[str, List]:
    """
    Detecta divergencias alcistas y bajistas en el RSI
    
    Args:
        df: DataFrame con columnas 'close' y 'RSI'
        lookback: Períodos hacia atrás para buscar divergencias
        
    Returns:
        Dict con listas de índices de divergencias alcistas y bajistas
    """
    divergencias_alcistas = []
    divergencias_bajistas = []
    
    for i in range(lookback, len(df)):
        # Divergencia alcista: precio hace mínimo más bajo, RSI hace mínimo más alto
        if df['close'].iloc[i] < df['close'].iloc[i-lookback]:
            if df['RSI'].iloc[i] > df['RSI'].iloc[i-lookback]:
                divergencias_alcistas.append(i)
        
        # Divergencia bajista: precio hace máximo más alto, RSI hace máximo más bajo
        if df['close'].iloc[i] > df['close'].iloc[i-lookback]:
            if df['RSI'].iloc[i] < df['RSI'].iloc[i-lookback]:
                divergencias_bajistas.append(i)
    
    return {
        'alcistas': divergencias_alcistas,
        'bajistas': divergencias_bajistas
    }


def detectar_patrones_velas(df: pd.DataFrame) -> Dict[str, List]:
    """
    Detecta patrones de velas japonesas básicos
    
    Args:
        df: DataFrame con columnas OHLC
        
    Returns:
        Dict con patrones detectados
    """
    patrones = {
        'doji': [],
        'hammer': [],
        'shooting_star': [],
        'engulfing_alcista': [],
        'engulfing_bajista': []
    }
    
    for i in range(1, len(df)):
        body = abs(df['close'].iloc[i] - df['open'].iloc[i])
        range_total = df['high'].iloc[i] - df['low'].iloc[i]
        
        if range_total == 0:
            continue
        
        # Doji: cuerpo muy pequeño
        if body / range_total < 0.1:
            patrones['doji'].append(i)
        
        # Hammer: mecha inferior larga, cuerpo pequeño arriba
        upper_shadow = df['high'].iloc[i] - max(df['open'].iloc[i], df['close'].iloc[i])
        lower_shadow = min(df['open'].iloc[i], df['close'].iloc[i]) - df['low'].iloc[i]
        
        if lower_shadow > 2 * body and upper_shadow < body:
            patrones['hammer'].append(i)
        
        # Shooting Star: mecha superior larga, cuerpo pequeño abajo
        if upper_shadow > 2 * body and lower_shadow < body:
            patrones['shooting_star'].append(i)
        
        # Engulfing alcista
        if (df['close'].iloc[i] > df['open'].iloc[i] and  # Vela alcista
            df['close'].iloc[i-1] < df['open'].iloc[i-1] and  # Vela anterior bajista
            df['open'].iloc[i] < df['close'].iloc[i-1] and  # Abre debajo del cierre anterior
            df['close'].iloc[i] > df['open'].iloc[i-1]):  # Cierra arriba de la apertura anterior
            patrones['engulfing_alcista'].append(i)
        
        # Engulfing bajista
        if (df['close'].iloc[i] < df['open'].iloc[i] and
            df['close'].iloc[i-1] > df['open'].iloc[i-1] and
            df['open'].iloc[i] > df['close'].iloc[i-1] and
            df['close'].iloc[i] < df['open'].iloc[i-1]):
            patrones['engulfing_bajista'].append(i)
    
    return patrones


def calcular_niveles_fibonacci(df: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
    """
    Calcula niveles de retroceso de Fibonacci
    
    Args:
        df: DataFrame con columnas high y low
        lookback: Períodos para encontrar swing high/low
        
    Returns:
        Dict con niveles de Fibonacci
    """
    # Encontrar swing high y swing low
    swing_high = df['high'].iloc[-lookback:].max()
    swing_low = df['low'].iloc[-lookback:].min()
    
    diff = swing_high - swing_low
    
    niveles = {
        'nivel_0': swing_high,
        'nivel_236': swing_high - 0.236 * diff,
        'nivel_382': swing_high - 0.382 * diff,
        'nivel_500': swing_high - 0.500 * diff,
        'nivel_618': swing_high - 0.618 * diff,
        'nivel_786': swing_high - 0.786 * diff,
        'nivel_100': swing_low,
        'swing_high': swing_high,
        'swing_low': swing_low
    }
    
    return niveles


def generar_senales_trading(df: pd.DataFrame) -> pd.Series:
    """
    Genera señales de trading basadas en múltiples indicadores
    
    Args:
        df: DataFrame con indicadores calculados
        
    Returns:
        pd.Series: 1 (compra), -1 (venta), 0 (neutral)
    """
    senales = pd.Series(0, index=df.index)
    
    for i in range(50, len(df)):
        puntos_compra = 0
        puntos_venta = 0
        
        # Señal RSI
        if df['RSI'].iloc[i] < 30:
            puntos_compra += 2
        elif df['RSI'].iloc[i] > 70:
            puntos_venta += 2
        
        # Señal MACD
        if (df['MACD'].iloc[i] > df['MACD_signal'].iloc[i] and 
            df['MACD'].iloc[i-1] <= df['MACD_signal'].iloc[i-1]):
            puntos_compra += 2
        elif (df['MACD'].iloc[i] < df['MACD_signal'].iloc[i] and 
              df['MACD'].iloc[i-1] >= df['MACD_signal'].iloc[i-1]):
            puntos_venta += 2
        
        # Señal cruce de medias móviles
        if (df['SMA_20'].iloc[i] > df['SMA_50'].iloc[i] and 
            df['SMA_20'].iloc[i-1] <= df['SMA_50'].iloc[i-1]):
            puntos_compra += 1
        elif (df['SMA_20'].iloc[i] < df['SMA_50'].iloc[i] and 
              df['SMA_20'].iloc[i-1] >= df['SMA_50'].iloc[i-1]):
            puntos_venta += 1
        
        # Señal Bollinger Bands
        if df['close'].iloc[i] < df['BB_lower'].iloc[i]:
            puntos_compra += 1
        elif df['close'].iloc[i] > df['BB_upper'].iloc[i]:
            puntos_venta += 1
        
        # Determinar señal final
        if puntos_compra >= 3:
            senales.iloc[i] = 1
        elif puntos_venta >= 3:
            senales.iloc[i] = -1
    
    return senales


def calcular_stop_loss_take_profit(precio_entrada: float, direccion: str, 
                                   atr: float, risk_reward: float = 2.0) -> Tuple[float, float]:
    """
    Calcula niveles de stop loss y take profit basados en ATR
    
    Args:
        precio_entrada: Precio de entrada de la operación
        direccion: 'LONG' o 'SHORT'
        atr: Valor del ATR
        risk_reward: Ratio riesgo/beneficio (default 2:1)
        
    Returns:
        Tuple[float, float]: (stop_loss, take_profit)
    """
    if direccion.upper() == 'LONG':
        stop_loss = precio_entrada - (2 * atr)
        take_profit = precio_entrada + (2 * atr * risk_reward)
    else:  # SHORT
        stop_loss = precio_entrada + (2 * atr)
        take_profit = precio_entrada - (2 * atr * risk_reward)
    
    return stop_loss, take_profit


def calcular_indicadores(symbol: str, fecha_str: str, interval: str, 
                        df: pd.DataFrame, comision: Optional[float] = None, 
                        display: int = 0) -> Dict:
    """
    Función mejorada que realiza todos los cálculos de indicadores técnicos.
    
    Args:
        symbol: Símbolo del activo (ej: 'BTCUSDT')
        fecha_str: Fecha para el nombre del archivo
        interval: Intervalo temporal
        df: DataFrame con columnas OHLCV (open, high, low, close, volume)
        comision: Comisión en decimal (ej: 0.002 para 0.2%)
        display: Si 1 guarda CSV
    
    Returns:
        Dict: Diccionario con todos los datos calculados
    """
    # Comisión por defecto si no se especifica
    if comision is None:
        comision = 0.002  # 0.2% (0.1% * 2 para entrada y salida)
    
    # Eliminar filas con NaN
    dfX = df.dropna().copy()
    
    if len(dfX) < 50:
        raise ValueError(f"Se necesitan al menos 50 velas. Recibidas: {len(dfX)}")
    
    logger.info(f"Calculando indicadores para {symbol} ({len(dfX)} velas)")
    
    # ========== INDICADORES BÁSICOS ==========
    
    # RSI
    dfX['RSI'] = calculate_rsi(dfX['close'], periodo=14)
    dfX['RSI_30'] = 30
    dfX['RSI_70'] = 70
    
    # Medias móviles
    dfX['SMA_20'] = calculate_sma(dfX['close'], periodo=20)
    dfX['SMA_50'] = calculate_sma(dfX['close'], periodo=50)
    dfX['SMA_200'] = calculate_sma(dfX['close'], periodo=200) if len(dfX) >= 200 else np.nan
    dfX['EMA_12'] = calculate_ema(dfX['close'], periodo=12)
    dfX['EMA_26'] = calculate_ema(dfX['close'], periodo=26)
    
    # MACD
    macd_data = calculate_macd(dfX['close'], fast=12, slow=26, signal=9)
    dfX['MACD'] = macd_data['macd']
    dfX['MACD_signal'] = macd_data['signal']
    dfX['MACD_histogram'] = macd_data['histogram']
    
    # Bandas de Bollinger
    bb_data = calculate_bollinger_bands(dfX['close'], periodo=20, std_dev=2)
    dfX['BB_upper'] = bb_data['upper']
    dfX['BB_middle'] = bb_data['middle']
    dfX['BB_lower'] = bb_data['lower']
    
    # ATR (Average True Range)
    dfX['ATR'] = calculate_atr(dfX['high'], dfX['low'], dfX['close'], periodo=14)
    
    # ========== DETECCIÓN DE SOPORTES Y RESISTENCIAS ==========
    
    highs = dfX['high'].values
    lows = dfX['low'].values
    
    # Parámetros para detección de picos
    min_distance_between_peaks = max(5, len(dfX) // 25)  # Ajuste dinámico
    min_prominence_of_peaks = dfX['close'].std() * 0.5  # Basado en volatilidad
    
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
    resistance_levels = dfX['high'].iloc[high_peaks_indices].tolist()
    support_levels = dfX['low'].iloc[low_peaks_indices].tolist()
    
    # Crear arrays para marcar en el gráfico
    resistance_prices = np.full(len(dfX), np.nan)
    resistance_prices[high_peaks_indices] = dfX['high'].iloc[high_peaks_indices]
    
    support_prices = np.full(len(dfX), np.nan)
    support_prices[low_peaks_indices] = dfX['low'].iloc[low_peaks_indices]
    
    # ========== LÍNEAS DE TENDENCIA ==========
    
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
    
    # ========== ANÁLISIS AVANZADO ==========
    
    # Detectar divergencias RSI
    divergencias = detectar_divergencias_rsi(dfX, lookback=14)
    
    # Detectar patrones de velas
    patrones_velas = detectar_patrones_velas(dfX)
    
    # Calcular niveles de Fibonacci
    niveles_fib = calcular_niveles_fibonacci(dfX, lookback=50)
    
    # Generar señales de trading
    dfX['senal'] = generar_senales_trading(dfX)
    
    # Calcular stop loss y take profit para última vela
    ultimo_precio = dfX['close'].iloc[-1]
    ultimo_atr = dfX['ATR'].iloc[-1]
    
    sl_long, tp_long = calcular_stop_loss_take_profit(ultimo_precio, 'LONG', ultimo_atr)
    sl_short, tp_short = calcular_stop_loss_take_profit(ultimo_precio, 'SHORT', ultimo_atr)
    
    # ========== MÉTRICAS DE MERCADO ==========
    
    precio_actual = dfX['close'].iloc[-1]
    cambio_porcentual = ((precio_actual - dfX['close'].iloc[0]) / dfX['close'].iloc[0]) * 100
    volatilidad = dfX['close'].pct_change().std() * 100
    
    # Tendencia general (basada en pendiente de SMA_50)
    if len(dfX) >= 52:
        tendencia_slope = (dfX['SMA_50'].iloc[-1] - dfX['SMA_50'].iloc[-50]) / 50
        if tendencia_slope > 0:
            tendencia = "ALCISTA"
        elif tendencia_slope < 0:
            tendencia = "BAJISTA"
        else:
            tendencia = "LATERAL"
    else:
        tendencia = "INSUFICIENTES DATOS"
    
    # ========== LOGGING DE RESULTADOS ==========
    
    logger.info(f"Resistencias detectadas: {len(resistance_levels)}")
    logger.info(f"Soportes detectados: {len(support_levels)}")
    logger.info(f"Tendencia: {tendencia}")
    logger.info(f"Volatilidad: {volatilidad:.2f}%")
    logger.info(f"Cambio total: {cambio_porcentual:.2f}%")
    
    # Guardar CSV si display=1
    if display == 1:
        filename = f'DATA_{symbol}_{fecha_str}-{interval}.csv'
        dfX.to_csv(filename)
        logger.info(f"Datos guardados en: {filename}")
        print(f"\nResistance levels: {resistance_levels}")
        print(f"Support levels: {support_levels}")
    
    # ========== RETORNAR TODOS LOS DATOS ==========
    
    return {
        # DataFrame con indicadores
        'dfX': dfX,
        
        # Información básica
        'symbol': symbol,
        'fecha_str': fecha_str,
        'interval': interval,
        'comision': comision,
        
        # Soportes y resistencias
        'resistance_prices': resistance_prices,
        'support_prices': support_prices,
        'high_peaks_indices': high_peaks_indices,
        'low_peaks_indices': low_peaks_indices,
        'resistance_levels': resistance_levels,
        'support_levels': support_levels,
        
        # Líneas de tendencia
        'slope_resistance': slope_resistance,
        'slope_support': slope_support,
        'coeffs_resistance': coeffs_resistance,
        'coeffs_support': coeffs_support,
        
        # Análisis avanzado
        'divergencias': divergencias,
        'patrones_velas': patrones_velas,
        'niveles_fibonacci': niveles_fib,
        
        # Stop Loss y Take Profit
        'sl_long': sl_long,
        'tp_long': tp_long,
        'sl_short': sl_short,
        'tp_short': tp_short,
        
        # Métricas de mercado
        'precio_actual': precio_actual,
        'cambio_porcentual': cambio_porcentual,
        'volatilidad': volatilidad,
        'tendencia': tendencia,
        'ultimo_atr': ultimo_atr
    }


def calcular_comision_binance(symbol: str) -> float:
    """
    Calcula la comisión específica de Binance
    (Función helper para mantener compatibilidad)
    
    Args:
        symbol: Símbolo del par
        
    Returns:
        float: Comisión en decimal
    """
    binance_comision_base = 0.1  # 0.1%
    comision = binance_comision_base / 100 * 2
    if symbol.startswith("BNB"):
        comision = comision * 0.75
    return comision