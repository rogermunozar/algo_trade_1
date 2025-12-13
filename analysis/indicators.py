# analysis/indicators.py
"""
Módulo mejorado con sistema de señales avanzado y filtros estrictos
Incluye: confluencias, filtros de tendencia, zonas de valor
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


def generar_senales_trading_mejoradas(df: pd.DataFrame, 
                                      strict_mode: bool = True,
                                      min_score: int = 5) -> pd.Series:
    """
    Sistema de señales MEJORADO con filtros estrictos y confluencias
    
    Mejoras:
    - Filtro de tendencia obligatorio
    - Confluencias múltiples requeridas
    - Filtro de volumen
    - Respeto a soportes/resistencias
    - No operar en lateralización
    
    Args:
        df: DataFrame con indicadores calculados
        strict_mode: Si True, requiere más confirmaciones
        min_score: Puntuación mínima para generar señal (default: 5)
        
    Returns:
        pd.Series: 1 (compra), -1 (venta), 0 (neutral)
    """
    senales = pd.Series(0, index=df.index)
    
    # Calcular volumen promedio
    if 'volume' in df.columns:
        df['volume_ma'] = df['volume'].rolling(20).mean()
    
    for i in range(50, len(df)):
        puntos_compra = 0
        puntos_venta = 0
        
        # ========== FILTRO 1: TENDENCIA (OBLIGATORIO) ==========
        # Solo operar a favor de la tendencia
        tendencia_alcista = (df['EMA_12'].iloc[i] > df['EMA_26'].iloc[i] and 
                            df['SMA_50'].iloc[i] > df['SMA_50'].iloc[i-10])
        tendencia_bajista = (df['EMA_12'].iloc[i] < df['EMA_26'].iloc[i] and 
                            df['SMA_50'].iloc[i] < df['SMA_50'].iloc[i-10])
        
        # Si no hay tendencia clara, NO operar
        if not tendencia_alcista and not tendencia_bajista:
            continue
        
        # ========== FILTRO 2: RSI CON ZONAS ESTRICTAS ==========
        rsi = df['RSI'].iloc[i]
        
        if tendencia_alcista:
            # En tendencia alcista: buscar RSI en zona de sobreventa
            if rsi < 35:  # Zona fuerte de sobreventa
                puntos_compra += 3
            elif rsi < 45:  # Zona moderada
                puntos_compra += 1
            
            # RSI en sobrecompra = no comprar
            if rsi > 65:
                puntos_compra -= 3
        
        if tendencia_bajista:
            # En tendencia bajista: buscar RSI en zona de sobrecompra
            if rsi > 65:
                puntos_venta += 3
            elif rsi > 55:
                puntos_venta += 1
            
            # RSI en sobreventa = no vender
            if rsi < 35:
                puntos_venta -= 3
        
        # ========== FILTRO 3: MACD CON CONFIRMACIÓN ==========
        macd = df['MACD'].iloc[i]
        macd_signal = df['MACD_signal'].iloc[i]
        macd_hist = df['MACD_histogram'].iloc[i]
        
        # Cruce alcista del MACD
        if (macd > macd_signal and 
            df['MACD'].iloc[i-1] <= df['MACD_signal'].iloc[i-1] and
            macd_hist > 0):
            if tendencia_alcista:
                puntos_compra += 2
        
        # Cruce bajista del MACD
        elif (macd < macd_signal and 
              df['MACD'].iloc[i-1] >= df['MACD_signal'].iloc[i-1] and
              macd_hist < 0):
            if tendencia_bajista:
                puntos_venta += 2
        
        # Momentum del MACD
        if macd_hist > df['MACD_histogram'].iloc[i-1] > 0:
            if tendencia_alcista:
                puntos_compra += 1
        elif macd_hist < df['MACD_histogram'].iloc[i-1] < 0:
            if tendencia_bajista:
                puntos_venta += 1
        
        # ========== FILTRO 4: BOLLINGER BANDS ==========
        precio = df['close'].iloc[i]
        bb_lower = df['BB_lower'].iloc[i]
        bb_upper = df['BB_upper'].iloc[i]
        bb_middle = df['BB_middle'].iloc[i]
        
        # Precio toca banda inferior (señal de compra)
        if precio <= bb_lower and tendencia_alcista:
            puntos_compra += 2
        
        # Precio toca banda superior (señal de venta)
        if precio >= bb_upper and tendencia_bajista:
            puntos_venta += 2
        
        # Precio cruza media de BB
        if precio > bb_middle and df['close'].iloc[i-1] <= df['BB_middle'].iloc[i-1]:
            if tendencia_alcista:
                puntos_compra += 1
        elif precio < bb_middle and df['close'].iloc[i-1] >= df['BB_middle'].iloc[i-1]:
            if tendencia_bajista:
                puntos_venta += 1
        
        # ========== FILTRO 5: VOLUMEN (CONFIRMACIÓN) ==========
        if 'volume' in df.columns and 'volume_ma' in df.columns:
            vol_ratio = df['volume'].iloc[i] / df['volume_ma'].iloc[i]
            
            # Volumen alto confirma la señal
            if vol_ratio > 1.5:
                puntos_compra += 1
                puntos_venta += 1
            # Volumen bajo penaliza
            elif vol_ratio < 0.7:
                puntos_compra -= 2
                puntos_venta -= 2
        
        # ========== FILTRO 6: PRICE ACTION ==========
        # Velas verdes/rojas consecutivas
        velas_verdes = 0
        velas_rojas = 0
        for j in range(max(0, i-3), i+1):
            if df['close'].iloc[j] > df['open'].iloc[j]:
                velas_verdes += 1
            else:
                velas_rojas += 1
        
        if velas_verdes >= 3 and tendencia_alcista:
            puntos_compra += 1
        if velas_rojas >= 3 and tendencia_bajista:
            puntos_venta += 1
        
        # ========== FILTRO 7: DISTANCIA A MEDIAS MÓVILES ==========
        # No comprar si el precio está muy alejado de las MAs
        distancia_ema12 = abs((precio - df['EMA_12'].iloc[i]) / precio * 100)
        
        if distancia_ema12 > 5:  # Más del 5% alejado
            puntos_compra -= 2
            puntos_venta -= 2
        
        # ========== DECISIÓN FINAL ==========
        if strict_mode:
            # Modo estricto: requiere puntuación alta
            if puntos_compra >= min_score and tendencia_alcista:
                senales.iloc[i] = 1
            elif puntos_venta >= min_score and tendencia_bajista:
                senales.iloc[i] = -1
        else:
            # Modo normal
            if puntos_compra >= 4 and tendencia_alcista:
                senales.iloc[i] = 1
            elif puntos_venta >= 4 and tendencia_bajista:
                senales.iloc[i] = -1
    
    return senales


def generar_senales_trading(df: pd.DataFrame) -> pd.Series:
    """
    Wrapper para mantener compatibilidad con código existente
    Usa el sistema mejorado por defecto
    """
    return generar_senales_trading_mejoradas(df, strict_mode=True, min_score=5)


# ========== FUNCIONES EXISTENTES (sin cambios) ==========

def detectar_divergencias_rsi(df: pd.DataFrame, lookback: int = 14) -> Dict[str, List]:
    """Detecta divergencias alcistas y bajistas en el RSI"""
    divergencias_alcistas = []
    divergencias_bajistas = []
    
    for i in range(lookback, len(df)):
        if df['close'].iloc[i] < df['close'].iloc[i-lookback]:
            if df['RSI'].iloc[i] > df['RSI'].iloc[i-lookback]:
                divergencias_alcistas.append(i)
        
        if df['close'].iloc[i] > df['close'].iloc[i-lookback]:
            if df['RSI'].iloc[i] < df['RSI'].iloc[i-lookback]:
                divergencias_bajistas.append(i)
    
    return {
        'alcistas': divergencias_alcistas,
        'bajistas': divergencias_bajistas
    }


def detectar_patrones_velas(df: pd.DataFrame) -> Dict[str, List]:
    """Detecta patrones de velas japonesas básicos"""
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
        
        if body / range_total < 0.1:
            patrones['doji'].append(i)
        
        upper_shadow = df['high'].iloc[i] - max(df['open'].iloc[i], df['close'].iloc[i])
        lower_shadow = min(df['open'].iloc[i], df['close'].iloc[i]) - df['low'].iloc[i]
        
        if lower_shadow > 2 * body and upper_shadow < body:
            patrones['hammer'].append(i)
        
        if upper_shadow > 2 * body and lower_shadow < body:
            patrones['shooting_star'].append(i)
        
        if (df['close'].iloc[i] > df['open'].iloc[i] and
            df['close'].iloc[i-1] < df['open'].iloc[i-1] and
            df['open'].iloc[i] < df['close'].iloc[i-1] and
            df['close'].iloc[i] > df['open'].iloc[i-1]):
            patrones['engulfing_alcista'].append(i)
        
        if (df['close'].iloc[i] < df['open'].iloc[i] and
            df['close'].iloc[i-1] > df['open'].iloc[i-1] and
            df['open'].iloc[i] > df['close'].iloc[i-1] and
            df['close'].iloc[i] < df['open'].iloc[i-1]):
            patrones['engulfing_bajista'].append(i)
    
    return patrones


def calcular_niveles_fibonacci(df: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
    """Calcula niveles de retroceso de Fibonacci"""
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


def calcular_stop_loss_take_profit(precio_entrada: float, direccion: str, 
                                   atr: float, risk_reward: float = 2.0) -> Tuple[float, float]:
    """Calcula niveles de stop loss y take profit basados en ATR"""
    if direccion.upper() == 'LONG':
        stop_loss = precio_entrada - (2 * atr)
        take_profit = precio_entrada + (2 * atr * risk_reward)
    else:
        stop_loss = precio_entrada + (2 * atr)
        take_profit = precio_entrada - (2 * atr * risk_reward)
    
    return stop_loss, take_profit


def calcular_indicadores(symbol: str, fecha_str: str, interval: str, 
                        df: pd.DataFrame, comision: Optional[float] = None, 
                        display: int = 0,
                        signal_mode: str = 'strict') -> Dict:
    """
    Función principal de cálculo de indicadores
    
    Args:
        symbol: Símbolo del activo
        fecha_str: Fecha para el nombre del archivo
        interval: Intervalo temporal
        df: DataFrame con OHLCV
        comision: Comisión del broker
        display: Si 1 guarda CSV
        signal_mode: 'strict' (estricto), 'normal', o 'aggressive'
    
    Returns:
        Dict con todos los datos calculados
    """
    if comision is None:
        logger.error("Error: no se especificó la comisión del broker")
        raise ValueError("Comisión requerida")

    dfX = df.dropna().copy()
    
    if len(dfX) < 50:
        raise ValueError(f"Se necesitan al menos 50 velas. Recibidas: {len(dfX)}")
    
    logger.info(f"Calculando indicadores para {symbol} ({len(dfX)} velas)")
    
    # ========== INDICADORES BÁSICOS ==========
    dfX['RSI'] = calculate_rsi(dfX['close'], periodo=14)
    dfX['RSI_30'] = 30
    dfX['RSI_70'] = 70
    
    dfX['SMA_20'] = calculate_sma(dfX['close'], periodo=20)
    dfX['SMA_50'] = calculate_sma(dfX['close'], periodo=50)
    dfX['SMA_200'] = calculate_sma(dfX['close'], periodo=200) if len(dfX) >= 200 else np.nan
    dfX['EMA_12'] = calculate_ema(dfX['close'], periodo=12)
    dfX['EMA_26'] = calculate_ema(dfX['close'], periodo=26)
    
    macd_data = calculate_macd(dfX['close'], fast=12, slow=26, signal=9)
    dfX['MACD'] = macd_data['macd']
    dfX['MACD_signal'] = macd_data['signal']
    dfX['MACD_histogram'] = macd_data['histogram']
    
    bb_data = calculate_bollinger_bands(dfX['close'], periodo=20, std_dev=2)
    dfX['BB_upper'] = bb_data['upper']
    dfX['BB_middle'] = bb_data['middle']
    dfX['BB_lower'] = bb_data['lower']
    
    dfX['ATR'] = calculate_atr(dfX['high'], dfX['low'], dfX['close'], periodo=14)
    
    # ========== SOPORTES Y RESISTENCIAS ==========
    highs = dfX['high'].values
    lows = dfX['low'].values
    
    min_distance_between_peaks = max(5, len(dfX) // 25)
    min_prominence_of_peaks = dfX['close'].std() * 0.5
    
    high_peaks_indices, _ = find_peaks(
        highs,
        distance=min_distance_between_peaks,
        prominence=min_prominence_of_peaks
    )
    
    low_peaks_indices, _ = find_peaks(
        -lows,
        distance=min_distance_between_peaks,
        prominence=min_prominence_of_peaks
    )
    
    resistance_levels = dfX['high'].iloc[high_peaks_indices].tolist()
    support_levels = dfX['low'].iloc[low_peaks_indices].tolist()
    
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
    divergencias = detectar_divergencias_rsi(dfX, lookback=14)
    patrones_velas = detectar_patrones_velas(dfX)
    niveles_fib = calcular_niveles_fibonacci(dfX, lookback=50)
    
    # ========== GENERAR SEÑALES (MEJORADAS) ==========
    if signal_mode == 'strict':
        dfX['senal'] = generar_senales_trading_mejoradas(dfX, strict_mode=True, min_score=5)
    elif signal_mode == 'normal':
        dfX['senal'] = generar_senales_trading_mejoradas(dfX, strict_mode=True, min_score=4)
    else:  # aggressive
        dfX['senal'] = generar_senales_trading_mejoradas(dfX, strict_mode=False, min_score=3)
    
    # ========== STOP LOSS Y TAKE PROFIT ==========
    ultimo_precio = dfX['close'].iloc[-1]
    ultimo_atr = dfX['ATR'].iloc[-1]
    
    sl_long, tp_long = calcular_stop_loss_take_profit(ultimo_precio, 'LONG', ultimo_atr)
    sl_short, tp_short = calcular_stop_loss_take_profit(ultimo_precio, 'SHORT', ultimo_atr)
    
    # ========== MÉTRICAS DE MERCADO ==========
    precio_actual = dfX['close'].iloc[-1]
    cambio_porcentual = ((precio_actual - dfX['close'].iloc[0]) / dfX['close'].iloc[0]) * 100
    volatilidad = dfX['close'].pct_change().std() * 100
    
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
    
    # ========== LOGGING ==========
    logger.info(f"Resistencias detectadas: {len(resistance_levels)}")
    logger.info(f"Soportes detectados: {len(support_levels)}")
    logger.info(f"Tendencia: {tendencia}")
    logger.info(f"Señales generadas: {(dfX['senal'] != 0).sum()}")
    
    if display == 1:
        filename = f'DATA_{symbol}_{fecha_str}-{interval}.csv'
        dfX.to_csv(filename)
        logger.info(f"Datos guardados en: {filename}")
    
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
        'coeffs_support': coeffs_support,
        'divergencias': divergencias,
        'patrones_velas': patrones_velas,
        'niveles_fibonacci': niveles_fib,
        'sl_long': sl_long,
        'tp_long': tp_long,
        'sl_short': sl_short,
        'tp_short': tp_short,
        'precio_actual': precio_actual,
        'cambio_porcentual': cambio_porcentual,
        'volatilidad': volatilidad,
        'tendencia': tendencia,
        'ultimo_atr': ultimo_atr
    }