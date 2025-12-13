# analysis/strategies.py
"""
Módulo de estrategias de trading probadas
Incluye: Mean Reversion, Breakout, Trend Following, y más
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TradingStrategy:
    """Clase base para estrategias de trading"""
    
    def __init__(self, name: str):
        self.name = name
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Genera señales de trading
        Returns: pd.Series con 1 (compra), -1 (venta), 0 (neutral)
        """
        raise NotImplementedError("Subclases deben implementar generate_signals()")
    
    def __str__(self):
        return f"Strategy: {self.name}"


class MeanReversionStrategy(TradingStrategy):
    """
    Estrategia de reversión a la media
    
    Concepto: Cuando el precio se aleja mucho de su media, tiende a volver
    Indicadores: Bollinger Bands + RSI
    """
    
    def __init__(self, rsi_oversold=30, rsi_overbought=70):
        super().__init__("Mean Reversion")
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        for i in range(50, len(df)):
            price = df['close'].iloc[i]
            rsi = df['RSI'].iloc[i]
            bb_lower = df['BB_lower'].iloc[i]
            bb_upper = df['BB_upper'].iloc[i]
            bb_middle = df['BB_middle'].iloc[i]
            
            # COMPRA: Precio toca banda inferior + RSI sobreventa
            if price <= bb_lower and rsi < self.rsi_oversold:
                # Confirmar que el precio está rebotando
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    signals.iloc[i] = 1
            
            # VENTA: Precio toca banda superior + RSI sobrecompra
            elif price >= bb_upper and rsi > self.rsi_overbought:
                # Confirmar que el precio está cayendo
                if df['close'].iloc[i] < df['close'].iloc[i-1]:
                    signals.iloc[i] = -1
        
        logger.info(f"[{self.name}] Señales generadas: {(signals != 0).sum()}")
        return signals


class BreakoutStrategy(TradingStrategy):
    """
    Estrategia de ruptura (Breakout)
    
    Concepto: Cuando el precio rompe resistencia/soporte con volumen, continúa
    Indicadores: Soportes/Resistencias + Volumen + ATR
    """
    
    def __init__(self, volume_threshold=1.5, lookback=20):
        super().__init__("Breakout")
        self.volume_threshold = volume_threshold
        self.lookback = lookback
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        # Calcular máximos y mínimos locales
        df['resistance'] = df['high'].rolling(self.lookback).max()
        df['support'] = df['low'].rolling(self.lookback).min()
        
        # Volumen promedio
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(20).mean()
        else:
            df['volume_ma'] = 1  # Dummy si no hay volumen
        
        for i in range(self.lookback + 10, len(df)):
            price = df['close'].iloc[i]
            prev_price = df['close'].iloc[i-1]
            resistance = df['resistance'].iloc[i-1]
            support = df['support'].iloc[i-1]
            
            # Ratio de volumen
            vol_ratio = df['volume'].iloc[i] / df['volume_ma'].iloc[i] if 'volume' in df.columns else 1
            
            # COMPRA: Ruptura de resistencia con volumen
            if (price > resistance and 
                prev_price <= resistance and
                vol_ratio > self.volume_threshold):
                
                # Confirmar que la tendencia es alcista
                if df['EMA_12'].iloc[i] > df['EMA_26'].iloc[i]:
                    signals.iloc[i] = 1
            
            # VENTA: Ruptura de soporte con volumen
            elif (price < support and 
                  prev_price >= support and
                  vol_ratio > self.volume_threshold):
                
                # Confirmar que la tendencia es bajista
                if df['EMA_12'].iloc[i] < df['EMA_26'].iloc[i]:
                    signals.iloc[i] = -1
        
        logger.info(f"[{self.name}] Señales generadas: {(signals != 0).sum()}")
        return signals


class TrendFollowingStrategy(TradingStrategy):
    """
    Estrategia de seguimiento de tendencia
    
    Concepto: La tendencia es tu amiga - seguir la dirección del mercado
    Indicadores: EMAs + MACD + ADX (simulado con ATR)
    """
    
    def __init__(self, ema_fast=12, ema_slow=26):
        super().__init__("Trend Following")
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        for i in range(50, len(df)):
            ema_fast = df['EMA_12'].iloc[i]
            ema_slow = df['EMA_26'].iloc[i]
            ema_fast_prev = df['EMA_12'].iloc[i-1]
            ema_slow_prev = df['EMA_26'].iloc[i-1]
            
            macd = df['MACD'].iloc[i]
            macd_signal = df['MACD_signal'].iloc[i]
            
            price = df['close'].iloc[i]
            sma_50 = df['SMA_50'].iloc[i]
            
            # Medir fuerza de tendencia (usando pendiente de SMA_50)
            if i >= 60:
                trend_strength = (df['SMA_50'].iloc[i] - df['SMA_50'].iloc[i-10]) / df['SMA_50'].iloc[i-10] * 100
            else:
                trend_strength = 0
            
            # COMPRA: Cruce dorado + MACD alcista + precio sobre SMA_50
            if (ema_fast > ema_slow and 
                ema_fast_prev <= ema_slow_prev and  # Cruce reciente
                macd > macd_signal and
                price > sma_50 and
                trend_strength > 0.5):  # Tendencia alcista fuerte
                
                signals.iloc[i] = 1
            
            # VENTA: Cruce de la muerte + MACD bajista + precio bajo SMA_50
            elif (ema_fast < ema_slow and 
                  ema_fast_prev >= ema_slow_prev and  # Cruce reciente
                  macd < macd_signal and
                  price < sma_50 and
                  trend_strength < -0.5):  # Tendencia bajista fuerte
                
                signals.iloc[i] = -1
        
        logger.info(f"[{self.name}] Señales generadas: {(signals != 0).sum()}")
        return signals


class RSI_BBStrategy(TradingStrategy):
    """
    Estrategia combinada RSI + Bollinger Bands
    
    Concepto: Combinar sobreventa/sobrecompra con expansión de volatilidad
    Indicadores: RSI + Bollinger Bands + Volumen
    """
    
    def __init__(self, rsi_low=35, rsi_high=65):
        super().__init__("RSI + Bollinger Bands")
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        for i in range(50, len(df)):
            price = df['close'].iloc[i]
            rsi = df['RSI'].iloc[i]
            rsi_prev = df['RSI'].iloc[i-1]
            
            bb_lower = df['BB_lower'].iloc[i]
            bb_upper = df['BB_upper'].iloc[i]
            bb_middle = df['BB_middle'].iloc[i]
            
            # Calcular ancho de bandas (volatilidad)
            bb_width = (bb_upper - bb_lower) / bb_middle * 100
            
            # COMPRA: RSI sale de sobreventa + precio rebota de banda inferior
            if (rsi > self.rsi_low and 
                rsi_prev <= self.rsi_low and  # RSI saliendo de sobreventa
                price > bb_lower and  # Precio sobre banda inferior
                price < bb_middle and  # Aún no llegó a la media
                bb_width > 3):  # Volatilidad significativa
                
                # Confirmar momentum alcista
                if df['close'].iloc[i] > df['close'].iloc[i-2]:
                    signals.iloc[i] = 1
            
            # VENTA: RSI sale de sobrecompra + precio cae de banda superior
            elif (rsi < self.rsi_high and 
                  rsi_prev >= self.rsi_high and  # RSI saliendo de sobrecompra
                  price < bb_upper and  # Precio bajo banda superior
                  price > bb_middle and  # Aún sobre la media
                  bb_width > 3):  # Volatilidad significativa
                
                # Confirmar momentum bajista
                if df['close'].iloc[i] < df['close'].iloc[i-2]:
                    signals.iloc[i] = -1
        
        logger.info(f"[{self.name}] Señales generadas: {(signals != 0).sum()}")
        return signals


class EMACrossoverStrategy(TradingStrategy):
    """
    Estrategia clásica de cruce de EMAs
    
    Concepto: Simple pero efectivo en tendencias fuertes
    Indicadores: EMA rápida y lenta con filtro de tendencia
    """
    
    def __init__(self, fast=9, slow=21, filter_period=50):
        super().__init__("EMA Crossover")
        self.fast = fast
        self.slow = slow
        self.filter_period = filter_period
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        # Calcular EMAs personalizadas si no existen
        from analysis.technical_indicators import calculate_ema
        df[f'EMA_{self.fast}'] = calculate_ema(df['close'], periodo=self.fast)
        df[f'EMA_{self.slow}'] = calculate_ema(df['close'], periodo=self.slow)
        
        for i in range(self.filter_period + 5, len(df)):
            ema_fast = df[f'EMA_{self.fast}'].iloc[i]
            ema_slow = df[f'EMA_{self.slow}'].iloc[i]
            ema_fast_prev = df[f'EMA_{self.fast}'].iloc[i-1]
            ema_slow_prev = df[f'EMA_{self.slow}'].iloc[i-1]
            
            # Filtro de tendencia a largo plazo
            sma_filter = df['SMA_50'].iloc[i]
            price = df['close'].iloc[i]
            
            # COMPRA: Cruce alcista + precio sobre filtro de tendencia
            if (ema_fast > ema_slow and 
                ema_fast_prev <= ema_slow_prev and
                price > sma_filter):
                
                # Verificar que no sea un falso cruce (esperar confirmación)
                if i >= 2 and df[f'EMA_{self.fast}'].iloc[i-1] > df[f'EMA_{self.slow}'].iloc[i-1]:
                    signals.iloc[i] = 1
            
            # VENTA: Cruce bajista + precio bajo filtro de tendencia
            elif (ema_fast < ema_slow and 
                  ema_fast_prev >= ema_slow_prev and
                  price < sma_filter):
                
                # Verificar confirmación
                if i >= 2 and df[f'EMA_{self.fast}'].iloc[i-1] < df[f'EMA_{self.slow}'].iloc[i-1]:
                    signals.iloc[i] = -1
        
        logger.info(f"[{self.name}] Señales generadas: {(signals != 0).sum()}")
        return signals


# ========================================
# FUNCIONES DE UTILIDAD
# ========================================

def get_strategy(strategy_name: str, **kwargs):
    """
    Factory para crear estrategias
    
    Args:
        strategy_name: Nombre de la estrategia
        **kwargs: Parámetros específicos de la estrategia
    
    Returns:
        Instancia de TradingStrategy
    """
    strategies = {
        'mean_reversion': MeanReversionStrategy,
        'breakout': BreakoutStrategy,
        'trend_following': TrendFollowingStrategy,
        'rsi_bb': RSI_BBStrategy,
        'ema_crossover': EMACrossoverStrategy
    }
    
    strategy_class = strategies.get(strategy_name.lower())
    
    if strategy_class is None:
        logger.warning(f"Estrategia '{strategy_name}' no encontrada. Usando Mean Reversion")
        strategy_class = MeanReversionStrategy
    
    return strategy_class(**kwargs)


def list_strategies() -> Dict[str, str]:
    """Retorna diccionario con estrategias disponibles y sus descripciones"""
    return {
        'mean_reversion': 'Reversión a la media con BB + RSI',
        'breakout': 'Ruptura de niveles con volumen',
        'trend_following': 'Seguimiento de tendencia con EMAs',
        'rsi_bb': 'Combinación RSI + Bollinger Bands',
        'ema_crossover': 'Cruce de medias móviles exponenciales'
    }


def compare_strategies(df: pd.DataFrame, strategies: list = None) -> pd.DataFrame:
    """
    Compara el rendimiento de múltiples estrategias
    
    Args:
        df: DataFrame con indicadores calculados
        strategies: Lista de nombres de estrategias (None = todas)
    
    Returns:
        DataFrame con comparación de señales
    """
    if strategies is None:
        strategies = list(list_strategies().keys())
    
    results = pd.DataFrame(index=df.index)
    results['close'] = df['close']
    
    for strategy_name in strategies:
        strategy = get_strategy(strategy_name)
        signals = strategy.generate_signals(df.copy())
        results[strategy_name] = signals
    
    return results


def apply_strategy(df: pd.DataFrame, strategy_name: str = 'mean_reversion', **kwargs) -> pd.Series:
    """
    Aplica una estrategia específica al DataFrame
    
    Args:
        df: DataFrame con indicadores
        strategy_name: Nombre de la estrategia
        **kwargs: Parámetros de la estrategia
    
    Returns:
        pd.Series con señales
    """
    strategy = get_strategy(strategy_name, **kwargs)
    logger.info(f"Aplicando estrategia: {strategy.name}")
    return strategy.generate_signals(df)