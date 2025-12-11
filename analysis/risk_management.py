# analysis/risk_management.py (MEJORADO)
"""
Sistema avanzado de gestión de riesgo con trailing stops inteligentes
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)


def apply_smart_exit(df: pd.DataFrame,
                     entry_col: str = 'senal',
                     price_col: str = 'close',
                     atr_col: str = 'ATR',
                     ema_short_col: str = 'EMA_12',
                     ema_long_col: str = 'EMA_26',
                     atr_mult: float = 1.5,
                     min_move_to_update: float = 0.5,
                     swing_lookback: int = 5,
                     break_even_pct: float = 0.9,
                     break_even_buffer_atr_mult: float = 0.2,
                     noise_filter: bool = True,
                     noise_filter_ema_gap: float = 0.0,
                     allow_reentry_after_exit: bool = False,
                     tp_multiplier: float = 2.0,
                     log_trades: bool = True
                     ) -> pd.DataFrame:
    """
    Aplica un sistema de salida inteligente (Full Smart Exit) para trades LONG.
    
    Características principales:
    - Trailing stop basado en ATR y nuevos máximos
    - Filtro de ruido para evitar updates innecesarios
    - Break-even automático
    - Swing low locking
    - Filtro de tendencia EMA
    
    Args:
        df: DataFrame con datos OHLCV e indicadores
        entry_col: columna con señales (1=LONG, -1=SHORT, 0=neutral)
        price_col: columna de precio (default 'close')
        atr_col: columna con ATR
        ema_short_col: EMA rápida para filtro de tendencia
        ema_long_col: EMA lenta para filtro de tendencia
        atr_mult: multiplicador ATR para trailing stop (1.5 = stop a 1.5*ATR)
        min_move_to_update: umbral mínimo de movimiento para actualizar (en ATRs)
        swing_lookback: ventana para detectar swing lows
        break_even_pct: % del camino a TP para activar break-even (0.9 = 90%)
        break_even_buffer_atr_mult: buffer sobre entry en break-even
        noise_filter: si True, solo actualiza con tendencia alcista
        noise_filter_ema_gap: gap mínimo EMA para considerar tendencia
        allow_reentry_after_exit: permitir reentrada inmediata
        tp_multiplier: multiplicador para calcular TP estimado
        log_trades: si True, registra trades en log
    
    Returns:
        DataFrame con columnas adicionales:
        - smart_trailing_stop: nivel actual del stop
        - smart_max_price: máximo alcanzado desde entrada
        - smart_exit_signal: -1 si sale por stop, 0 otherwise
        - smart_break_even: True si break-even fue activado
        - smart_tp_est: Take Profit estimado
        - smart_trade_pnl: P&L acumulado del trade actual
    """
    df = df.copy().reset_index(drop=False)
    n = len(df)
    
    # Inicializar columnas
    df['smart_trailing_stop'] = np.nan
    df['smart_max_price'] = np.nan
    df['smart_exit_signal'] = 0
    df['smart_break_even'] = False
    df['smart_tp_est'] = np.nan
    df['smart_trade_pnl'] = np.nan
    df['smart_trade_id'] = np.nan
    
    # Variables de estado
    in_trade = False
    entry_price = np.nan
    max_price = np.nan
    trailing_stop = np.nan
    trade_counter = 0
    total_trades = 0
    winning_trades = 0
    
    for i in range(n):
        price = df.at[i, price_col]
        atr = df.at[i, atr_col] if atr_col in df.columns else np.nan
        
        # Calcular ATR fallback si es necesario
        if np.isnan(atr) or atr <= 0:
            recent = df[price_col].iloc[max(0, i-14):i+1]
            atr = recent.pct_change().std() * recent.mean() if len(recent) > 1 else price * 0.01
        
        # ========== DETECCIÓN DE ENTRADA ==========
        if (not in_trade) and (entry_col in df.columns) and (df.at[i, entry_col] == 1):
            # Iniciar trade LONG
            in_trade = True
            entry_price = price
            max_price = price
            trailing_stop = entry_price - atr * atr_mult
            trade_counter += 1
            total_trades += 1
            
            tp_est = entry_price + atr * atr_mult * tp_multiplier
            
            df.at[i, 'smart_max_price'] = max_price
            df.at[i, 'smart_trailing_stop'] = trailing_stop
            df.at[i, 'smart_tp_est'] = tp_est
            df.at[i, 'smart_trade_id'] = trade_counter
            df.at[i, 'smart_trade_pnl'] = 0
            
            if log_trades:
                logger.info(f"📈 LONG ENTRY #{trade_counter} @ {entry_price:.2f} | SL: {trailing_stop:.2f} | TP: {tp_est:.2f}")
            
            continue
        
        # ========== GESTIÓN DE TRADE ACTIVO ==========
        if in_trade:
            # 1. Actualizar máximo precio
            should_update_max = False
            
            if price > max_price:
                required_move = atr * min_move_to_update
                
                if (price - max_price) >= required_move:
                    # Aplicar filtro de tendencia si está activo
                    if noise_filter and ema_short_col in df.columns and ema_long_col in df.columns:
                        ema_gap = df.at[i, ema_short_col] - df.at[i, ema_long_col]
                        if ema_gap >= noise_filter_ema_gap:
                            should_update_max = True
                    else:
                        should_update_max = True
            
            if should_update_max:
                old_max = max_price
                max_price = price
                if log_trades:
                    logger.debug(f"  ↗️ Nuevo máximo: {old_max:.2f} -> {max_price:.2f}")
            
            # 2. Calcular nuevo trailing stop
            candidate_stop = max_price - atr * atr_mult
            old_stop = trailing_stop
            trailing_stop = max(trailing_stop, candidate_stop)
            
            # 3. Swing low anchoring
            if swing_lookback and swing_lookback >= 2:
                start = max(0, i - swing_lookback)
                swing_low = df[price_col].iloc[start:i+1].min()
                trailing_stop = max(trailing_stop, swing_low)
            
            # 4. Break-even logic
            tp_est = entry_price + atr * atr_mult * tp_multiplier
            progress_to_tp = (price - entry_price) / (tp_est - entry_price) if tp_est > entry_price else 0
            
            if progress_to_tp >= break_even_pct and not df.at[i-1 if i > 0 else i, 'smart_break_even']:
                be_stop = entry_price + atr * break_even_buffer_atr_mult
                old_stop_be = trailing_stop
                trailing_stop = max(trailing_stop, be_stop)
                df.at[i, 'smart_break_even'] = True
                
                if log_trades:
                    logger.info(f"  ⚖️ BREAK-EVEN activado @ {price:.2f} | SL: {old_stop_be:.2f} -> {trailing_stop:.2f}")
            else:
                df.at[i, 'smart_break_even'] = df.at[i-1, 'smart_break_even'] if i > 0 else False
            
            # Actualizar columnas
            df.at[i, 'smart_max_price'] = max_price
            df.at[i, 'smart_trailing_stop'] = trailing_stop
            df.at[i, 'smart_tp_est'] = tp_est
            df.at[i, 'smart_trade_id'] = trade_counter
            df.at[i, 'smart_trade_pnl'] = price - entry_price
            
            # 5. Verificar salida por STOP
            if price <= trailing_stop:
                df.at[i, 'smart_exit_signal'] = -1
                pnl = price - entry_price
                pnl_pct = (pnl / entry_price) * 100
                
                if pnl > 0:
                    winning_trades += 1
                
                if log_trades:
                    emoji = "✅" if pnl > 0 else "❌"
                    logger.info(f"{emoji} EXIT por STOP #{trade_counter} @ {price:.2f} | P&L: {pnl:+.2f} ({pnl_pct:+.2f}%)")
                
                in_trade = False
                entry_price = np.nan
                max_price = np.nan
                trailing_stop = np.nan
                continue
            
            # 6. Verificar salida por SEÑAL
            if (entry_col in df.columns) and (df.at[i, entry_col] == -1):
                df.at[i, 'smart_exit_signal'] = -1
                pnl = price - entry_price
                pnl_pct = (pnl / entry_price) * 100
                
                if pnl > 0:
                    winning_trades += 1
                
                if log_trades:
                    emoji = "✅" if pnl > 0 else "❌"
                    logger.info(f"{emoji} EXIT por SEÑAL #{trade_counter} @ {price:.2f} | P&L: {pnl:+.2f} ({pnl_pct:+.2f}%)")
                
                in_trade = False
                entry_price = np.nan
                max_price = np.nan
                trailing_stop = np.nan
                continue
            
            # 7. Verificar llegada a TP
            if price >= tp_est:
                df.at[i, 'smart_exit_signal'] = 1  # Exit por TP
                pnl = price - entry_price
                pnl_pct = (pnl / entry_price) * 100
                winning_trades += 1
                
                if log_trades:
                    logger.info(f"🎯 EXIT por TP #{trade_counter} @ {price:.2f} | P&L: {pnl:+.2f} ({pnl_pct:+.2f}%)")
                
                in_trade = False
                entry_price = np.nan
                max_price = np.nan
                trailing_stop = np.nan
                continue
    
    # Restaurar índice original
    df_out = df.set_index(df.columns[0])
    df_out['smart_exit_signal'] = df_out['smart_exit_signal'].astype(int)
    
    # Logging final
    if log_trades and total_trades > 0:
        win_rate = (winning_trades / total_trades) * 100
        logger.info(f"\n{'='*50}")
        logger.info(f"RESUMEN DE TRADES")
        logger.info(f"{'='*50}")
        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Ganadores: {winning_trades}")
        logger.info(f"Perdedores: {total_trades - winning_trades}")
        logger.info(f"Win Rate: {win_rate:.1f}%")
        logger.info(f"{'='*50}\n")
    
    return df_out


def calcular_metricas_trailing(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula métricas de performance del trailing stop
    
    Args:
        df: DataFrame con columnas smart_* generadas por apply_smart_exit
        
    Returns:
        Dict con métricas de performance
    """
    if 'smart_exit_signal' not in df.columns:
        return {}
    
    exits = df[df['smart_exit_signal'] != 0].copy()
    
    if len(exits) == 0:
        return {'total_trades': 0}
    
    # Calcular métricas
    trades_totales = len(exits)
    exits_por_stop = (exits['smart_exit_signal'] == -1).sum()
    exits_por_tp = (exits['smart_exit_signal'] == 1).sum()
    
    pnl_total = exits['smart_trade_pnl'].sum()
    pnl_medio = exits['smart_trade_pnl'].mean()
    
    winning_trades = (exits['smart_trade_pnl'] > 0).sum()
    losing_trades = (exits['smart_trade_pnl'] < 0).sum()
    
    win_rate = (winning_trades / trades_totales * 100) if trades_totales > 0 else 0
    
    avg_win = exits[exits['smart_trade_pnl'] > 0]['smart_trade_pnl'].mean() if winning_trades > 0 else 0
    avg_loss = exits[exits['smart_trade_pnl'] < 0]['smart_trade_pnl'].mean() if losing_trades > 0 else 0
    
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    
    return {
        'total_trades': trades_totales,
        'exits_por_stop': exits_por_stop,
        'exits_por_tp': exits_por_tp,
        'pnl_total': pnl_total,
        'pnl_medio': pnl_medio,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor
    }