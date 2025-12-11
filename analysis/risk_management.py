# analysis/risk_management.py
"""
Sistema avanzado de gestión de riesgo con trailing stops inteligentes
Versión mejorada con soporte para LONG y SHORT
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
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
                     tp_multiplier: float = 2.0,
                     support_short: bool = True,
                     log_trades: bool = True
                     ) -> pd.DataFrame:
    """
    Sistema de salida inteligente (Smart Exit) para trades LONG y SHORT.
    
    Características principales:
    - Trailing stop basado en ATR y nuevos extremos
    - Filtro de ruido para evitar updates innecesarios
    - Break-even automático
    - Swing locking (low para LONG, high para SHORT)
    - Filtro de tendencia EMA
    - Soporte para posiciones LONG y SHORT
    
    Args:
        df: DataFrame con datos OHLCV e indicadores
        entry_col: columna con señales (1=LONG, -1=SHORT, 0=neutral)
        price_col: columna de precio (default 'close')
        atr_col: columna con ATR
        ema_short_col: EMA rápida para filtro de tendencia
        ema_long_col: EMA lenta para filtro de tendencia
        atr_mult: multiplicador ATR para trailing stop
        min_move_to_update: umbral mínimo de movimiento (en ATRs)
        swing_lookback: ventana para detectar swings
        break_even_pct: % del camino a TP para activar break-even
        break_even_buffer_atr_mult: buffer en break-even
        noise_filter: si True, solo actualiza con tendencia favorable
        tp_multiplier: multiplicador para TP
        support_short: si True, permite trades SHORT
        log_trades: si True, registra trades
    
    Returns:
        DataFrame con columnas adicionales de gestión de riesgo
    """
    df = df.copy().reset_index(drop=False)
    n = len(df)
    
    # Inicializar columnas
    df['smart_trailing_stop'] = np.nan
    df['smart_extreme_price'] = np.nan  # Máximo para LONG, mínimo para SHORT
    df['smart_exit_signal'] = 0
    df['smart_break_even'] = False
    df['smart_tp_est'] = np.nan
    df['smart_trade_pnl'] = np.nan
    df['smart_trade_id'] = np.nan
    df['smart_position_type'] = ''  # 'LONG' o 'SHORT'
    
    # Variables de estado
    in_trade = False
    position_type = None  # 'LONG' o 'SHORT'
    entry_price = np.nan
    extreme_price = np.nan
    trailing_stop = np.nan
    trade_counter = 0
    stats = {'total': 0, 'long': 0, 'short': 0, 'wins': 0, 'losses': 0}
    
    for i in range(n):
        price = df.at[i, price_col]
        atr = df.at[i, atr_col] if atr_col in df.columns else np.nan
        
        # Calcular ATR fallback
        if np.isnan(atr) or atr <= 0:
            recent = df[price_col].iloc[max(0, i-14):i+1]
            atr = recent.pct_change().std() * recent.mean() if len(recent) > 1 else price * 0.01
        
        # ========== DETECCIÓN DE ENTRADA ==========
        if not in_trade and entry_col in df.columns:
            signal = df.at[i, entry_col]
            
            # Entrada LONG
            if signal == 1:
                position_type = 'LONG'
                in_trade = True
                entry_price = price
                extreme_price = price
                trailing_stop = entry_price - atr * atr_mult
                trade_counter += 1
                stats['total'] += 1
                stats['long'] += 1
                
                tp_est = entry_price + atr * atr_mult * tp_multiplier
                
                df.at[i, 'smart_extreme_price'] = extreme_price
                df.at[i, 'smart_trailing_stop'] = trailing_stop
                df.at[i, 'smart_tp_est'] = tp_est
                df.at[i, 'smart_trade_id'] = trade_counter
                df.at[i, 'smart_position_type'] = position_type
                df.at[i, 'smart_trade_pnl'] = 0
                
                if log_trades:
                    logger.info(f"📈 LONG #{trade_counter} @ ${entry_price:.2f} | SL: ${trailing_stop:.2f} | TP: ${tp_est:.2f}")
                continue
            
            # Entrada SHORT
            elif signal == -1 and support_short:
                position_type = 'SHORT'
                in_trade = True
                entry_price = price
                extreme_price = price
                trailing_stop = entry_price + atr * atr_mult
                trade_counter += 1
                stats['total'] += 1
                stats['short'] += 1
                
                tp_est = entry_price - atr * atr_mult * tp_multiplier
                
                df.at[i, 'smart_extreme_price'] = extreme_price
                df.at[i, 'smart_trailing_stop'] = trailing_stop
                df.at[i, 'smart_tp_est'] = tp_est
                df.at[i, 'smart_trade_id'] = trade_counter
                df.at[i, 'smart_position_type'] = position_type
                df.at[i, 'smart_trade_pnl'] = 0
                
                if log_trades:
                    logger.info(f"📉 SHORT #{trade_counter} @ ${entry_price:.2f} | SL: ${trailing_stop:.2f} | TP: ${tp_est:.2f}")
                continue
        
        # ========== GESTIÓN DE TRADE ACTIVO ==========
        if in_trade:
            # Actualizar extremo según tipo de posición
            should_update = False
            
            if position_type == 'LONG':
                # Para LONG: buscar nuevos máximos
                if price > extreme_price:
                    required_move = atr * min_move_to_update
                    if (price - extreme_price) >= required_move:
                        if _check_trend_filter(df, i, noise_filter, ema_short_col, ema_long_col, 'LONG'):
                            should_update = True
                            extreme_price = price
            
            elif position_type == 'SHORT':
                # Para SHORT: buscar nuevos mínimos
                if price < extreme_price:
                    required_move = atr * min_move_to_update
                    if (extreme_price - price) >= required_move:
                        if _check_trend_filter(df, i, noise_filter, ema_short_col, ema_long_col, 'SHORT'):
                            should_update = True
                            extreme_price = price
            
            # Calcular nuevo trailing stop
            if position_type == 'LONG':
                candidate_stop = extreme_price - atr * atr_mult
                trailing_stop = max(trailing_stop, candidate_stop)
                
                # Swing low anchoring
                if swing_lookback >= 2:
                    swing_low = df[price_col].iloc[max(0, i-swing_lookback):i+1].min()
                    trailing_stop = max(trailing_stop, swing_low)
                
            elif position_type == 'SHORT':
                candidate_stop = extreme_price + atr * atr_mult
                trailing_stop = min(trailing_stop, candidate_stop)
                
                # Swing high anchoring
                if swing_lookback >= 2:
                    swing_high = df[price_col].iloc[max(0, i-swing_lookback):i+1].max()
                    trailing_stop = min(trailing_stop, swing_high)
            
            # Break-even logic
            tp_est = df.at[i-1, 'smart_tp_est'] if i > 0 else np.nan
            
            if position_type == 'LONG':
                progress = (price - entry_price) / (tp_est - entry_price) if tp_est > entry_price else 0
            else:  # SHORT
                progress = (entry_price - price) / (entry_price - tp_est) if entry_price > tp_est else 0
            
            if progress >= break_even_pct and not df.at[i-1 if i > 0 else i, 'smart_break_even']:
                if position_type == 'LONG':
                    be_stop = entry_price + atr * break_even_buffer_atr_mult
                    trailing_stop = max(trailing_stop, be_stop)
                else:  # SHORT
                    be_stop = entry_price - atr * break_even_buffer_atr_mult
                    trailing_stop = min(trailing_stop, be_stop)
                
                df.at[i, 'smart_break_even'] = True
                if log_trades:
                    logger.info(f"  ⚖️ BREAK-EVEN activado @ ${price:.2f}")
            else:
                df.at[i, 'smart_break_even'] = df.at[i-1, 'smart_break_even'] if i > 0 else False
            
            # Actualizar columnas
            df.at[i, 'smart_extreme_price'] = extreme_price
            df.at[i, 'smart_trailing_stop'] = trailing_stop
            df.at[i, 'smart_tp_est'] = tp_est
            df.at[i, 'smart_trade_id'] = trade_counter
            df.at[i, 'smart_position_type'] = position_type
            
            if position_type == 'LONG':
                df.at[i, 'smart_trade_pnl'] = price - entry_price
            else:  # SHORT
                df.at[i, 'smart_trade_pnl'] = entry_price - price
            
            # Verificar salidas
            exit_type = None
            
            # Salida por STOP
            if position_type == 'LONG' and price <= trailing_stop:
                exit_type = 'STOP'
            elif position_type == 'SHORT' and price >= trailing_stop:
                exit_type = 'STOP'
            
            # Salida por TP
            elif position_type == 'LONG' and price >= tp_est:
                exit_type = 'TP'
            elif position_type == 'SHORT' and price <= tp_est:
                exit_type = 'TP'
            
            # Salida por señal contraria
            elif entry_col in df.columns:
                signal = df.at[i, entry_col]
                if (position_type == 'LONG' and signal == -1) or (position_type == 'SHORT' and signal == 1):
                    exit_type = 'SIGNAL'
            
            # Procesar salida
            if exit_type:
                pnl = df.at[i, 'smart_trade_pnl']
                pnl_pct = (pnl / entry_price) * 100
                
                if exit_type == 'STOP':
                    df.at[i, 'smart_exit_signal'] = -1
                elif exit_type == 'TP':
                    df.at[i, 'smart_exit_signal'] = 1
                else:  # SIGNAL
                    df.at[i, 'smart_exit_signal'] = -1
                
                if pnl > 0:
                    stats['wins'] += 1
                    emoji = "✅"
                else:
                    stats['losses'] += 1
                    emoji = "❌"
                
                if log_trades:
                    logger.info(f"{emoji} EXIT por {exit_type} #{trade_counter} @ ${price:.2f} | "
                              f"P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
                
                # Reset estado
                in_trade = False
                position_type = None
                entry_price = np.nan
                extreme_price = np.nan
                trailing_stop = np.nan
                continue
    
    # Restaurar índice
    df_out = df.set_index(df.columns[0])
    df_out['smart_exit_signal'] = df_out['smart_exit_signal'].astype(int)
    
    # Logging final
    if log_trades and stats['total'] > 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"RESUMEN SMART EXIT")
        logger.info(f"{'='*60}")
        logger.info(f"Total trades: {stats['total']} (LONG: {stats['long']}, SHORT: {stats['short']})")
        logger.info(f"Ganadores: {stats['wins']} ({stats['wins']/stats['total']*100:.1f}%)")
        logger.info(f"Perdedores: {stats['losses']} ({stats['losses']/stats['total']*100:.1f}%)")
        logger.info(f"{'='*60}\n")
    
    return df_out


def _check_trend_filter(df: pd.DataFrame, idx: int, enabled: bool, 
                       ema_short: str, ema_long: str, position_type: str) -> bool:
    """Verifica el filtro de tendencia EMA"""
    if not enabled or ema_short not in df.columns or ema_long not in df.columns:
        return True
    
    ema_gap = df.at[idx, ema_short] - df.at[idx, ema_long]
    
    if position_type == 'LONG':
        return ema_gap >= 0  # EMA corta > EMA larga
    else:  # SHORT
        return ema_gap <= 0  # EMA corta < EMA larga


def calcular_metricas_trailing(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula métricas de performance del sistema Smart Exit
    
    Args:
        df: DataFrame con columnas smart_* generadas por apply_smart_exit
        
    Returns:
        Dict con métricas detalladas de performance
    """
    if 'smart_exit_signal' not in df.columns:
        return {}
    
    exits = df[df['smart_exit_signal'] != 0].copy()
    
    if len(exits) == 0:
        return {'total_trades': 0}
    
    # Métricas básicas
    trades_totales = len(exits)
    exits_por_stop = (exits['smart_exit_signal'] == -1).sum()
    exits_por_tp = (exits['smart_exit_signal'] == 1).sum()
    
    # P&L
    pnl_total = exits['smart_trade_pnl'].sum()
    pnl_medio = exits['smart_trade_pnl'].mean()
    pnl_std = exits['smart_trade_pnl'].std()
    
    # Wins/Losses
    wins = exits[exits['smart_trade_pnl'] > 0]
    losses = exits[exits['smart_trade_pnl'] < 0]
    
    winning_trades = len(wins)
    losing_trades = len(losses)
    
    win_rate = (winning_trades / trades_totales * 100) if trades_totales > 0 else 0
    
    avg_win = wins['smart_trade_pnl'].mean() if winning_trades > 0 else 0
    avg_loss = losses['smart_trade_pnl'].mean() if losing_trades > 0 else 0
    
    # Profit Factor
    total_wins = wins['smart_trade_pnl'].sum() if winning_trades > 0 else 0
    total_losses = abs(losses['smart_trade_pnl'].sum()) if losing_trades > 0 else 0
    profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')
    
    # Drawdown
    cumulative_pnl = exits['smart_trade_pnl'].cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = running_max - cumulative_pnl
    max_drawdown = drawdown.max() if len(drawdown) > 0 else 0
    
    # Racha más larga
    exits['is_win'] = exits['smart_trade_pnl'] > 0
    max_win_streak = _calculate_max_streak(exits['is_win'], True)
    max_loss_streak = _calculate_max_streak(exits['is_win'], False)
    
    # Posiciones por tipo
    long_trades = (exits['smart_position_type'] == 'LONG').sum()
    short_trades = (exits['smart_position_type'] == 'SHORT').sum()
    
    return {
        # Trades
        'total_trades': trades_totales,
        'long_trades': long_trades,
        'short_trades': short_trades,
        'exits_por_stop': exits_por_stop,
        'exits_por_tp': exits_por_tp,
        
        # P&L
        'pnl_total': pnl_total,
        'pnl_medio': pnl_medio,
        'pnl_std': pnl_std,
        
        # Performance
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        
        # Risk
        'max_drawdown': max_drawdown,
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak,
        
        # Expectancy
        'expectancy': (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
    }


def _calculate_max_streak(series: pd.Series, target_value: bool) -> int:
    """Calcula la racha más larga de un valor en una serie"""
    max_streak = 0
    current_streak = 0
    
    for val in series:
        if val == target_value:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    return max_streak


def generar_reporte_trades(df: pd.DataFrame, filename: Optional[str] = None) -> pd.DataFrame:
    """
    Genera un reporte detallado de todos los trades ejecutados
    
    Args:
        df: DataFrame con información de Smart Exit
        filename: Si se provee, guarda el reporte en CSV
        
    Returns:
        DataFrame con reporte de trades
    """
    if 'smart_exit_signal' not in df.columns:
        return pd.DataFrame()
    
    exits = df[df['smart_exit_signal'] != 0].copy()
    
    if len(exits) == 0:
        return pd.DataFrame()
    
    # Construir reporte
    reporte = pd.DataFrame({
        'trade_id': exits['smart_trade_id'],
        'fecha_salida': exits.index,
        'tipo_posicion': exits['smart_position_type'],
        'precio_salida': exits['close'],
        'pnl': exits['smart_trade_pnl'],
        'pnl_pct': (exits['smart_trade_pnl'] / exits['close']) * 100,
        'tipo_salida': exits['smart_exit_signal'].map({-1: 'STOP', 1: 'TP'}),
        'break_even_activado': exits['smart_break_even']
    })
    
    if filename:
        reporte.to_csv(filename, index=False)
        logger.info(f"Reporte guardado en: {filename}")
    
    return reporte