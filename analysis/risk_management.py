# analysis/risk_management.py
"""
Sistema avanzado de gestión de riesgo con CAPITAL MANAGEMENT
Incluye: tamaño de posición, comisiones, apalancamiento, y P&L real
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
                     # Parámetros de trading
                     atr_mult: float = 1.5,
                     min_move_to_update: float = 0.5,
                     swing_lookback: int = 5,
                     break_even_pct: float = 0.9,
                     break_even_buffer_atr_mult: float = 0.2,
                     noise_filter: bool = True,
                     tp_multiplier: float = 2.0,
                     support_short: bool = True,
                     # 🆕 PARÁMETROS DE CAPITAL
                     initial_capital: float = 10000.0,  # Capital inicial en USD
                     risk_per_trade_pct: float = 2.0,   # % de capital a arriesgar por trade
                     commission: float = None,          # Comisión ya calculada por binance_client
                     leverage: float = 1.0,             # Apalancamiento (1 = sin apalancamiento)
                     # Logging
                     log_trades: bool = True
                     ) -> pd.DataFrame:
    """
    Sistema de salida inteligente con GESTIÓN DE CAPITAL REALISTA
    
    🆕 NUEVO: Calcula P&L real considerando:
    - Capital inicial y actual
    - Tamaño de posición basado en riesgo
    - Comisiones de entrada y salida (usa la comisión real de Binance)
    - Apalancamiento
    - Profit/Loss en USD y %
    
    Args:
        df: DataFrame con datos OHLCV e indicadores
        entry_col: columna con señales (1=LONG, -1=SHORT, 0=neutral)
        price_col: columna de precio (default 'close')
        atr_col: columna con ATR
        
        # Capital Management
        initial_capital: Capital inicial en USD (default: $10,000)
        risk_per_trade_pct: % de capital a arriesgar por trade (default: 2%)
        commission: Comisión en decimal (ej: 0.001 = 0.1%) obtenida de binance_client
                   Si es None, usa 0.001 (0.1%) como default
        leverage: Apalancamiento (default: 1x = sin apalancamiento)
        
    Returns:
        DataFrame con métricas de capital real
    """
    
    # Si no se pasa comisión, usar default de Binance Spot (0.1%)
    if commission is None:
        commission = 0.001
        logger.warning(f"⚠️ Comisión no especificada, usando default: {commission*100}%")
    
    # Convertir a porcentaje si está en decimal
    commission_pct = commission * 100 if commission < 1 else commission
    df = df.copy().reset_index(drop=False)
    n = len(df)
    
    # Inicializar columnas
    df['smart_trailing_stop'] = np.nan
    df['smart_extreme_price'] = np.nan
    df['smart_exit_signal'] = 0
    df['smart_break_even'] = False
    df['smart_tp_est'] = np.nan
    df['smart_trade_pnl_usd'] = np.nan  # 🆕 P&L en USD
    df['smart_trade_pnl_pct'] = np.nan  # 🆕 P&L en %
    df['smart_position_size'] = np.nan  # 🆕 Tamaño de posición (unidades)
    df['smart_position_value'] = np.nan  # 🆕 Valor de posición (USD)
    df['smart_commission_paid'] = np.nan  # 🆕 Comisión pagada
    df['smart_capital'] = np.nan  # 🆕 Capital actual
    df['smart_trade_id'] = np.nan
    df['smart_position_type'] = ''
    df['smart_position_status'] = ''
    
    # ========== VARIABLES DE ESTADO ==========
    in_trade = False
    position_type = None
    entry_price = np.nan
    entry_index = -1
    extreme_price = np.nan
    trailing_stop = np.nan
    tp_est = np.nan
    break_even_activated = False
    trade_counter = 0
    
    # 🆕 VARIABLES DE CAPITAL
    current_capital = initial_capital
    position_size = 0.0  # Cantidad de unidades (ej: BTC)
    position_value = 0.0  # Valor en USD
    entry_commission = 0.0
    
    # Estadísticas
    stats = {
        'total': 0,
        'long': 0,
        'short': 0,
        'wins': 0,
        'losses': 0,
        'total_pnl_usd': 0.0,
        'total_commission_paid': 0.0,
        'signals_ignored': 0,
        'max_capital': initial_capital,
        'min_capital': initial_capital
    }
    
    # ========== LOOP PRINCIPAL ==========
    for i in range(n):
        price = df.at[i, price_col]
        atr = df.at[i, atr_col] if atr_col in df.columns else np.nan
        
        # Calcular ATR fallback
        if np.isnan(atr) or atr <= 0:
            recent = df[price_col].iloc[max(0, i-14):i+1]
            atr = recent.pct_change().std() * recent.mean() if len(recent) > 1 else price * 0.01
        
        # Actualizar capital en el DataFrame
        df.at[i, 'smart_capital'] = current_capital
        
        # ========== DETECCIÓN DE ENTRADA ==========
        if not in_trade and entry_col in df.columns:
            signal = df.at[i, entry_col]
            
            if signal == 1 or (signal == -1 and support_short):
                # ========== CALCULAR TAMAÑO DE POSICIÓN ==========
                
                # 1. Calcular riesgo en USD
                risk_amount = current_capital * (risk_per_trade_pct / 100)
                
                # 2. Calcular distancia al Stop Loss
                if signal == 1:  # LONG
                    position_type = 'LONG'
                    stop_distance = atr * atr_mult
                    stop_price = price - stop_distance
                    tp_est = price + atr * atr_mult * tp_multiplier
                else:  # SHORT
                    position_type = 'SHORT'
                    stop_distance = atr * atr_mult
                    stop_price = price + stop_distance
                    tp_est = price - atr * atr_mult * tp_multiplier
                
                # 3. Calcular tamaño de posición
                # position_size = (riesgo / distancia_stop) * leverage
                position_size = (risk_amount / stop_distance) * leverage
                
                # 4. Calcular valor de la posición
                position_value = position_size * price
                
                # 5. Calcular comisión de entrada
                entry_commission = position_value * (commission_pct / 100)
                
                # 6. Verificar que tengamos suficiente capital
                # Necesitamos: margen (position_value/leverage) + comisión de entrada
                margin_required = position_value / leverage
                total_required = margin_required + entry_commission

                logger.warning(
                    f" current_capital: ${current_capital:.2f}")
                logger.warning(
                    f" risk_per_trade_pct: ${risk_per_trade_pct:.2f}")
                logger.warning(
                    f" risk_amount: ${risk_amount:.2f}")
                logger.warning(
                    f" stop_distance: ${stop_distance:.2f}")
                logger.warning(
                    f" leverage: ${leverage:.2f}")
                logger.warning(
                    f" price: ${price:.2f}")
                logger.warning(
                    f" position_size: ${position_size:.2f}")
                logger.warning(
                    f" commission_pct: ${commission_pct:.2f}")
                logger.warning(
                    f" position_value: ${position_value:.2f}")
                logger.warning(
                    f" leverage: ${leverage:.2f}")
                logger.warning(
                    f" margin_required: ${margin_required:.2f}, entry_commission ${entry_commission:.2f}")

                if total_required > current_capital:
                    # No hay suficiente capital
                    if log_trades:
                        logger.warning(f"⚠️ Capital insuficiente: Requerido ${total_required:.2f}, Disponible ${current_capital:.2f}")
                    continue
                
                # 7. Abrir posición
                in_trade = True
                entry_price = price
                entry_index = i
                extreme_price = price
                trailing_stop = stop_price
                break_even_activated = False
                trade_counter += 1
                
                # 🔑 BLOQUEAR margen y pagar comisión de entrada
                current_capital -= total_required  # Margen + comisión
                
                # Actualizar estadísticas
                stats['total'] += 1
                stats['total_commission_paid'] += entry_commission
                if position_type == 'LONG':
                    stats['long'] += 1
                else:
                    stats['short'] += 1
                
                # Actualizar DataFrame
                df.at[i, 'smart_extreme_price'] = extreme_price
                df.at[i, 'smart_trailing_stop'] = trailing_stop
                df.at[i, 'smart_tp_est'] = tp_est
                df.at[i, 'smart_trade_id'] = trade_counter
                df.at[i, 'smart_position_type'] = position_type
                df.at[i, 'smart_position_status'] = 'OPEN'
                df.at[i, 'smart_position_size'] = position_size
                df.at[i, 'smart_position_value'] = position_value
                df.at[i, 'smart_commission_paid'] = entry_commission
                df.at[i, 'smart_trade_pnl_usd'] = -entry_commission
                df.at[i, 'smart_trade_pnl_pct'] = -(entry_commission / current_capital * 100)
                df.at[i, 'smart_capital'] = current_capital
                
                if log_trades:
                    logger.info(f"{'='*70}")
                    logger.info(f"{'📈' if position_type == 'LONG' else '📉'} {position_type} #{trade_counter} ABIERTO")
                    logger.info(f"  💰 Capital disponible: ${current_capital:,.2f}")
                    logger.info(f"  📊 Riesgo por trade: ${risk_amount:.2f} ({risk_per_trade_pct}% del capital)")
                    logger.info(f"  📏 Tamaño posición: {position_size:.6f} unidades")
                    logger.info(f"  💵 Valor posición: ${position_value:,.2f}")
                    logger.info(f"  ⚡ Apalancamiento: {leverage}x")
                    logger.info(f"  💸 Comisión entrada: ${entry_commission:.2f}")
                    logger.info(f"  🎯 Entrada: ${entry_price:.2f}")
                    logger.info(f"  🛡️  Stop Loss: ${trailing_stop:.2f} (-{stop_distance:.2f})")
                    logger.info(f"  🎯 Take Profit: ${tp_est:.2f}")
                
                continue
        
        # ========== GESTIÓN DE TRADE ACTIVO ==========
        elif in_trade:
            # Ignorar nuevas señales
            if entry_col in df.columns:
                signal = df.at[i, entry_col]
                if signal != 0:
                    stats['signals_ignored'] += 1
            
            # Actualizar trailing stop (lógica simplificada)
            if position_type == 'LONG':
                if price > extreme_price:
                    extreme_price = price
                    candidate_stop = extreme_price - atr * atr_mult
                    trailing_stop = max(trailing_stop, candidate_stop)
            else:  # SHORT
                if price < extreme_price:
                    extreme_price = price
                    candidate_stop = extreme_price + atr * atr_mult
                    trailing_stop = min(trailing_stop, candidate_stop)
            
            # Break-even
            if position_type == 'LONG':
                progress = (price - entry_price) / (tp_est - entry_price) if tp_est > entry_price else 0
            else:
                progress = (entry_price - price) / (entry_price - tp_est) if entry_price > tp_est else 0
            
            if progress >= break_even_pct and not break_even_activated:
                if position_type == 'LONG':
                    trailing_stop = max(trailing_stop, entry_price + atr * break_even_buffer_atr_mult)
                else:
                    trailing_stop = min(trailing_stop, entry_price - atr * break_even_buffer_atr_mult)
                break_even_activated = True
                if log_trades:
                    logger.info(f"  ⚖️  BREAK-EVEN activado @ ${price:.2f}")
            
            # Calcular P&L no realizado
            if position_type == 'LONG':
                price_diff = price - entry_price
            else:
                price_diff = entry_price - price
            
            unrealized_pnl = price_diff * position_size
            unrealized_pnl_pct = (price_diff / entry_price) * 100 * leverage
            
            # Actualizar DataFrame
            df.at[i, 'smart_extreme_price'] = extreme_price
            df.at[i, 'smart_trailing_stop'] = trailing_stop
            df.at[i, 'smart_tp_est'] = tp_est
            df.at[i, 'smart_trade_id'] = trade_counter
            df.at[i, 'smart_position_type'] = position_type
            df.at[i, 'smart_position_status'] = 'OPEN'
            df.at[i, 'smart_position_size'] = position_size
            df.at[i, 'smart_position_value'] = position_size * price
            df.at[i, 'smart_break_even'] = break_even_activated
            df.at[i, 'smart_trade_pnl_usd'] = unrealized_pnl - entry_commission
            df.at[i, 'smart_trade_pnl_pct'] = unrealized_pnl_pct
            df.at[i, 'smart_capital'] = current_capital
            
            # ========== VERIFICAR SALIDAS ==========
            exit_type = None
            
            if position_type == 'LONG':
                if price <= trailing_stop:
                    exit_type = 'STOP'
                elif price >= tp_est:
                    exit_type = 'TP'
            else:  # SHORT
                if price >= trailing_stop:
                    exit_type = 'STOP'
                elif price <= tp_est:
                    exit_type = 'TP'
            
            # ========== PROCESAR SALIDA ==========
            if exit_type:
                # Calcular P&L bruto (sin comisiones)
                if position_type == 'LONG':
                    price_diff = price - entry_price
                else:
                    price_diff = entry_price - price
                
                gross_pnl = price_diff * position_size
                
                # Calcular comisión de salida
                exit_value = position_size * price
                exit_commission = exit_value * (commission_pct / 100)
                
                # P&L neto = P&L bruto - comisión de salida
                # (La comisión de entrada YA fue descontada al abrir)
                net_pnl = gross_pnl - exit_commission
                
                # Calcular % de P&L sobre el margen usado
                margin_used = position_value / leverage
                pnl_pct = (net_pnl / margin_used) * 100
                
                # Actualizar capital: devolver margen + P&L neto
                current_capital += margin_used + net_pnl
                
                # Actualizar estadísticas
                stats['total_pnl_usd'] += net_pnl
                stats['total_commission_paid'] += exit_commission  # Solo cuenta exit commission aquí
                stats['max_capital'] = max(stats['max_capital'], current_capital)
                stats['min_capital'] = min(stats['min_capital'], current_capital)
                
                if net_pnl > 0:
                    stats['wins'] += 1
                    emoji = "✅"
                else:
                    stats['losses'] += 1
                    emoji = "❌"
                
                # Marcar salida
                df.at[i, 'smart_exit_signal'] = 1 if exit_type == 'TP' else -1
                df.at[i, 'smart_position_status'] = 'CLOSED'
                df.at[i, 'smart_commission_paid'] = exit_commission
                df.at[i, 'smart_trade_pnl_usd'] = net_pnl
                df.at[i, 'smart_trade_pnl_pct'] = pnl_pct
                df.at[i, 'smart_capital'] = current_capital
                
                # Log
                if log_trades:
                    total_commission = entry_commission + exit_commission
                    roi_trade = (net_pnl / margin_used) * 100
                    
                    logger.info(f"{emoji} {position_type} #{trade_counter} CERRADO por {exit_type}")
                    logger.info(f"  🎯 Entrada: ${entry_price:.2f}")
                    logger.info(f"  🏁 Salida: ${price:.2f}")
                    logger.info(f"  📊 P&L Bruto: ${gross_pnl:+.2f}")
                    logger.info(f"  💸 Comisiones totales: ${total_commission:.2f} (entrada: ${entry_commission:.2f} + salida: ${exit_commission:.2f})")
                    logger.info(f"  💰 P&L Neto: ${net_pnl:+.2f} ({roi_trade:+.2f}% sobre margen de ${margin_used:.2f})")
                    logger.info(f"  💼 Capital actual: ${current_capital:,.2f}")
                    logger.info(f"  📈 ROI acumulado: {((current_capital/initial_capital - 1)*100):+.2f}%")
                    logger.info(f"{'='*70}\n")
                
                # Reset
                in_trade = False
                position_type = None
                entry_price = np.nan
                position_size = 0.0
                position_value = 0.0
                entry_commission = 0.0
                continue
    
    # ========== RESTAURAR ÍNDICE ==========
    df_out = df.set_index(df.columns[0])
    
    # ========== LOGGING FINAL ==========
    if log_trades and stats['total'] > 0:
        final_roi = ((current_capital / initial_capital) - 1) * 100
        
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 RESUMEN FINAL - GESTIÓN DE CAPITAL")
        logger.info(f"{'='*70}")
        logger.info(f"💰 CAPITAL:")
        logger.info(f"  ├─ Inicial: ${initial_capital:,.2f}")
        logger.info(f"  ├─ Final: ${current_capital:,.2f}")
        logger.info(f"  ├─ Máximo: ${stats['max_capital']:,.2f}")
        logger.info(f"  ├─ Mínimo: ${stats['min_capital']:,.2f}")
        logger.info(f"  └─ ROI Total: {final_roi:+.2f}%")
        logger.info(f"")
        logger.info(f"📈 TRADES:")
        logger.info(f"  ├─ Total: {stats['total']}")
        logger.info(f"  ├─ LONG: {stats['long']} | SHORT: {stats['short']}")
        logger.info(f"  ├─ Ganadores: {stats['wins']} ({stats['wins']/stats['total']*100:.1f}%)")
        logger.info(f"  └─ Perdedores: {stats['losses']} ({stats['losses']/stats['total']*100:.1f}%)")
        logger.info(f"")
        logger.info(f"💵 FINANCIERO:")
        logger.info(f"  ├─ P&L Total: ${stats['total_pnl_usd']:+,.2f}")
        logger.info(f"  ├─ Comisiones totales: ${(stats['total_commission_paid'] + stats['total'] * (stats['total_commission_paid']/stats['total'] if stats['total'] > 0 else 0)):,.2f}")
        logger.info(f"  │   ├─ Entrada: ${(stats['total'] * entry_commission if 'entry_commission' in locals() else 0):,.2f}")
        logger.info(f"  │   └─ Salida: ${stats['total_commission_paid']:,.2f}")
        logger.info(f"  └─ P&L Neto después de comisiones: ${stats['total_pnl_usd']:+,.2f}")
        logger.info(f"")
        logger.info(f"🔒 Señales ignoradas: {stats['signals_ignored']}")
        logger.info(f"{'='*70}\n")
    
    return df_out


def calcular_metricas_trailing(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula métricas de performance con gestión de capital real
    """
    if 'smart_exit_signal' not in df.columns:
        return {}
    
    exits = df[df['smart_exit_signal'] != 0].copy()
    
    if len(exits) == 0:
        return {'total_trades': 0}
    
    # Métricas básicas
    trades_totales = len(exits)
    
    # P&L en USD (ahora es real)
    pnl_total = exits['smart_trade_pnl_usd'].sum()
    pnl_medio = exits['smart_trade_pnl_usd'].mean()
    
    # Wins/Losses
    wins = exits[exits['smart_trade_pnl_usd'] > 0]
    losses = exits[exits['smart_trade_pnl_usd'] < 0]
    
    winning_trades = len(wins)
    losing_trades = len(losses)
    win_rate = (winning_trades / trades_totales * 100) if trades_totales > 0 else 0
    
    avg_win = wins['smart_trade_pnl_usd'].mean() if winning_trades > 0 else 0
    avg_loss = losses['smart_trade_pnl_usd'].mean() if losing_trades > 0 else 0
    
    # Profit Factor
    total_wins = wins['smart_trade_pnl_usd'].sum() if winning_trades > 0 else 0
    total_losses = abs(losses['smart_trade_pnl_usd'].sum()) if losing_trades > 0 else 0
    profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')
    
    # Drawdown
    cumulative_pnl = exits['smart_trade_pnl_usd'].cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = running_max - cumulative_pnl
    max_drawdown = drawdown.max()
    
    # ROI
    capital_inicial = df['smart_capital'].iloc[0] if 'smart_capital' in df.columns else 10000
    capital_final = df['smart_capital'].iloc[-1] if 'smart_capital' in df.columns else 10000
    roi_total = ((capital_final / capital_inicial) - 1) * 100
    
    # Comisiones totales
    total_commission = exits['smart_commission_paid'].sum() if 'smart_commission_paid' in exits.columns else 0
    
    return {
        'total_trades': trades_totales,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'pnl_total': pnl_total,
        'pnl_medio': pnl_medio,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'roi_total': roi_total,
        'capital_inicial': capital_inicial,
        'capital_final': capital_final,
        'total_commission': total_commission,
        'exits_por_stop': (exits['smart_exit_signal'] == -1).sum(),
        'exits_por_tp': (exits['smart_exit_signal'] == 1).sum(),
    }