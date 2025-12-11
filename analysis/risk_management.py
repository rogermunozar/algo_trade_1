# risk_management.py  (o pégalo en analysis/indicators.py)
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from scipy.signal import find_peaks

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
                     allow_reentry_after_exit: bool = False
                     ) -> pd.DataFrame:
    """
    Aplica un sistema de salida inteligente (Full Smart Exit) para trades LONG.
    - Trailing stop basado en ATR y en nuevos máximos de la corrida (max_price)
    - Solo actualiza el trailing si el nuevo máximo supera el anterior por un umbral
      relacionado con ATR (evita updates por ruido)
    - Break-even: si el price alcanza break_even_pct del camino a TP, mover SL a break-even + buffer
    - Swing locking: opcionalmente anclar el SL en swing lows detectados
    - Noise filter: opcional, solo actualizar trailing si EMA_short > EMA_long (tendencia)
    
    Args:
        df: DataFrame con al menos columnas price_col y atr_col y columna de señales de entrada (entry_col)
        entry_col: columna con entradas (1 = long entry)
        atr_col: columna con ATR
        atr_mult: multiplicador ATR para calcular distancia del trailing
        min_move_to_update: multiplicador (fracción de ATR) que define cuánto debe subir el max_price para actualizar stop
        swing_lookback: ventana para detectar swing lows (en velas)
        break_even_pct: ratio (0..1) del camino a TP donde activamos break-even
        break_even_buffer_atr_mult: buffer sobre entry en ATRs al mover a break-even
        noise_filter: si True solo se actualiza trailing cuando EMA_short > EMA_long (tendencia alcista)
        noise_filter_ema_gap: gap mínimo entre EMA_short y EMA_long para permitir update (en unidades de precio)
        allow_reentry_after_exit: si True permite volver a entrar sin esperar nueva señal (rápido)
    
    Returns:
        df_out: copia de df con columnas añadidas:
            - smart_trailing_stop: nivel actual del stop
            - smart_max_price: máximo alcanzado desde la entrada
            - smart_exit_signal: -1 si sale por stop; 0 otherwise
            - smart_break_even: booleano si break-even fue activado
    """
    df = df.copy().reset_index(drop=False)  # keep timestamp if present as column 0
    n = len(df)
    df['smart_trailing_stop'] = np.nan
    df['smart_max_price'] = np.nan
    df['smart_exit_signal'] = 0
    df['smart_break_even'] = False
    in_trade = False
    entry_price = np.nan
    max_price = np.nan
    trailing_stop = np.nan
    # optional: compute TP estimate if you have stop/tp logic elsewhere; here we just use entry+3*atr for reference
    df['smart_tp_est'] = np.nan

    for i in range(n):
        price = df.at[i, price_col]
        atr = df.at[i, atr_col] if atr_col in df.columns else np.nan

        # new entry detection (only LONG handled here)
        if (not in_trade) and (entry_col in df.columns) and (df.at[i, entry_col] == 1):
            # start trade
            if np.isnan(atr) or atr <= 0:
                # fallback: small ATR using recent volatility
                recent = df[price_col].iloc[max(0, i-14):i+1]
                atr = recent.pct_change().std() * recent.mean() if len(recent)>1 else 0.01
            in_trade = True
            entry_price = price
            max_price = price
            trailing_stop = entry_price - atr * atr_mult
            df.at[i, 'smart_max_price'] = max_price
            df.at[i, 'smart_trailing_stop'] = trailing_stop
            df.at[i, 'smart_tp_est'] = entry_price + atr * atr_mult * 2  # heuristic TP est
            continue

        # if in trade, update trailing logic
        if in_trade:
            # update max price
            if price > max_price:
                # require minimum meaningful move to update (reduce noise)
                required_move = atr * min_move_to_update if (not np.isnan(atr)) else 0.0
                if (price - max_price) >= required_move:
                    # optional noise filter: only update trailing when short EMA > long EMA (alcista)
                    if not noise_filter or (ema_short_col in df.columns and ema_long_col in df.columns and
                                            (df.at[i, ema_short_col] - df.at[i, ema_long_col] >= noise_filter_ema_gap)):
                        max_price = price

            # recompute trailing as max(previous trailing, max_price - atr*atr_mult)
            if not np.isnan(atr):
                candidate_stop = max_price - atr * atr_mult
            else:
                candidate_stop = trailing_stop  # no change if no ATR

            # prevent moving stop backwards
            trailing_stop = max(trailing_stop, candidate_stop)

            # Swing low anchoring: compute local swing low in lookback window and optionally lock SL above it
            if swing_lookback and swing_lookback >= 2:
                start = max(0, i - swing_lookback)
                swing_low = df[price_col].iloc[start:i+1].min()
                # only move trailing stop up to be at least swing_low - small buffer
                # (for LONG, trailing_stop can't be below a swing low - we set to max)
                trailing_stop = max(trailing_stop, swing_low - 0.0)  # buffer zero; could use atr*0.1

            # Break-even: if price reached a high fraction of a typical TP, move SL to break-even + buffer
            # We'll estimate TP as entry + 2 * atr * atr_mult (heuristic). If price >= entry + (TP-entry) * break_even_pct -> move SL
            tp_est = entry_price + atr * atr_mult * 2
            if (price >= entry_price + (tp_est - entry_price) * break_even_pct):
                be_stop = entry_price + atr * break_even_buffer_atr_mult
                trailing_stop = max(trailing_stop, be_stop)
                df.at[i, 'smart_break_even'] = True

            df.at[i, 'smart_max_price'] = max_price
            df.at[i, 'smart_trailing_stop'] = trailing_stop
            df.at[i, 'smart_tp_est'] = tp_est

            # Check stop hit: use close price crossing below stop (you can change to low<=stop if prefer)
            # use price_col close as measurement currently in row i
            if price <= trailing_stop:
                # exit
                df.at[i, 'smart_exit_signal'] = -1
                in_trade = False
                # optionally reset entry_price etc.
                entry_price = np.nan
                max_price = np.nan
                trailing_stop = np.nan
                # if allow_reentry_after_exit is False, we will ignore entries until next new signal
                continue

            # Also allow strategy-based exit if a sell signal exists in entry_col (-1)
            if (entry_col in df.columns) and (df.at[i, entry_col] == -1):
                df.at[i, 'smart_exit_signal'] = -1
                in_trade = False
                entry_price = np.nan
                max_price = np.nan
                trailing_stop = np.nan
                continue

    # reindex back to original index
    df_out = df.set_index(df.columns[0])  # previous reset_index used timestamp as first col; restore
    # ensure types
    df_out['smart_exit_signal'] = df_out['smart_exit_signal'].astype(int)
    return df_out
