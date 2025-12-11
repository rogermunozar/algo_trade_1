# visualization/chart_plotter.py
"""
Módulo mejorado para visualización con marcadores de trades
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
import mplfinance as mpf
import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def graficar_analisis(datos_calculados: Dict, mostrar_avanzado: bool = True, 
                     mostrar_trades: bool = True, guardar: bool = False, 
                     filename: Optional[str] = None):
    """
    Función mejorada que genera el gráfico con visualización de trades
    
    Args:
        datos_calculados: Diccionario retornado por calcular_indicadores()
        mostrar_avanzado: Si True, muestra patrones y divergencias
        mostrar_trades: Si True, muestra entradas/salidas de trades
        guardar: Si True, guarda el gráfico como imagen
        filename: Nombre del archivo para guardar (opcional)
    """
    # Extraer datos del diccionario
    dfX = datos_calculados['dfX']
    symbol = datos_calculados['symbol']
    fecha_str = datos_calculados['fecha_str']
    interval = datos_calculados['interval']
    comision = datos_calculados['comision']
    resistance_prices = datos_calculados['resistance_prices']
    support_prices = datos_calculados['support_prices']
    high_peaks_indices = datos_calculados['high_peaks_indices']
    low_peaks_indices = datos_calculados['low_peaks_indices']
    slope_resistance = datos_calculados['slope_resistance']
    slope_support = datos_calculados['slope_support']
    coeffs_resistance = datos_calculados['coeffs_resistance']
    coeffs_support = datos_calculados['coeffs_support']
    
    # Datos avanzados
    divergencias = datos_calculados.get('divergencias', {})
    patrones_velas = datos_calculados.get('patrones_velas', {})
    niveles_fib = datos_calculados.get('niveles_fibonacci', {})
    tendencia = datos_calculados.get('tendencia', 'N/A')
    volatilidad = datos_calculados.get('volatilidad', 0)
    cambio_porcentual = datos_calculados.get('cambio_porcentual', 0)
    
    logger.info(f"Generando gráfico para {symbol}")
    
    # ========== DETECTAR SI HAY TRADES ==========
    tiene_trades = ('smart_exit_signal' in dfX.columns and 
                   (dfX['smart_exit_signal'] != 0).any())
    
    # ========== CREAR PLOTS ADICIONALES ==========
    apds = []
    
    # Soportes y resistencias
    apd_resistance = mpf.make_addplot(
        resistance_prices, 
        type='scatter', 
        markersize=120, 
        marker='v', 
        color='red',
        alpha=0.8
    )
    apd_support = mpf.make_addplot(
        support_prices, 
        type='scatter', 
        markersize=120, 
        marker='^', 
        color='green',
        alpha=0.8
    )
    apds.extend([apd_resistance, apd_support])
    
    # Medias móviles en el panel principal
    apds.append(mpf.make_addplot(dfX['SMA_20'], color='blue', width=1.5, alpha=0.7, label='SMA 20'))
    apds.append(mpf.make_addplot(dfX['SMA_50'], color='orange', width=1.5, alpha=0.7, label='SMA 50'))
    if 'SMA_200' in dfX.columns and not dfX['SMA_200'].isna().all():
        apds.append(mpf.make_addplot(dfX['SMA_200'], color='purple', width=2, alpha=0.5, label='SMA 200'))
    
    # Bandas de Bollinger
    apds.append(mpf.make_addplot(dfX['BB_upper'], color='gray', linestyle='--', width=1, alpha=0.5))
    apds.append(mpf.make_addplot(dfX['BB_lower'], color='gray', linestyle='--', width=1, alpha=0.5))
    
    # Trailing Stop (si existe)
    if tiene_trades and mostrar_trades and 'smart_trailing_stop' in dfX.columns:
        apds.append(mpf.make_addplot(
            dfX['smart_trailing_stop'], 
            color='red', 
            linestyle=':', 
            width=2, 
            alpha=0.7,
            label='Trailing Stop'
        ))
    
    # RSI en panel 2
    apds.append(mpf.make_addplot(dfX['RSI'], panel=2, color='purple', ylabel='RSI', 
                                 ylim=(0, 100), width=2))
    apds.append(mpf.make_addplot(dfX['RSI_30'], panel=2, color='green', 
                                 linestyle='--', width=1))
    apds.append(mpf.make_addplot(dfX['RSI_70'], panel=2, color='red', 
                                 linestyle='--', width=1))
    
    # MACD en panel 3
    apds.append(mpf.make_addplot(dfX['MACD'], panel=3, color='blue', 
                                 ylabel='MACD', width=1.5))
    apds.append(mpf.make_addplot(dfX['MACD_signal'], panel=3, color='red', 
                                 width=1.5))
    apds.append(mpf.make_addplot(dfX['MACD_histogram'], panel=3, 
                                 type='bar', color='gray', alpha=0.5))
    
    # Señales de trading (solo si NO hay trades del Smart Exit)
    if mostrar_avanzado and 'senal' in dfX.columns and not tiene_trades:
        # Señales de compra (verde)
        compras = dfX['close'].copy()
        compras[dfX['senal'] != 1] = np.nan
        apds.append(mpf.make_addplot(compras, type='scatter', markersize=150, 
                                     marker='^', color='lime', alpha=0.9))
        
        # Señales de venta (rojo)
        ventas = dfX['close'].copy()
        ventas[dfX['senal'] != -1] = np.nan
        apds.append(mpf.make_addplot(ventas, type='scatter', markersize=150, 
                                     marker='v', color='red', alpha=0.9))
    
    # ========== CREAR TÍTULO INFORMATIVO ==========
    titulo = (f"{symbol} - {fecha_str} - {interval}\n"
             f"Tendencia: {tendencia} | Volatilidad: {volatilidad:.2f}% | "
             f"Cambio: {cambio_porcentual:+.2f}%")
    
    if tiene_trades:
        exits = dfX[dfX['smart_exit_signal'] != 0]
        if len(exits) > 0:
            pnl_total = exits['smart_trade_pnl'].sum()
            win_rate = (exits['smart_trade_pnl'] > 0).sum() / len(exits) * 100
            titulo += f"\nTrades: {len(exits)} | P&L: ${pnl_total:+.2f} | Win Rate: {win_rate:.1f}%"
    
    # ========== CONFIGURAR ESTILO ==========
    mc = mpf.make_marketcolors(
        up='#26A69A',
        down='#EF5350',
        edge='inherit',
        wick='inherit',
        volume='in',
        alpha=0.9
    )
    
    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='-',
        gridcolor='#E0E0E0',
        facecolor='white',
        figcolor='white',
        gridaxis='both',
        y_on_right=False
    )
    
    # ========== CREAR GRÁFICO ==========
    fig, axes = mpf.plot(
        dfX, 
        type='candle', 
        style=s,
        volume=True,
        addplot=apds,
        title=titulo,
        warn_too_much_data=2000,
        figsize=(24, 14),
        panel_ratios=(6, 2, 2, 2),
        returnfig=True,
        tight_layout=True
    )
    
    ax = axes[0]  # Panel principal
    
    # ========== CONFIGURAR EJES ==========
    ax.xaxis.set_major_locator(MaxNLocator(nbins=20))
    plt.setp(ax.get_xticklabels(), fontsize=8, rotation=45)
    
    for i in range(1, len(axes)):
        axes[i].xaxis.set_major_locator(MaxNLocator(nbins=20))
        plt.setp(axes[i].get_xticklabels(), fontsize=8, rotation=45)
    
    # ========== MARCAR TRADES ==========
    if tiene_trades and mostrar_trades:
        _marcar_trades_en_grafico(ax, dfX)
    
    # ========== AGREGAR ANOTACIONES DE NIVELES ==========
    price_range = dfX['high'].max() - dfX['low'].min()
    offset_up = price_range * 0.02
    offset_down = price_range * 0.02
    
    # RESISTENCIAS
    for idx in high_peaks_indices:
        price = dfX['high'].iloc[idx]
        
        ax.text(
            idx, price + offset_up, f'${price:.2f}',
            ha='center', va='bottom',
            fontsize=9, color='darkred',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor='red', alpha=0.8, linewidth=2)
        )
        
        precio_objetivo_short = price * (1 - comision)
        ax.axhline(
            y=precio_objetivo_short, 
            color='orange', 
            linestyle=':', 
            linewidth=2, 
            alpha=0.6,
            xmin=idx/len(dfX), 
            xmax=min(1, (idx+40)/len(dfX))
        )
        
        ax.text(
            idx, precio_objetivo_short, 
            f' TP SHORT: ${precio_objetivo_short:.2f}',
            ha='left', va='center',
            fontsize=8, color='darkorange',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', 
                     edgecolor='orange', alpha=0.9, linewidth=1.5)
        )
    
    # SOPORTES
    for idx in low_peaks_indices:
        price = dfX['low'].iloc[idx]
        
        ax.text(
            idx, price - offset_down, f'${price:.2f}',
            ha='center', va='top',
            fontsize=9, color='darkgreen',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor='green', alpha=0.8, linewidth=2)
        )
        
        precio_objetivo_long = price * (1 + comision)
        ax.axhline(
            y=precio_objetivo_long, 
            color='dodgerblue', 
            linestyle=':', 
            linewidth=2, 
            alpha=0.6,
            xmin=idx/len(dfX), 
            xmax=min(1, (idx+40)/len(dfX))
        )
        
        ax.text(
            idx, precio_objetivo_long, 
            f' TP LONG: ${precio_objetivo_long:.2f}',
            ha='left', va='center',
            fontsize=8, color='darkblue',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E1F5FE', 
                     edgecolor='dodgerblue', alpha=0.9, linewidth=1.5)
        )
    
    # ========== LÍNEAS DE TENDENCIA ==========
    if coeffs_resistance is not None:
        poly_resistance = np.poly1d(coeffs_resistance)
        x_line = np.arange(0, len(dfX))
        y_line_resistance = poly_resistance(x_line)
        ax.plot(
            x_line, y_line_resistance, 
            color='red', 
            linestyle='--',
            linewidth=2.5, 
            alpha=0.7, 
            label=f'Resistencia (m={slope_resistance:.6f})'
        )
    
    if coeffs_support is not None:
        poly_support = np.poly1d(coeffs_support)
        x_line = np.arange(0, len(dfX))
        y_line_support = poly_support(x_line)
        ax.plot(
            x_line, y_line_support, 
            color='green', 
            linestyle='--',
            linewidth=2.5, 
            alpha=0.7,
            label=f'Soporte (m={slope_support:.6f})'
        )
    
    # ========== NIVELES DE FIBONACCI ==========
    if mostrar_avanzado and niveles_fib:
        niveles_mostrar = ['nivel_236', 'nivel_382', 'nivel_500', 'nivel_618']
        colores_fib = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        for nivel, color in zip(niveles_mostrar, colores_fib):
            if nivel in niveles_fib:
                precio_fib = niveles_fib[nivel]
                ax.axhline(
                    y=precio_fib, 
                    color=color, 
                    linestyle=':', 
                    linewidth=1, 
                    alpha=0.4
                )
                ax.text(
                    len(dfX) - 1, precio_fib, 
                    f' Fib {nivel.split("_")[1]}',
                    ha='left', va='center',
                    fontsize=7, color=color,
                    alpha=0.7
                )
    
    # ========== MARCAR PATRONES DE VELAS ==========
    if mostrar_avanzado and patrones_velas and not tiene_trades:
        for idx in patrones_velas.get('hammer', []):
            if idx < len(dfX):
                ax.annotate(
                    'H', 
                    xy=(idx, dfX['low'].iloc[idx]), 
                    xytext=(0, -15),
                    textcoords='offset points',
                    fontsize=10,
                    color='green',
                    fontweight='bold',
                    ha='center'
                )
        
        for idx in patrones_velas.get('shooting_star', []):
            if idx < len(dfX):
                ax.annotate(
                    'S', 
                    xy=(idx, dfX['high'].iloc[idx]), 
                    xytext=(0, 15),
                    textcoords='offset points',
                    fontsize=10,
                    color='red',
                    fontweight='bold',
                    ha='center'
                )
        
        for idx in patrones_velas.get('engulfing_alcista', []):
            if idx < len(dfX):
                ax.annotate(
                    'E+', 
                    xy=(idx, dfX['low'].iloc[idx]), 
                    xytext=(0, -20),
                    textcoords='offset points',
                    fontsize=9,
                    color='darkgreen',
                    fontweight='bold',
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.7)
                )
        
        for idx in patrones_velas.get('engulfing_bajista', []):
            if idx < len(dfX):
                ax.annotate(
                    'E-', 
                    xy=(idx, dfX['high'].iloc[idx]), 
                    xytext=(0, 20),
                    textcoords='offset points',
                    fontsize=9,
                    color='darkred',
                    fontweight='bold',
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcoral', alpha=0.7)
                )
    
    # ========== LEYENDA ==========
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    # ========== GUARDAR O MOSTRAR ==========
    if guardar:
        if filename is None:
            filename = f'chart_{symbol}_{fecha_str}_{interval}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        logger.info(f"Gráfico guardado: {filename}")
    
    plt.show()
    logger.info("Gráfico generado exitosamente")


def _marcar_trades_en_grafico(ax, df: pd.DataFrame):
    """
    Marca las entradas y salidas de trades en el gráfico
    
    Args:
        ax: Eje de matplotlib
        df: DataFrame con información de trades
    """
    # Detectar entradas (cambio de NaN a valor en trade_id)
    df = df.reset_index(drop=True)
    
    entries = []
    exits = []
    current_trade_id = None
    
    for i in range(len(df)):
        trade_id = df.at[i, 'smart_trade_id']
        
        # Detectar entrada (nueva trade_id)
        if pd.notna(trade_id) and trade_id != current_trade_id:
            current_trade_id = trade_id
            entries.append(i)
        
        # Detectar salida (exit_signal != 0)
        if df.at[i, 'smart_exit_signal'] != 0:
            exits.append(i)
            current_trade_id = None
    
    # Marcar ENTRADAS
    for idx in entries:
        price = df.at[idx, 'close']
        position_type = df.at[idx, 'smart_position_type']
        
        if position_type == 'LONG':
            # Flecha verde apuntando arriba
            ax.annotate(
                'LONG', 
                xy=(idx, price),
                xytext=(0, -30),
                textcoords='offset points',
                ha='center',
                fontsize=10,
                fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='green', 
                         edgecolor='darkgreen', linewidth=2, alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='green', lw=2)
            )
        else:  # SHORT
            # Flecha roja apuntando abajo
            ax.annotate(
                'SHORT', 
                xy=(idx, price),
                xytext=(0, 30),
                textcoords='offset points',
                ha='center',
                fontsize=10,
                fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='red', 
                         edgecolor='darkred', linewidth=2, alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='red', lw=2)
            )
    
    # Marcar SALIDAS
    for idx in exits:
        price = df.at[idx, 'close']
        pnl = df.at[idx, 'smart_trade_pnl']
        exit_type = df.at[idx, 'smart_exit_signal']
        position_type = df.at[idx, 'smart_position_type']
        
        # Determinar tipo de salida
        if exit_type == 1:
            exit_label = 'TP'
            exit_color = 'blue'
        else:
            exit_label = 'STOP'
            exit_color = 'orange'
        
        # Determinar si fue ganador o perdedor
        if pnl > 0:
            bg_color = '#90EE90'  # Verde claro
            edge_color = 'green'
            emoji = '✓'
        else:
            bg_color = '#FFB6C1'  # Rosa claro
            edge_color = 'red'
            emoji = '✗'
        
        # Crear etiqueta con P&L
        pnl_pct = (pnl / price) * 100
        label = f'{emoji} {exit_label}\n${pnl:+.2f}\n({pnl_pct:+.1f}%)'
        
        offset_y = -40 if position_type == 'LONG' else 40
        
        ax.annotate(
            label,
            xy=(idx, price),
            xytext=(0, offset_y),
            textcoords='offset points',
            ha='center',
            fontsize=9,
            fontweight='bold',
            color='black',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=bg_color, 
                     edgecolor=edge_color, linewidth=2, alpha=0.9),
            arrowprops=dict(arrowstyle='->', color=exit_color, lw=2)
        )
    
    # Resaltar zonas de trades con fondo
    for i, entry_idx in enumerate(entries):
        if i < len(exits):
            exit_idx = exits[i]
            entry_price = df.at[entry_idx, 'close']
            exit_price = df.at[exit_idx, 'close']
            pnl = df.at[exit_idx, 'smart_trade_pnl']
            
            # Color de fondo según resultado
            if pnl > 0:
                bg_color = 'lightgreen'
            else:
                bg_color = 'lightcoral'
            
            # Agregar rectángulo de fondo
            y_min = min(entry_price, exit_price) * 0.995
            y_max = max(entry_price, exit_price) * 1.005
            
            rect = Rectangle(
                (entry_idx, y_min),
                exit_idx - entry_idx,
                y_max - y_min,
                facecolor=bg_color,
                alpha=0.1,
                zorder=0
            )
            ax.add_patch(rect)


def graficar_comparacion(datos_list: list, symbols: list):
    """
    Grafica múltiples activos para comparación
    
    Args:
        datos_list: Lista de diccionarios de datos calculados
        symbols: Lista de símbolos correspondientes
    """
    fig, axes = plt.subplots(len(datos_list), 1, figsize=(22, 6*len(datos_list)))
    
    if len(datos_list) == 1:
        axes = [axes]
    
    for i, (datos, symbol) in enumerate(zip(datos_list, symbols)):
        dfX = datos['dfX']
        
        # Normalizar precios para comparación
        precio_inicial = dfX['close'].iloc[0]
        dfX_norm = dfX['close'] / precio_inicial * 100
        
        axes[i].plot(dfX_norm, label=symbol, linewidth=2, color=f'C{i}')
        axes[i].set_title(f'{symbol} - Evolución Normalizada (Base 100)', fontsize=14, fontweight='bold')
        axes[i].set_ylabel('Precio Normalizado', fontsize=12)
        axes[i].grid(True, alpha=0.3, linestyle='--')
        axes[i].legend(fontsize=12)
        
        # Agregar info de tendencia
        tendencia = datos.get('tendencia', 'N/A')
        cambio = datos.get('cambio_porcentual', 0)
        axes[i].text(
            0.02, 0.98, 
            f'Tendencia: {tendencia}\nCambio: {cambio:+.2f}%',
            transform=axes[i].transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10
        )
    
    plt.tight_layout()
    plt.show()


def recalc(symbol: str, fecha_str: str, interval: str, df, 
          comision: Optional[float] = None, display: int = 0) -> Dict:
    """
    Función wrapper para mantener compatibilidad
    """
    from analysis.indicators import calcular_indicadores
    
    logger.info(f"Ejecutando recalc para {symbol}")
    
    datos = calcular_indicadores(symbol, fecha_str, interval, df, comision, display)
    graficar_analisis(datos, mostrar_avanzado=True)
    
    return {
        'slope_resistance': datos['slope_resistance'],
        'slope_support': datos['slope_support'],
        'resistance_levels': datos['resistance_levels'],
        'support_levels': datos['support_levels'],
        'tendencia': datos.get('tendencia'),
        'volatilidad': datos.get('volatilidad'),
        'precio_actual': datos.get('precio_actual'),
        'sl_long': datos.get('sl_long'),
        'tp_long': datos.get('tp_long'),
        'sl_short': datos.get('sl_short'),
        'tp_short': datos.get('tp_short')
    }