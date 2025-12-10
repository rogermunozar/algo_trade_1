# visualization/chart_plotter.py
"""
Módulo agnóstico para visualización de gráficos con mplfinance
Funciona con cualquier DataFrame OHLCV independientemente del broker
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import mplfinance as mpf
import numpy as np


def graficar_analisis(datos_calculados):
    """
    Función que genera el gráfico usando los datos calculados.
    Completamente agnóstica del broker - solo necesita el diccionario de datos.
    
    Args:
        datos_calculados (dict): Diccionario retornado por calcular_indicadores()
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
    
    # Crear plots adicionales para mplfinance
    apd_resistance = mpf.make_addplot(resistance_prices, type='scatter', 
                                      markersize=100, marker='v', color='red')
    apd_support = mpf.make_addplot(support_prices, type='scatter', 
                                   markersize=100, marker='^', color='green')
    
    apds = [
        mpf.make_addplot(dfX['RSI'], panel=2, color='purple', ylabel='RSI', 
                        ylim=(0, 100), width=1.5),
        mpf.make_addplot(dfX['RSI_30'], panel=2, color='green', 
                        linestyle='--', width=0.7),
        mpf.make_addplot(dfX['RSI_70'], panel=2, color='red', 
                        linestyle='--', width=0.7),
        apd_resistance,
        apd_support,
        mpf.make_addplot(support_prices, panel=1, color='blue', linestyle='--')
    ]
    
    # Crear gráfico
    fig, axes = mpf.plot(dfX, type='candle', style='charles', volume=True,
                        addplot=apds,
                        title=symbol + ' - ' + fecha_str + "   " + interval,
                        warn_too_much_data=1000,
                        figsize=(20, 8),
                        returnfig=True)
    
    # Configurar eje X
    ax = axes[0]
    ax.xaxis.set_major_locator(MaxNLocator(nbins=14))
    plt.setp(ax.get_xticklabels(), fontsize=6, rotation=45)
    
    if len(axes) > 1:
        ax_vol = axes[1]
        ax_vol.xaxis.set_major_locator(MaxNLocator(nbins=14))
    
    # Calcular offset para etiquetas
    price_range = dfX['high'].max() - dfX['low'].min()
    offset = price_range * 0.015
    
    # Agregar etiquetas y objetivos para RESISTENCIAS (SHORT)
    for idx in high_peaks_indices:
        price = dfX['high'].iloc[idx]
        ax.text(idx, price + offset, f'{price:.2f}',
                ha='center', va='bottom',
                fontsize=8, color='red',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        precio_objetivo_short = price * (1 - comision)
        ax.axhline(y=precio_objetivo_short, color='orange', linestyle=':',
                   linewidth=1.5, alpha=0.6, 
                   xmin=idx/len(dfX), xmax=min(1, (idx+30)/len(dfX)))
        
        ax.text(idx, precio_objetivo_short, f' TP SHORT: {precio_objetivo_short:.2f}',
                ha='left', va='center',
                fontsize=7, color='orange',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))
    
    # Agregar etiquetas y objetivos para SOPORTES (LONG)
    for idx in low_peaks_indices:
        price = dfX['low'].iloc[idx]
        ax.text(idx, price - offset, f'{price:.2f}',
                ha='center', va='top',
                fontsize=8, color='green',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        precio_objetivo_long = price * (1 + comision)
        ax.axhline(y=precio_objetivo_long, color='blue', linestyle=':',
                   linewidth=1.5, alpha=0.6,
                   xmin=idx/len(dfX), xmax=min(1, (idx+30)/len(dfX)))
        
        ax.text(idx, precio_objetivo_long, f' TP L: {precio_objetivo_long:.2f}',
                ha='left', va='center',
                fontsize=7, color='blue',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='cyan', alpha=0.8))
    
    # Dibujar líneas de tendencia
    if coeffs_resistance is not None:
        poly_resistance = np.poly1d(coeffs_resistance)
        x_line = np.arange(0, len(dfX))
        y_line_resistance = poly_resistance(x_line)
        ax.plot(x_line, y_line_resistance, color='red', linestyle='--',
                linewidth=2, alpha=0.7, 
                label=f'Resistencia (m={slope_resistance:.4f})')
    
    if coeffs_support is not None:
        poly_support = np.poly1d(coeffs_support)
        x_line = np.arange(0, len(dfX))
        y_line_support = poly_support(x_line)
        ax.plot(x_line, y_line_support, color='green', linestyle='--',
                linewidth=2, alpha=0.7,
                label=f'Soporte (m={slope_support:.4f})')
    
    ax.legend(loc='best', fontsize=8)
    plt.show()


def recalc(symbol, fecha_str, interval, df, comision=None, display=0):
    """
    Función wrapper para mantener compatibilidad con código existente.
    Ejecuta cálculos y gráficos en secuencia.
    
    Args:
        symbol: str - Símbolo del activo
        fecha_str: str - Fecha para identificación
        interval: str - Intervalo temporal
        df: DataFrame - Datos OHLCV
        comision: float - Comisión en decimal (opcional)
        display: int - Si 1, guarda CSV
    
    Returns:
        dict: Diccionario con niveles de soporte/resistencia y pendientes
    """
    from analysis.indicators import calcular_indicadores
    
    datos = calcular_indicadores(symbol, fecha_str, interval, df, comision, display)
    graficar_analisis(datos)
    
    return {
        'slope_resistance': datos['slope_resistance'],
        'slope_support': datos['slope_support'],
        'resistance_levels': datos['resistance_levels'],
        'support_levels': datos['support_levels']
    }