# visualization/chart_plotter.py
"""
Módulo mejorado para visualización de gráficos con mplfinance
Incluye: más indicadores, mejor diseño, y gráficos interactivos
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import mplfinance as mpf
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def graficar_analisis(datos_calculados: Dict, mostrar_avanzado: bool = True, 
                     guardar: bool = False, filename: Optional[str] = None):
    """
    Función mejorada que genera el gráfico usando los datos calculados.
    Completamente agnóstica del broker - solo necesita el diccionario de datos.
    
    Args:
        datos_calculados: Diccionario retornado por calcular_indicadores()
        mostrar_avanzado: Si True, muestra patrones y divergencias
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
    
    # Datos avanzados (si existen)
    divergencias = datos_calculados.get('divergencias', {})
    patrones_velas = datos_calculados.get('patrones_velas', {})
    niveles_fib = datos_calculados.get('niveles_fibonacci', {})
    tendencia = datos_calculados.get('tendencia', 'N/A')
    volatilidad = datos_calculados.get('volatilidad', 0)
    cambio_porcentual = datos_calculados.get('cambio_porcentual', 0)
    
    logger.info(f"Generando gráfico para {symbol}")
    
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
    apds.append(mpf.make_addplot(dfX['SMA_20'], color='blue', width=1.5, alpha=0.7))
    apds.append(mpf.make_addplot(dfX['SMA_50'], color='orange', width=1.5, alpha=0.7))
    if 'SMA_200' in dfX.columns and not dfX['SMA_200'].isna().all():
        apds.append(mpf.make_addplot(dfX['SMA_200'], color='purple', width=2, alpha=0.5))
    
    # Bandas de Bollinger
    apds.append(mpf.make_addplot(dfX['BB_upper'], color='gray', linestyle='--', width=1, alpha=0.5))
    apds.append(mpf.make_addplot(dfX['BB_lower'], color='gray', linestyle='--', width=1, alpha=0.5))
    
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
    
    # Señales de trading (si mostrar_avanzado)
    if mostrar_avanzado and 'senal' in dfX.columns:
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
    
    # ========== CONFIGURAR ESTILO ==========
    
    # Estilo personalizado
    mc = mpf.make_marketcolors(
        up='#26A69A',      # Verde para velas alcistas
        down='#EF5350',    # Rojo para velas bajistas
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
        figsize=(22, 12),
        panel_ratios=(6, 2, 2, 2),  # Ratios de tamaño de paneles
        returnfig=True,
        tight_layout=True
    )
    
    ax = axes[0]  # Panel principal (velas)
    
    # ========== CONFIGURAR EJES ==========
    
    ax.xaxis.set_major_locator(MaxNLocator(nbins=20))
    plt.setp(ax.get_xticklabels(), fontsize=8, rotation=45)
    
    # Configurar otros paneles
    for i in range(1, len(axes)):
        axes[i].xaxis.set_major_locator(MaxNLocator(nbins=20))
        plt.setp(axes[i].get_xticklabels(), fontsize=8, rotation=45)
    
    # ========== AGREGAR ANOTACIONES ==========
    
    price_range = dfX['high'].max() - dfX['low'].min()
    offset_up = price_range * 0.02
    offset_down = price_range * 0.02
    
    # RESISTENCIAS con objetivos SHORT
    for idx in high_peaks_indices:
        price = dfX['high'].iloc[idx]
        
        # Etiqueta de precio
        ax.text(
            idx, price + offset_up, f'{price:.2f}',
            ha='center', va='bottom',
            fontsize=9, color='darkred',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor='red', alpha=0.8, linewidth=2)
        )
        
        # Precio objetivo SHORT
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
        
        # Etiqueta objetivo SHORT
        ax.text(
            idx, precio_objetivo_short, 
            f' TP SHORT: {precio_objetivo_short:.2f}',
            ha='left', va='center',
            fontsize=8, color='darkorange',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', 
                     edgecolor='orange', alpha=0.9, linewidth=1.5)
        )
    
    # SOPORTES con objetivos LONG
    for idx in low_peaks_indices:
        price = dfX['low'].iloc[idx]
        
        # Etiqueta de precio
        ax.text(
            idx, price - offset_down, f'{price:.2f}',
            ha='center', va='top',
            fontsize=9, color='darkgreen',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor='green', alpha=0.8, linewidth=2)
        )
        
        # Precio objetivo LONG
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
        
        # Etiqueta objetivo LONG
        ax.text(
            idx, precio_objetivo_long, 
            f' TP LONG: {precio_objetivo_long:.2f}',
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
            label=f'Línea Resistencia (m={slope_resistance:.6f})'
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
            label=f'Línea Soporte (m={slope_support:.6f})'
        )
    
    # ========== NIVELES DE FIBONACCI ==========
    
    if mostrar_avanzado and niveles_fib:
        # Solo mostrar algunos niveles clave
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
                # Etiqueta pequeña a la derecha
                ax.text(
                    len(dfX) - 1, precio_fib, 
                    f' Fib {nivel.split("_")[1]}',
                    ha='left', va='center',
                    fontsize=7, color=color,
                    alpha=0.7
                )
    
    # ========== MARCAR PATRONES DE VELAS ==========
    
    if mostrar_avanzado and patrones_velas:
        # Hammers (alcistas)
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
        
        # Shooting stars (bajistas)
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
        
        # Engulfing alcista
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
        
        # Engulfing bajista
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


def graficar_comparacion(datos_list: list, symbols: list):
    """
    Grafica múltiples activos para comparación
    
    Args:
        datos_list: Lista de diccionarios de datos calculados
        symbols: Lista de símbolos correspondientes
    """
    fig, axes = plt.subplots(len(datos_list), 1, figsize=(20, 6*len(datos_list)))
    
    if len(datos_list) == 1:
        axes = [axes]
    
    for i, (datos, symbol) in enumerate(zip(datos_list, symbols)):
        dfX = datos['dfX']
        
        # Normalizar precios para comparación
        precio_inicial = dfX['close'].iloc[0]
        dfX_norm = dfX['close'] / precio_inicial * 100
        
        axes[i].plot(dfX_norm, label=symbol, linewidth=2)
        axes[i].set_title(f'{symbol} - Evolución Normalizada (Base 100)')
        axes[i].set_ylabel('Precio Normalizado')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()


def recalc(symbol: str, fecha_str: str, interval: str, df, 
          comision: Optional[float] = None, display: int = 0) -> Dict:
    """
    Función wrapper mejorada para mantener compatibilidad con código existente.
    Ejecuta cálculos y gráficos en secuencia.
    
    Args:
        symbol: Símbolo del activo
        fecha_str: Fecha para identificación
        interval: Intervalo temporal
        df: DataFrame con datos OHLCV
        comision: Comisión en decimal (opcional)
        display: Si 1, guarda CSV
    
    Returns:
        Dict: Diccionario completo con todos los resultados
    """
    from analysis.indicators import calcular_indicadores
    
    logger.info(f"Ejecutando recalc para {symbol}")
    
    # Calcular indicadores
    datos = calcular_indicadores(symbol, fecha_str, interval, df, comision, display)
    
    # Graficar
    graficar_analisis(datos, mostrar_avanzado=True)
    
    # Retornar resumen
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