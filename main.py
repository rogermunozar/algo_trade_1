# main.py
"""
Script principal mejorado con sistema de Smart Exit y métricas avanzadas
Versión limpia sin duplicados
"""
from clients.binance_client import BinanceClient, load_secrets
from analysis.indicators import calcular_indicadores
from analysis.risk_management import apply_smart_exit, calcular_metricas_trailing
from visualization.chart_plotter import graficar_analisis, graficar_comparacion
import logging
import pandas as pd
import sys
import argparse

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ejemplo_basico(symbol='BTCUSDT', interval='1h', limit=500):
    """
    Ejemplo básico: análisis técnico estándar sin Smart Exit
    
    Args:
        symbol: Par de trading (ej: 'BTCUSDT', 'ETHUSDT')
        interval: Intervalo temporal (ej: '1h', '15m', '1d')
        limit: Cantidad de velas
    """
    logger.info(f"Ejecutando ejemplo básico para {symbol}")
    
    print("\n" + "="*60)
    print(f"📊 ANÁLISIS BÁSICO - {symbol}")
    print("="*60)
    
    # Inicializar cliente
    binance = BinanceClient()
    
    # Obtener datos
    print(f"\n🔍 Obteniendo {limit} velas de {symbol} ({interval})...")
    df = binance.history(symbol, interval, limit)
    
    # Calcular indicadores
    print("🔧 Calculando indicadores técnicos...")
    fecha = df.index[-1].strftime('%Y-%m-%d')
    comision = binance.calcular_comision(symbol)
    datos = calcular_indicadores(symbol, fecha, interval, df, comision)
    
    # Mostrar resultados
    _mostrar_resultados_basicos(datos, symbol)
    
    # Graficar
    print("\n📈 Generando gráfico...")
    graficar_analisis(datos, mostrar_avanzado=True)


def ejemplo_con_smart_exit(symbol='BTCUSDT', interval='1h', limit=500):
    """
    Ejemplo con Smart Exit: gestión avanzada de riesgo con trailing stop
    
    Args:
        symbol: Par de trading
        interval: Intervalo temporal
        limit: Cantidad de velas
    """
    logger.info(f"Ejecutando Smart Exit para {symbol}")
    
    print("\n" + "="*60)
    print(f"🎯 SMART EXIT - {symbol}")
    print("="*60)
    
    # Inicializar cliente
    binance = BinanceClient()
    
    # Obtener datos
    print(f"\n🔍 Obteniendo {limit} velas de {symbol} ({interval})...")
    df = binance.history(symbol, interval, limit)
    
    # Calcular indicadores
    print("🔧 Calculando indicadores técnicos...")
    fecha = df.index[-1].strftime('%Y-%m-%d')
    comision = binance.calcular_comision(symbol)
    datos = calcular_indicadores(symbol, fecha, interval, df, comision)
    
    dfX = datos['dfX']
    
    # Aplicar Smart Exit
    print("🎯 Aplicando sistema de Smart Exit...")
    df_with_exit = apply_smart_exit(
        dfX,
        entry_col='senal',
        price_col='close',
        atr_col='ATR',
        ema_short_col='EMA_12',
        ema_long_col='EMA_26',
        atr_mult=1.5,
        tp_multiplier=2.0,
        break_even_pct=0.9,
        log_trades=True
    )
    
    # Calcular métricas
    metricas = calcular_metricas_trailing(df_with_exit)
    
    # Mostrar resultados
    _mostrar_resultados_basicos(datos, symbol)
    _mostrar_metricas_smart_exit(metricas)
    _mostrar_senal_actual(dfX, df_with_exit, symbol)
    
    # Graficar
    print("\n📈 Generando gráfico...")
    graficar_analisis(datos, mostrar_avanzado=True)


def ejemplo_optimizacion(symbol='ETHUSDT', interval='1h', limit=1000):
    """
    Optimización de parámetros: prueba múltiples configuraciones
    
    Args:
        symbol: Par de trading
        interval: Intervalo temporal
        limit: Cantidad de velas
    """
    logger.info(f"Optimizando parámetros para {symbol}")
    
    print("\n" + "="*60)
    print(f"⚙️  OPTIMIZACIÓN DE PARÁMETROS - {symbol}")
    print("="*60)
    
    binance = BinanceClient()
    
    # Obtener datos
    print(f"\n🔍 Obteniendo {limit} velas...")
    df = binance.history(symbol, interval, limit)
    fecha = df.index[-1].strftime('%Y-%m-%d')
    comision = binance.calcular_comision(symbol)
    datos = calcular_indicadores(symbol, fecha, interval, df, comision)
    dfX = datos['dfX']
    
    # Configuraciones a probar
    configuraciones = [
        {
            'nombre': 'Conservador',
            'atr_mult': 2.0,
            'tp_mult': 3.0,
            'break_even_pct': 0.8
        },
        {
            'nombre': 'Balanceado',
            'atr_mult': 1.5,
            'tp_mult': 2.0,
            'break_even_pct': 0.9
        },
        {
            'nombre': 'Agresivo',
            'atr_mult': 1.0,
            'tp_mult': 1.5,
            'break_even_pct': 0.95
        },
    ]
    
    print(f"\n🧪 Probando {len(configuraciones)} configuraciones...\n")
    
    resultados = []
    
    for config in configuraciones:
        df_test = apply_smart_exit(
            dfX.copy(),
            entry_col='senal',
            atr_mult=config['atr_mult'],
            tp_multiplier=config['tp_mult'],
            break_even_pct=config['break_even_pct'],
            log_trades=False
        )
        
        metricas = calcular_metricas_trailing(df_test)
        
        if metricas and metricas.get('total_trades', 0) > 0:
            resultados.append({
                'nombre': config['nombre'],
                'trades': metricas['total_trades'],
                'win_rate': metricas['win_rate'],
                'profit_factor': metricas['profit_factor'],
                'pnl_total': metricas['pnl_total']
            })
    
    # Mostrar resultados
    _mostrar_resultados_optimizacion(resultados, symbol)


def ejemplo_monitoreo(symbol='BNBUSDT', interval='5m', limit=100):
    """
    Simulación de monitoreo en tiempo real de una posición
    
    Args:
        symbol: Par de trading
        interval: Intervalo temporal
        limit: Cantidad de velas
    """
    logger.info(f"Monitoreando {symbol}")
    
    print("\n" + "="*60)
    print(f"📡 MONITOREO EN TIEMPO REAL - {symbol}")
    print("="*60)
    
    binance = BinanceClient()
    
    # Obtener datos recientes
    print(f"\n🔍 Obteniendo {limit} velas...")
    df = binance.history(symbol, interval, limit)
    fecha = df.index[-1].strftime('%Y-%m-%d')
    comision = binance.calcular_comision(symbol)
    datos = calcular_indicadores(symbol, fecha, interval, df, comision)
    dfX = datos['dfX']
    
    # Aplicar smart exit
    df_with_exit = apply_smart_exit(dfX, entry_col='senal', log_trades=False)
    
    # Mostrar estado de posición
    _mostrar_estado_posicion(df_with_exit, symbol)


def ejemplo_multi_timeframe(symbol='BTCUSDT'):
    """
    Análisis del mismo activo en múltiples timeframes
    
    Args:
        symbol: Par de trading
    """
    logger.info(f"Análisis multi-timeframe para {symbol}")
    
    print("\n" + "="*60)
    print(f"⏱️  ANÁLISIS MULTI-TIMEFRAME - {symbol}")
    print("="*60)
    
    binance = BinanceClient()
    timeframes = ['15m', '1h', '4h']
    
    for tf in timeframes:
        print(f"\n--- {symbol} en {tf} ---")
        
        df = binance.history(symbol, tf, 500)
        fecha = df.index[-1].strftime('%Y-%m-%d')
        comision = binance.calcular_comision(symbol)
        datos = calcular_indicadores(symbol, fecha, tf, df, comision)
        
        print(f"Tendencia: {datos['tendencia']}")
        print(f"Volatilidad: {datos['volatilidad']:.2f}%")
        print(f"Precio: ${datos['precio_actual']:.2f}")


def ejemplo_comparacion(symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'], interval='1h'):
    """
    Comparación de múltiples activos
    
    Args:
        symbols: Lista de pares de trading
        interval: Intervalo temporal
    """
    logger.info(f"Comparando {len(symbols)} activos")
    
    print("\n" + "="*60)
    print(f"🔬 COMPARACIÓN DE ACTIVOS")
    print("="*60)
    
    binance = BinanceClient()
    resultados = []
    datos_list = []
    
    for symbol in symbols:
        print(f"\nAnalizando {symbol}...")
        
        df = binance.history(symbol, interval, 500)
        fecha = df.index[-1].strftime('%Y-%m-%d')
        comision = binance.calcular_comision(symbol)
        datos = calcular_indicadores(symbol, fecha, interval, df, comision)
        
        datos_list.append(datos)
        resultados.append({
            'symbol': symbol,
            'precio': datos['precio_actual'],
            'tendencia': datos['tendencia'],
            'volatilidad': datos['volatilidad'],
            'cambio': datos['cambio_porcentual']
        })
    
    # Mostrar tabla comparativa
    _mostrar_tabla_comparacion(resultados)
    
    # Graficar comparación
    graficar_comparacion(datos_list, symbols)


# ========================================
# FUNCIONES AUXILIARES PARA MOSTRAR INFO
# ========================================

def _mostrar_resultados_basicos(datos, symbol):
    """Muestra resultados básicos del análisis"""
    print(f"\n{'='*60}")
    print(f"📊 ANÁLISIS DE {symbol}")
    print(f"{'='*60}")
    print(f"💰 Precio actual: ${datos['precio_actual']:.2f}")
    print(f"📈 Tendencia: {datos['tendencia']}")
    print(f"📊 Volatilidad: {datos['volatilidad']:.2f}%")
    print(f"📉 Cambio: {datos['cambio_porcentual']:+.2f}%")
    print(f"🎯 ATR: ${datos['ultimo_atr']:.2f}")
    
    print(f"\n🎯 NIVELES CLAVE:")
    print(f"  Resistencias: {len(datos['resistance_levels'])} detectadas")
    if datos['resistance_levels']:
        print(f"    Más cercana: ${max(datos['resistance_levels']):.2f}")
    print(f"  Soportes: {len(datos['support_levels'])} detectados")
    if datos['support_levels']:
        print(f"    Más cercano: ${max(datos['support_levels']):.2f}")
    
    print(f"\n💼 GESTIÓN DE RIESGO:")
    print(f"  LONG:")
    print(f"    Stop Loss: ${datos['sl_long']:.2f} "
          f"({((datos['sl_long']/datos['precio_actual']-1)*100):.2f}%)")
    print(f"    Take Profit: ${datos['tp_long']:.2f} "
          f"({((datos['tp_long']/datos['precio_actual']-1)*100):.2f}%)")
    
    # Patrones
    if datos.get('patrones_velas'):
        patrones_detectados = {k: v for k, v in datos['patrones_velas'].items() if v}
        if patrones_detectados:
            print(f"\n🕯️  PATRONES DETECTADOS:")
            for patron, indices in patrones_detectados.items():
                print(f"  {patron.replace('_', ' ').title()}: {len(indices)}")
    
    # Divergencias
    if datos.get('divergencias'):
        if datos['divergencias']['alcistas'] or datos['divergencias']['bajistas']:
            print(f"\n📉 DIVERGENCIAS RSI:")
            print(f"  Alcistas: {len(datos['divergencias']['alcistas'])}")
            print(f"  Bajistas: {len(datos['divergencias']['bajistas'])}")


def _mostrar_metricas_smart_exit(metricas):
    """Muestra métricas del Smart Exit"""
    if not metricas or metricas.get('total_trades', 0) == 0:
        print(f"\n⚠️  No se detectaron trades en el período")
        return
    
    print(f"\n{'='*60}")
    print(f"🎯 MÉTRICAS DEL SMART EXIT (Backtest)")
    print(f"{'='*60}")
    print(f"📊 Trades totales: {metricas['total_trades']}")
    print(f"   ├─ Por Stop: {metricas['exits_por_stop']}")
    print(f"   └─ Por TP: {metricas['exits_por_tp']}")
    
    print(f"\n💰 Performance:")
    print(f"   ├─ Ganadores: {metricas['winning_trades']} "
          f"({metricas['win_rate']:.1f}%)")
    print(f"   ├─ Perdedores: {metricas['losing_trades']}")
    print(f"   └─ Profit Factor: {metricas['profit_factor']:.2f}")
    
    print(f"\n📈 P&L:")
    print(f"   ├─ Total: ${metricas['pnl_total']:+.2f}")
    print(f"   ├─ Promedio: ${metricas['pnl_medio']:+.2f}")
    print(f"   ├─ Avg Win: ${metricas['avg_win']:+.2f}")
    print(f"   └─ Avg Loss: ${metricas['avg_loss']:+.2f}")
    
    # Evaluación
    print(f"\n🎯 EVALUACIÓN:")
    if metricas['win_rate'] >= 60:
        print(f"   ✅ Win Rate excelente (≥60%)")
    elif metricas['win_rate'] >= 50:
        print(f"   ⚠️  Win Rate aceptable (50-60%)")
    else:
        print(f"   ❌ Win Rate bajo (<50%)")
    
    if metricas['profit_factor'] >= 2.0:
        print(f"   ✅ Profit Factor excelente (≥2.0)")
    elif metricas['profit_factor'] >= 1.5:
        print(f"   ⚠️  Profit Factor aceptable (1.5-2.0)")
    else:
        print(f"   ❌ Profit Factor bajo (<1.5)")


def _mostrar_senal_actual(dfX, df_with_exit, symbol):
    """Muestra la señal actual del mercado"""
    ultima_senal = dfX['senal'].iloc[-1]
    
    print(f"\n🎲 SEÑAL ACTUAL:")
    if ultima_senal == 1:
        print(f"   🟢 COMPRA - Considerar entrada LONG en {symbol}")
        if 'smart_trailing_stop' in df_with_exit.columns:
            ultimo_sl = df_with_exit['smart_trailing_stop'].iloc[-1]
            if not pd.isna(ultimo_sl):
                print(f"   Stop Loss sugerido: ${ultimo_sl:.2f}")
    elif ultima_senal == -1:
        print(f"   🔴 VENTA - Considerar salida o no entrar en {symbol}")
    else:
        print(f"   ⚪ NEUTRAL - Esperar mejor oportunidad en {symbol}")


def _mostrar_resultados_optimizacion(resultados, symbol):
    """Muestra tabla de resultados de optimización"""
    if not resultados:
        print("\n⚠️  No se obtuvieron resultados válidos")
        return
    
    print(f"\n{'='*70}")
    print(f"RESULTADOS DE OPTIMIZACIÓN - {symbol}")
    print(f"{'='*70}")
    print(f"{'Config':<15} {'Trades':<10} {'Win Rate':<12} {'P.Factor':<12} {'P&L':<12}")
    print("-" * 70)
    
    for r in resultados:
        print(f"{r['nombre']:<15} {r['trades']:<10} {r['win_rate']:>10.1f}% "
              f"{r['profit_factor']:>11.2f} ${r['pnl_total']:>10.2f}")
    
    # Mejor configuración
    mejor = max(resultados, key=lambda x: x['profit_factor'])
    print(f"\n🏆 Mejor: {mejor['nombre']}")
    print(f"   Profit Factor: {mejor['profit_factor']:.2f}")
    print(f"   Win Rate: {mejor['win_rate']:.1f}%")


def _mostrar_estado_posicion(df_with_exit, symbol):
    """Muestra el estado actual de una posición"""
    if 'smart_trailing_stop' not in df_with_exit.columns:
        print("\n⚠️  No hay información de trailing stop")
        return
    
    ultimo_sl = df_with_exit['smart_trailing_stop'].iloc[-1]
    ultimo_precio = df_with_exit['close'].iloc[-1]
    ultimo_max = df_with_exit['smart_max_price'].iloc[-1]
    
    if pd.isna(ultimo_sl):
        print(f"\n⚠️  No hay posición activa en {symbol}")
        return
    
    print(f"\n{'='*60}")
    print(f"📍 ESTADO DE POSICIÓN - {symbol}")
    print(f"{'='*60}")
    print(f"💰 Precio actual: ${ultimo_precio:.2f}")
    print(f"📈 Máximo alcanzado: ${ultimo_max:.2f}")
    print(f"🛡️  Stop Loss: ${ultimo_sl:.2f}")
    print(f"📊 Distancia al SL: ${(ultimo_precio - ultimo_sl):.2f} "
          f"({((ultimo_precio/ultimo_sl - 1)*100):.2f}%)")
    
    if df_with_exit['smart_break_even'].iloc[-1]:
        print(f"✅ Break-even ACTIVADO")
    else:
        print(f"⏳ Break-even pendiente")


def _mostrar_tabla_comparacion(resultados):
    """Muestra tabla comparativa de activos"""
    print(f"\n{'='*70}")
    print("COMPARACIÓN DE ACTIVOS")
    print(f"{'='*70}")
    print(f"{'Símbolo':<12} {'Precio':<15} {'Tendencia':<12} {'Volatil':<10} {'Cambio':<10}")
    print("-" * 70)
    
    for r in resultados:
        print(f"{r['symbol']:<12} ${r['precio']:<14.2f} {r['tendencia']:<12} "
              f"{r['volatilidad']:<9.2f}% {r['cambio']:+9.2f}%")


# ========================================
# CLI Y MENÚ PRINCIPAL
# ========================================

def parse_args():
    """Parser de argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Sistema de Trading con Smart Exit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py                                    # Menú interactivo
  python main.py --ejemplo 1 --symbol ETHUSDT      # Ejecutar ejemplo 1
  python main.py -s BNBUSDT -i 15m -l 1000         # Parámetros personalizados
        """
    )
    
    parser.add_argument('--symbol', '-s', type=str, default='BTCUSDT',
                       help='Par de trading (default: BTCUSDT)')
    parser.add_argument('--interval', '-i', type=str, default='1h',
                       help='Intervalo (default: 1h)')
    parser.add_argument('--limit', '-l', type=int, default=500,
                       help='Cantidad de velas (default: 500)')
    parser.add_argument('--ejemplo', '-e', type=int, choices=range(7),
                       help='Ejecutar ejemplo específico (0-6)')
    
    return parser.parse_args()


def menu_interactivo(args):
    """Menú interactivo para seleccionar ejemplos"""
    print("\n" + "="*60)
    print("🚀 SISTEMA DE TRADING CON SMART EXIT")
    print("="*60)
    print(f"\nConfiguración: {args.symbol} | {args.interval} | {args.limit} velas")
    print("\nEJEMPLOS DISPONIBLES:")
    print("\n0. Análisis básico (sin Smart Exit)")
    print("1. Análisis con Smart Exit ⭐")
    print("2. Optimización de parámetros")
    print("3. Monitoreo en tiempo real")
    print("4. Multi-timeframe")
    print("5. Comparación de activos")
    print("\n9. Salir")
    
    try:
        opcion = input("\nSelecciona una opción: ").strip()
        
        if opcion == '0':
            ejemplo_basico(args.symbol, args.interval, args.limit)
        elif opcion == '1':
            ejemplo_con_smart_exit(args.symbol, args.interval, args.limit)
        elif opcion == '2':
            ejemplo_optimizacion(args.symbol, args.interval, args.limit)
        elif opcion == '3':
            ejemplo_monitoreo(args.symbol, '5m', 100)
        elif opcion == '4':
            ejemplo_multi_timeframe(args.symbol)
        elif opcion == '5':
            ejemplo_comparacion()
        elif opcion == '9':
            print("\n👋 ¡Hasta luego!")
            return
        else:
            print("\n❌ Opción inválida")
            
    except KeyboardInterrupt:
        print("\n\n👋 Programa interrumpido")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")


def main():
    """Función principal"""
    args = parse_args()
    
    # Si se especificó un ejemplo, ejecutarlo directamente
    if args.ejemplo is not None:
        ejemplos = {
            0: ejemplo_basico,
            1: ejemplo_con_smart_exit,
            2: ejemplo_optimizacion,
            3: ejemplo_monitoreo,
            4: ejemplo_multi_timeframe,
            5: ejemplo_comparacion,
        }
        
        if args.ejemplo in [0, 1, 2]:
            ejemplos[args.ejemplo](args.symbol, args.interval, args.limit)
        elif args.ejemplo == 3:
            ejemplos[args.ejemplo](args.symbol, '5m', 100)
        else:
            ejemplos[args.ejemplo](args.symbol)
    else:
        # Modo interactivo
        menu_interactivo(args)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Modo CLI
        main()
    else:
        # Ejecución por defecto: Smart Exit
        ejemplo_con_smart_exit(
            symbol='BTCUSDT',
            interval='1h',
            limit=500
        )