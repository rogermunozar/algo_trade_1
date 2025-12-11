# main.py
"""
Script principal mejorado con sistema de Smart Exit y métricas avanzadas
Soporta parámetros desde línea de comandos
"""
from clients.binance_client import BinanceClient, load_secrets
from analysis.indicators import calcular_indicadores
from analysis.risk_management import apply_smart_exit, calcular_metricas_trailing
from visualization.chart_plotter import graficar_analisis, graficar_comparacion, recalc
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
    Ejemplo básico sin Smart Exit (original)
    
    Args:
        symbol: Par de trading (ej: 'BTCUSDT', 'ETHUSDT')
        interval: Intervalo temporal (ej: '1h', '15m', '1d')
        limit: Cantidad de velas
    """
    
    print("\n" + "="*60)
    print(f"EJEMPLO BÁSICO - Análisis de {symbol}")
    print("="*60)
    
    # Inicializar cliente
    binance = BinanceClient()
    
    # Obtener datos
    print(f"\n📊 Obteniendo {limit} velas de {symbol} ({interval})...")
    df = binance.history(symbol, interval, limit)
    
    # Calcular indicadores
    fecha = df.index[-1].strftime('%Y-%m-%d')
    comision = binance.calcular_comision(symbol)
    datos = calcular_indicadores(symbol, fecha, interval, df, comision)
    
    # Mostrar resultados en consola
    print(f"\n{'='*50}")
    print(f"ANÁLISIS DE {symbol}")
    print(f"{'='*50}")
    print(f"Precio actual: ${datos['precio_actual']:.2f}")
    print(f"Tendencia: {datos['tendencia']}")
    print(f"Volatilidad: {datos['volatilidad']:.2f}%")
    print(f"Cambio: {datos['cambio_porcentual']:+.2f}%")
    print(f"\nResistencias: {len(datos['resistance_levels'])} detectadas")
    if datos['resistance_levels']:
        print(f"  Más cercana: ${max(datos['resistance_levels']):.2f}")
    print(f"Soportes: {len(datos['support_levels'])} detectados")
    if datos['support_levels']:
        print(f"  Más cercano: ${max(datos['support_levels']):.2f}")
    
    print(f"\nPara posición LONG:")
    print(f"  Stop Loss: ${datos['sl_long']:.2f}")
    print(f"  Take Profit: ${datos['tp_long']:.2f}")
    print(f"  Risk/Reward: {(datos['tp_long']-datos['precio_actual'])/(datos['precio_actual']-datos['sl_long']):.2f}")
    
    print(f"\nPara posición SHORT:")
    print(f"  Stop Loss: ${datos['sl_short']:.2f}")
    print(f"  Take Profit: ${datos['tp_short']:.2f}")
    
    # Mostrar patrones detectados
    if datos['patrones_velas']:
        print(f"\nPatrones de velas detectados:")
        for patron, indices in datos['patrones_velas'].items():
            if indices:
                print(f"  {patron}: {len(indices)} ocurrencias")
    
    # Mostrar divergencias
    if datos['divergencias']['alcistas']:
        print(f"\nDivergencias alcistas RSI: {len(datos['divergencias']['alcistas'])}")
    if datos['divergencias']['bajistas']:
        print(f"Divergencias bajistas RSI: {len(datos['divergencias']['bajistas'])}")
    
    # Graficar
    graficar_analisis(datos, mostrar_avanzado=True)


def ejemplo_basico_con_smart_exit(symbol='BTCUSDT', interval='1h', limit=500):
    """
    Ejemplo básico con Smart Exit integrado
    
    Args:
        symbol: Par de trading (ej: 'BTCUSDT', 'ETHUSDT')
        interval: Intervalo temporal (ej: '1h', '15m', '1d')
        limit: Cantidad de velas
    """
    
    print("\n" + "="*60)
    print(f"EJEMPLO CON SMART EXIT - Análisis de {symbol}")
    print("="*60)
    
    # Inicializar cliente
    binance = BinanceClient()
    
    # Obtener datos
    print(f"\n📊 Obteniendo {limit} velas de {symbol} ({interval})...")
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
        min_move_to_update=0.5,
        swing_lookback=5,
        break_even_pct=0.9,
        break_even_buffer_atr_mult=0.2,
        noise_filter=True,
        tp_multiplier=2.0,
        log_trades=True
    )
    
    # Calcular métricas
    metricas = calcular_metricas_trailing(df_with_exit)
    
    # Mostrar resultados básicos
    print(f"\n{'='*60}")
    print(f"ANÁLISIS DE {symbol}")
    print(f"{'='*60}")
    print(f"Precio actual: ${datos['precio_actual']:.2f}")
    print(f"Tendencia: {datos['tendencia']}")
    print(f"Volatilidad: {datos['volatilidad']:.2f}%")
    print(f"Cambio: {datos['cambio_porcentual']:+.2f}%")
    print(f"ATR actual: ${datos['ultimo_atr']:.2f}")
    
    print(f"\n📍 NIVELES CLAVE:")
    print(f"  Resistencias detectadas: {len(datos['resistance_levels'])}")
    if datos['resistance_levels']:
        print(f"    Más cercana: ${max(datos['resistance_levels']):.2f}")
    print(f"  Soportes detectados: {len(datos['support_levels'])}")
    if datos['support_levels']:
        print(f"    Más cercano: ${max(datos['support_levels']):.2f}")
    
    print(f"\n💼 GESTIÓN DE RIESGO (ESTÁTICA):")
    print(f"  Para posición LONG:")
    print(f"    Entry: ${datos['precio_actual']:.2f}")
    print(f"    Stop Loss: ${datos['sl_long']:.2f} ({((datos['sl_long']/datos['precio_actual']-1)*100):.2f}%)")
    print(f"    Take Profit: ${datos['tp_long']:.2f} ({((datos['tp_long']/datos['precio_actual']-1)*100):.2f}%)")
    
    # Mostrar métricas de Smart Exit
    if metricas and metricas.get('total_trades', 0) > 0:
        print(f"\n{'='*60}")
        print(f"MÉTRICAS DEL SMART EXIT (Backtest histórico)")
        print(f"{'='*60}")
        print(f"📊 Trades totales: {metricas['total_trades']}")
        print(f"   ├─ Exits por Stop: {metricas['exits_por_stop']}")
        print(f"   └─ Exits por TP: {metricas['exits_por_tp']}")
        
        print(f"\n💰 Performance:")
        print(f"   ├─ Ganadores: {metricas['winning_trades']} ({metricas['win_rate']:.1f}%)")
        print(f"   ├─ Perdedores: {metricas['losing_trades']}")
        print(f"   ├─ Win Rate: {metricas['win_rate']:.1f}%")
        print(f"   └─ Profit Factor: {metricas['profit_factor']:.2f}")
        
        print(f"\n📈 P&L:")
        print(f"   ├─ Total: ${metricas['pnl_total']:+.2f}")
        print(f"   ├─ Promedio: ${metricas['pnl_medio']:+.2f}")
        print(f"   ├─ Avg Win: ${metricas['avg_win']:+.2f}")
        print(f"   └─ Avg Loss: ${metricas['avg_loss']:+.2f}")
        
        # Análisis del sistema
        print(f"\n🎯 EVALUACIÓN DEL SISTEMA:")
        if metricas['win_rate'] >= 60:
            print(f"   ✅ Win Rate excelente (>60%)")
        elif metricas['win_rate'] >= 50:
            print(f"   ⚠️  Win Rate aceptable (50-60%)")
        else:
            print(f"   ❌ Win Rate bajo (<50%) - Revisar estrategia")
        
        if metricas['profit_factor'] >= 2.0:
            print(f"   ✅ Profit Factor excelente (>2.0)")
        elif metricas['profit_factor'] >= 1.5:
            print(f"   ⚠️  Profit Factor aceptable (1.5-2.0)")
        else:
            print(f"   ❌ Profit Factor bajo (<1.5) - Revisar gestión de riesgo")
    else:
        print(f"\n⚠️  No se detectaron trades en el período analizado")
    
    # Información sobre última señal
    ultima_senal = dfX['senal'].iloc[-1]
    print(f"\n🎲 SEÑAL ACTUAL:")
    if ultima_senal == 1:
        print(f"   🟢 COMPRA - Considerar entrada LONG en {symbol}")
        if 'smart_trailing_stop' in df_with_exit.columns:
            ultimo_sl = df_with_exit['smart_trailing_stop'].iloc[-1]
            if not pd.isna(ultimo_sl):
                print(f"   Stop Loss sugerido: ${ultimo_sl:.2f}")
    elif ultima_senal == -1:
        print(f"   🔴 VENTA - Considerar salir o no entrar en {symbol}")
    else:
        print(f"   ⚪ NEUTRAL - Esperar mejor oportunidad en {symbol}")
    
    print(f"\n{'='*60}\n")
    
    # Graficar
    print("📊 Generando gráfico...")
    graficar_analisis(datos, mostrar_avanzado=True)


def ejemplo_optimizacion_parametros(symbol='ETHUSDT', interval='1h', limit=1000):
    """
    Ejemplo: probar diferentes configuraciones de Smart Exit
    
    Args:
        symbol: Par de trading
        interval: Intervalo temporal
        limit: Cantidad de velas
    """
    
    print("\n" + "="*60)
    print(f"OPTIMIZACIÓN DE PARÁMETROS - Smart Exit para {symbol}")
    print("="*60)
    
    binance = BinanceClient()
    
    # Obtener datos
    print(f"\n📊 Obteniendo {limit} velas de {symbol} ({interval})...")
    df = binance.history(symbol, interval, limit)
    fecha = df.index[-1].strftime('%Y-%m-%d')
    comision = binance.calcular_comision(symbol)
    datos = calcular_indicadores(symbol, fecha, interval, df, comision)
    dfX = datos['dfX']
    
    # Probar diferentes configuraciones
    configuraciones = [
        {'nombre': 'Conservador', 'atr_mult': 2.0, 'tp_mult': 3.0, 'break_even_pct': 0.8},
        {'nombre': 'Balanceado', 'atr_mult': 1.5, 'tp_mult': 2.0, 'break_even_pct': 0.9},
        {'nombre': 'Agresivo', 'atr_mult': 1.0, 'tp_mult': 1.5, 'break_even_pct': 0.95},
    ]
    
    print(f"\nProbando {len(configuraciones)} configuraciones en {symbol}...\n")
    
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
        
        if metricas.get('total_trades', 0) > 0:
            resultados.append({
                'nombre': config['nombre'],
                'trades': metricas['total_trades'],
                'win_rate': metricas['win_rate'],
                'profit_factor': metricas['profit_factor'],
                'pnl_total': metricas['pnl_total']
            })
    
    # Mostrar resultados comparativos
    print(f"{'Configuración':<15} {'Trades':<10} {'Win Rate':<12} {'P.Factor':<12} {'P&L Total':<12}")
    print("-" * 70)
    
    for r in resultados:
        print(f"{r['nombre']:<15} {r['trades']:<10} {r['win_rate']:>10.1f}% {r['profit_factor']:>11.2f} ${r['pnl_total']:>10.2f}")
    
    # Determinar mejor configuración
    if resultados:
        mejor = max(resultados, key=lambda x: x['profit_factor'])
        print(f"\n🏆 Mejor configuración para {symbol}: {mejor['nombre']}")
        print(f"   Profit Factor: {mejor['profit_factor']:.2f}")
        print(f"   Win Rate: {mejor['win_rate']:.1f}%")


def ejemplo_monitoreo_tiempo_real(symbol='BNBUSDT', interval='5m', limit=100):
    """
    Ejemplo: simular monitoreo de una posición abierta
    
    Args:
        symbol: Par de trading
        interval: Intervalo temporal
        limit: Cantidad de velas
    """
    
    print("\n" + "="*60)
    print(f"SIMULACIÓN DE MONITOREO EN TIEMPO REAL - {symbol}")
    print("="*60)
    
    binance = BinanceClient()
    
    # Obtener datos recientes
    print(f"\n📊 Obteniendo {limit} velas de {symbol} ({interval})...")
    df = binance.history(symbol, interval, limit)
    fecha = df.index[-1].strftime('%Y-%m-%d')
    comision = binance.calcular_comision(symbol)
    datos = calcular_indicadores(symbol, fecha, interval, df, comision)
    dfX = datos['dfX']
    
    # Aplicar smart exit
    df_with_exit = apply_smart_exit(
        dfX,
        entry_col='senal',
        log_trades=True
    )
    
    # Simular posición actual
    if 'smart_trailing_stop' in df_with_exit.columns:
        ultimo_sl = df_with_exit['smart_trailing_stop'].iloc[-1]
        ultimo_precio = df_with_exit['close'].iloc[-1]
        ultimo_max = df_with_exit['smart_max_price'].iloc[-1]
        
        if not pd.isna(ultimo_sl):
            print(f"\n{'='*60}")
            print(f"ESTADO DE LA POSICIÓN ACTUAL - {symbol}")
            print(f"{'='*60}")
            print(f"💰 Precio actual: ${ultimo_precio:.2f}")
            print(f"📈 Máximo alcanzado: ${ultimo_max:.2f}")
            print(f"🛡️  Stop Loss actual: ${ultimo_sl:.2f}")
            print(f"📊 Distancia al stop: ${(ultimo_precio - ultimo_sl):.2f} ({((ultimo_precio/ultimo_sl - 1)*100):.2f}%)")
            
            if df_with_exit['smart_break_even'].iloc[-1]:
                print(f"✅ Break-even ACTIVADO - Riesgo eliminado")
            else:
                print(f"⏳ Break-even pendiente")


def main():
    """Menú principal con soporte de argumentos de línea de comandos"""
    
    # Parser de argumentos
    parser = argparse.ArgumentParser(
        description='Sistema de Trading con Smart Exit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py                              # Menú interactivo
  python main.py --symbol ETHUSDT             # Analizar ETHUSDT
  python main.py --symbol BNBUSDT --interval 15m --limit 1000
  python main.py --ejemplo 1 --symbol SOLUSDT # Ejecutar ejemplo específico
        """
    )
    
    parser.add_argument('--symbol', '-s', type=str, default='BTCUSDT',
                       help='Par de trading (default: BTCUSDT)')
    parser.add_argument('--interval', '-i', type=str, default='1h',
                       help='Intervalo temporal (default: 1h)')
    parser.add_argument('--limit', '-l', type=int, default=500,
                       help='Cantidad de velas (default: 500)')
    parser.add_argument('--ejemplo', '-e', type=int, choices=[0,1,2,3,4],
                       help='Ejecutar ejemplo específico (0-4)')
    
    args = parser.parse_args()
    
    # Si se especificó un ejemplo, ejecutarlo directamente
    if args.ejemplo is not None:
        if args.ejemplo == 0:
            print(f"\n✨ Ejecutando ejemplo básico con {args.symbol}...")
            ejemplo_basico(args.symbol, args.interval, args.limit)
        elif args.ejemplo == 1:
            print(f"\n✨ Ejecutando Smart Exit con {args.symbol}...")
            ejemplo_basico_con_smart_exit(args.symbol, args.interval, args.limit)
        elif args.ejemplo == 2:
            print(f"\n✨ Ejecutando optimización con {args.symbol}...")
            ejemplo_optimizacion_parametros(args.symbol, args.interval, args.limit)
        elif args.ejemplo == 3:
            print(f"\n✨ Ejecutando monitoreo con {args.symbol}...")
            ejemplo_monitoreo_tiempo_real(args.symbol, args.interval, args.limit)
        return
    
    # Menú interactivo
    print("\n" + "="*60)
    print("SISTEMA DE TRADING CON SMART EXIT")
    print("="*60)
    print(f"\nSymbol configurado: {args.symbol}")
    print(f"Interval: {args.interval}")
    print(f"Limit: {args.limit}")
    print("\n1. Análisis básico con Smart Exit")
    print("2. Optimización de parámetros")
    print("3. Simulación de monitoreo en tiempo real")
    print("4. Ejemplo original (sin Smart Exit)")
    print("\n0. Salir")
    
    try:
        opcion = input("\nOpción: ").strip()
        
        if opcion == '1':
            ejemplo_basico_con_smart_exit(args.symbol, args.interval, args.limit)
        elif opcion == '2':
            ejemplo_optimizacion_parametros(args.symbol, args.interval, args.limit)
        elif opcion == '3':
            ejemplo_monitoreo_tiempo_real(args.symbol, args.interval, args.limit)
        elif opcion == '4':
            ejemplo_basico(args.symbol, args.interval, args.limit)
        elif opcion == '0':
            print("\n¡Hasta luego!")
            return
        else:
            print("\n❌ Opción inválida")
            
    except KeyboardInterrupt:
        print("\n\n¡Programa interrumpido!")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    # Si hay argumentos de línea de comandos, usar main()
    if len(sys.argv) > 1:
        main()
    else:
        # Ejecución por defecto con Smart Exit
        # Puedes cambiar el symbol, interval y limit aquí
        ejemplo_basico_con_smart_exit(
            symbol='BTCUSDT',  # Cambiar aquí el par
            interval='1h',      # Cambiar aquí el intervalo
            limit=500           # Cambiar aquí la cantidad de velas
        )
        
        # O usar el menú interactivo
        # main()
    """Ejemplo básico sin Smart Exit (original)"""
    
    print("\n" + "="*60)
    print("EJEMPLO BÁSICO - Análisis de BTCUSDT")
    print("="*60)
    
    # Inicializar cliente
    binance = BinanceClient()
    
    # Obtener datos
    symbol = 'BTCUSDT'
    df = binance.history(symbol, '1h', 500)
    
    # Calcular indicadores
    fecha = df.index[-1].strftime('%Y-%m-%d')
    comision = binance.calcular_comision(symbol)
    datos = calcular_indicadores(symbol, fecha, '1h', df, comision)
    
    # Mostrar resultados en consola
    print(f"\n{'='*50}")
    print(f"ANÁLISIS DE {symbol}")
    print(f"{'='*50}")
    print(f"Precio actual: ${datos['precio_actual']:.2f}")
    print(f"Tendencia: {datos['tendencia']}")
    print(f"Volatilidad: {datos['volatilidad']:.2f}%")
    print(f"Cambio: {datos['cambio_porcentual']:+.2f}%")
    print(f"\nResistencias: {len(datos['resistance_levels'])} detectadas")
    if datos['resistance_levels']:
        print(f"  Más cercana: ${max(datos['resistance_levels']):.2f}")
    print(f"Soportes: {len(datos['support_levels'])} detectados")
    if datos['support_levels']:
        print(f"  Más cercano: ${max(datos['support_levels']):.2f}")
    
    print(f"\nPara posición LONG:")
    print(f"  Stop Loss: ${datos['sl_long']:.2f}")
    print(f"  Take Profit: ${datos['tp_long']:.2f}")
    print(f"  Risk/Reward: {(datos['tp_long']-datos['precio_actual'])/(datos['precio_actual']-datos['sl_long']):.2f}")
    
    print(f"\nPara posición SHORT:")
    print(f"  Stop Loss: ${datos['sl_short']:.2f}")
    print(f"  Take Profit: ${datos['tp_short']:.2f}")
    
    # Mostrar patrones detectados
    if datos['patrones_velas']:
        print(f"\nPatrones de velas detectados:")
        for patron, indices in datos['patrones_velas'].items():
            if indices:
                print(f"  {patron}: {len(indices)} ocurrencias")
    
    # Mostrar divergencias
    if datos['divergencias']['alcistas']:
        print(f"\nDivergencias alcistas RSI: {len(datos['divergencias']['alcistas'])}")
    if datos['divergencias']['bajistas']:
        print(f"Divergencias bajistas RSI: {len(datos['divergencias']['bajistas'])}")
    
    # Graficar
    graficar_analisis(datos, mostrar_avanzado=True)


def ejemplo_basico_con_smart_exit():
    """Ejemplo básico con Smart Exit integrado"""
    
    print("\n" + "="*60)
    print("EJEMPLO CON SMART EXIT - Análisis completo")
    print("="*60)
    
    # Inicializar cliente
    binance = BinanceClient()
    
    # Configuración
    symbol = 'BTCUSDT'
    interval = '1h'
    limit = 500
    
    # Obtener datos
    print(f"\n📊 Obteniendo datos de {symbol} ({interval})...")
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
        min_move_to_update=0.5,
        swing_lookback=5,
        break_even_pct=0.9,
        break_even_buffer_atr_mult=0.2,
        noise_filter=True,
        tp_multiplier=2.0,
        log_trades=True
    )
    
    # Calcular métricas
    metricas = calcular_metricas_trailing(df_with_exit)
    
    # Mostrar resultados básicos
    print(f"\n{'='*60}")
    print(f"ANÁLISIS DE {symbol}")
    print(f"{'='*60}")
    print(f"Precio actual: ${datos['precio_actual']:.2f}")
    print(f"Tendencia: {datos['tendencia']}")
    print(f"Volatilidad: {datos['volatilidad']:.2f}%")
    print(f"Cambio: {datos['cambio_porcentual']:+.2f}%")
    print(f"ATR actual: ${datos['ultimo_atr']:.2f}")
    
    print(f"\n📍 NIVELES CLAVE:")
    print(f"  Resistencias detectadas: {len(datos['resistance_levels'])}")
    if datos['resistance_levels']:
        print(f"    Más cercana: ${max(datos['resistance_levels']):.2f}")
    print(f"  Soportes detectados: {len(datos['support_levels'])}")
    if datos['support_levels']:
        print(f"    Más cercano: ${max(datos['support_levels']):.2f}")
    
    print(f"\n💼 GESTIÓN DE RIESGO (ESTÁTICA):")
    print(f"  Para posición LONG:")
    print(f"    Entry: ${datos['precio_actual']:.2f}")
    print(f"    Stop Loss: ${datos['sl_long']:.2f} ({((datos['sl_long']/datos['precio_actual']-1)*100):.2f}%)")
    print(f"    Take Profit: ${datos['tp_long']:.2f} ({((datos['tp_long']/datos['precio_actual']-1)*100):.2f}%)")
    
    # Mostrar métricas de Smart Exit
    if metricas and metricas.get('total_trades', 0) > 0:
        print(f"\n{'='*60}")
        print(f"MÉTRICAS DEL SMART EXIT (Backtest histórico)")
        print(f"{'='*60}")
        print(f"📊 Trades totales: {metricas['total_trades']}")
        print(f"   ├─ Exits por Stop: {metricas['exits_por_stop']}")
        print(f"   └─ Exits por TP: {metricas['exits_por_tp']}")
        
        print(f"\n💰 Performance:")
        print(f"   ├─ Ganadores: {metricas['winning_trades']} ({metricas['win_rate']:.1f}%)")
        print(f"   ├─ Perdedores: {metricas['losing_trades']}")
        print(f"   ├─ Win Rate: {metricas['win_rate']:.1f}%")
        print(f"   └─ Profit Factor: {metricas['profit_factor']:.2f}")
        
        print(f"\n📈 P&L:")
        print(f"   ├─ Total: ${metricas['pnl_total']:+.2f}")
        print(f"   ├─ Promedio: ${metricas['pnl_medio']:+.2f}")
        print(f"   ├─ Avg Win: ${metricas['avg_win']:+.2f}")
        print(f"   └─ Avg Loss: ${metricas['avg_loss']:+.2f}")
        
        # Análisis del sistema
        print(f"\n🎯 EVALUACIÓN DEL SISTEMA:")
        if metricas['win_rate'] >= 60:
            print(f"   ✅ Win Rate excelente (>60%)")
        elif metricas['win_rate'] >= 50:
            print(f"   ⚠️  Win Rate aceptable (50-60%)")
        else:
            print(f"   ❌ Win Rate bajo (<50%) - Revisar estrategia")
        
        if metricas['profit_factor'] >= 2.0:
            print(f"   ✅ Profit Factor excelente (>2.0)")
        elif metricas['profit_factor'] >= 1.5:
            print(f"   ⚠️  Profit Factor aceptable (1.5-2.0)")
        else:
            print(f"   ❌ Profit Factor bajo (<1.5) - Revisar gestión de riesgo")
    else:
        print(f"\n⚠️  No se detectaron trades en el período analizado")
    
    # Información sobre última señal
    ultima_senal = dfX['senal'].iloc[-1]
    print(f"\n🎲 SEÑAL ACTUAL:")
    if ultima_senal == 1:
        print(f"   🟢 COMPRA - Considerar entrada LONG")
        if 'smart_trailing_stop' in df_with_exit.columns:
            ultimo_sl = df_with_exit['smart_trailing_stop'].iloc[-1]
            if not pd.isna(ultimo_sl):
                print(f"   Stop Loss sugerido: ${ultimo_sl:.2f}")
    elif ultima_senal == -1:
        print(f"   🔴 VENTA - Considerar salir o no entrar")
    else:
        print(f"   ⚪ NEUTRAL - Esperar mejor oportunidad")
    
    print(f"\n{'='*60}\n")
    
    # Graficar
    print("📊 Generando gráfico...")
    graficar_analisis(datos, mostrar_avanzado=True)


def ejemplo_optimizacion_parametros():
    """Ejemplo: probar diferentes configuraciones de Smart Exit"""
    
    print("\n" + "="*60)
    print("OPTIMIZACIÓN DE PARÁMETROS - Smart Exit")
    print("="*60)
    
    binance = BinanceClient()
    symbol = 'ETHUSDT'
    
    # Obtener datos
    df = binance.history(symbol, '1h', 1000)
    fecha = df.index[-1].strftime('%Y-%m-%d')
    comision = binance.calcular_comision(symbol)
    datos = calcular_indicadores(symbol, fecha, '1h', df, comision)
    dfX = datos['dfX']
    
    # Probar diferentes configuraciones
    configuraciones = [
        {'nombre': 'Conservador', 'atr_mult': 2.0, 'tp_mult': 3.0, 'break_even_pct': 0.8},
        {'nombre': 'Balanceado', 'atr_mult': 1.5, 'tp_mult': 2.0, 'break_even_pct': 0.9},
        {'nombre': 'Agresivo', 'atr_mult': 1.0, 'tp_mult': 1.5, 'break_even_pct': 0.95},
    ]
    
    print(f"\nProbando {len(configuraciones)} configuraciones en {symbol}...\n")
    
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
        
        if metricas.get('total_trades', 0) > 0:
            resultados.append({
                'nombre': config['nombre'],
                'trades': metricas['total_trades'],
                'win_rate': metricas['win_rate'],
                'profit_factor': metricas['profit_factor'],
                'pnl_total': metricas['pnl_total']
            })
    
    # Mostrar resultados comparativos
    print(f"{'Configuración':<15} {'Trades':<10} {'Win Rate':<12} {'P.Factor':<12} {'P&L Total':<12}")
    print("-" * 70)
    
    for r in resultados:
        print(f"{r['nombre']:<15} {r['trades']:<10} {r['win_rate']:>10.1f}% {r['profit_factor']:>11.2f} ${r['pnl_total']:>10.2f}")
    
    # Determinar mejor configuración
    if resultados:
        mejor = max(resultados, key=lambda x: x['profit_factor'])
        print(f"\n🏆 Mejor configuración: {mejor['nombre']}")
        print(f"   Profit Factor: {mejor['profit_factor']:.2f}")
        print(f"   Win Rate: {mejor['win_rate']:.1f}%")


def ejemplo_monitoreo_tiempo_real():
    """Ejemplo: simular monitoreo de una posición abierta"""
    
    print("\n" + "="*60)
    print("SIMULACIÓN DE MONITOREO EN TIEMPO REAL")
    print("="*60)
    
    binance = BinanceClient()
    symbol = 'BNBUSDT'
    
    # Obtener datos recientes
    df = binance.history(symbol, '5m', 100)
    fecha = df.index[-1].strftime('%Y-%m-%d')
    comision = binance.calcular_comision(symbol)
    datos = calcular_indicadores(symbol, fecha, '5m', df, comision)
    dfX = datos['dfX']
    
    # Aplicar smart exit
    df_with_exit = apply_smart_exit(
        dfX,
        entry_col='senal',
        log_trades=True
    )
    
    # Simular posición actual
    if 'smart_trailing_stop' in df_with_exit.columns:
        ultimo_sl = df_with_exit['smart_trailing_stop'].iloc[-1]
        ultimo_precio = df_with_exit['close'].iloc[-1]
        ultimo_max = df_with_exit['smart_max_price'].iloc[-1]
        
        if not pd.isna(ultimo_sl):
            print(f"\n{'='*60}")
            print(f"ESTADO DE LA POSICIÓN ACTUAL")
            print(f"{'='*60}")
            print(f"💰 Precio actual: ${ultimo_precio:.2f}")
            print(f"📈 Máximo alcanzado: ${ultimo_max:.2f}")
            print(f"🛡️  Stop Loss actual: ${ultimo_sl:.2f}")
            print(f"📊 Distancia al stop: ${(ultimo_precio - ultimo_sl):.2f} ({((ultimo_precio/ultimo_sl - 1)*100):.2f}%)")
            
            if df_with_exit['smart_break_even'].iloc[-1]:
                print(f"✅ Break-even ACTIVADO - Riesgo eliminado")
            else:
                print(f"⏳ Break-even pendiente")


def ejemplo_monitoreo_tiempo_real():
    """Ejemplo: simular monitoreo de una posición abierta"""
    
    print("\n" + "="*60)
    print("SIMULACIÓN DE MONITOREO EN TIEMPO REAL")
    print("="*60)
    
    binance = BinanceClient()
    symbol = 'BNBUSDT'
    
    # Obtener datos recientes
    df = binance.history(symbol, '5m', 100)
    fecha = df.index[-1].strftime('%Y-%m-%d')
    comision = binance.calcular_comision(symbol)
    datos = calcular_indicadores(symbol, fecha, '5m', df, comision)
    dfX = datos['dfX']
    
    # Aplicar smart exit
    df_with_exit = apply_smart_exit(
        dfX,
        entry_col='senal',
        log_trades=True
    )
    
    # Simular posición actual
    if 'smart_trailing_stop' in df_with_exit.columns:
        ultimo_sl = df_with_exit['smart_trailing_stop'].iloc[-1]
        ultimo_precio = df_with_exit['close'].iloc[-1]
        ultimo_max = df_with_exit['smart_max_price'].iloc[-1]
        
        if not pd.isna(ultimo_sl):
            print(f"\n{'='*60}")
            print(f"ESTADO DE LA POSICIÓN ACTUAL")
            print(f"{'='*60}")
            print(f"💰 Precio actual: ${ultimo_precio:.2f}")
            print(f"📈 Máximo alcanzado: ${ultimo_max:.2f}")
            print(f"🛡️  Stop Loss actual: ${ultimo_sl:.2f}")
            print(f"📊 Distancia al stop: ${(ultimo_precio - ultimo_sl):.2f} ({((ultimo_precio/ultimo_sl - 1)*100):.2f}%)")
            
            if df_with_exit['smart_break_even'].iloc[-1]:
                print(f"✅ Break-even ACTIVADO - Riesgo eliminado")
            else:
                print(f"⏳ Break-even pendiente")


def main():
    """Menú principal"""
    
    print("\n" + "="*60)
    print("SISTEMA DE TRADING CON SMART EXIT")
    print("="*60)
    print("\n1. Análisis básico con Smart Exit")
    print("2. Optimización de parámetros")
    print("3. Simulación de monitoreo en tiempo real")
    print("4. Ejemplo original (sin Smart Exit)")
    print("\n0. Salir")
    
    try:
        opcion = input("\nOpción: ").strip()
        
        if opcion == '1':
            ejemplo_basico_con_smart_exit()
        elif opcion == '2':
            ejemplo_optimizacion_parametros()
        elif opcion == '3':
            ejemplo_monitoreo_tiempo_real()
        elif opcion == '4':
            ejemplo_basico()
        elif opcion == '0':
            print("\n¡Hasta luego!")
            return
        else:
            print("\n❌ Opción inválida")
            
    except KeyboardInterrupt:
        print("\n\n¡Programa interrumpido!")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    # Ejecutar ejemplo con Smart Exit por defecto
    ejemplo_basico_con_smart_exit()
    
    # O descomentar para menú interactivo
    # main()
    """Ejemplo básico: análisis rápido de un activo"""
    
    print("\n" + "="*60)
    print("EJEMPLO BÁSICO - Análisis de BTCUSDT")
    print("="*60)
    
    # Inicializar cliente
    binance = BinanceClient()
    
    # Obtener datos
    symbol = 'BTCUSDT'
    df = binance.history(symbol, '1h', 500)
    
    # Calcular indicadores
    fecha = df.index[-1].strftime('%Y-%m-%d')
    comision = binance.calcular_comision(symbol)
    datos = calcular_indicadores(symbol, fecha, '1h', df, comision)
    
    # Mostrar resultados en consola
    print(f"\n{'='*50}")
    print(f"ANÁLISIS DE {symbol}")
    print(f"{'='*50}")
    print(f"Precio actual: ${datos['precio_actual']:.2f}")
    print(f"Tendencia: {datos['tendencia']}")
    print(f"Volatilidad: {datos['volatilidad']:.2f}%")
    print(f"Cambio: {datos['cambio_porcentual']:+.2f}%")
    print(f"\nResistencias: {len(datos['resistance_levels'])} detectadas")
    if datos['resistance_levels']:
        print(f"  Más cercana: ${max(datos['resistance_levels']):.2f}")
    print(f"Soportes: {len(datos['support_levels'])} detectados")
    if datos['support_levels']:
        print(f"  Más cercano: ${max(datos['support_levels']):.2f}")
    
    print(f"\nPara posición LONG:")
    print(f"  Stop Loss: ${datos['sl_long']:.2f}")
    print(f"  Take Profit: ${datos['tp_long']:.2f}")
    print(f"  Risk/Reward: {(datos['tp_long']-datos['precio_actual'])/(datos['precio_actual']-datos['sl_long']):.2f}")
    
    print(f"\nPara posición SHORT:")
    print(f"  Stop Loss: ${datos['sl_short']:.2f}")
    print(f"  Take Profit: ${datos['tp_short']:.2f}")
    
    # Mostrar patrones detectados
    if datos['patrones_velas']:
        print(f"\nPatrones de velas detectados:")
        for patron, indices in datos['patrones_velas'].items():
            if indices:
                print(f"  {patron}: {len(indices)} ocurrencias")
    
    # Mostrar divergencias
    if datos['divergencias']['alcistas']:
        print(f"\nDivergencias alcistas RSI: {len(datos['divergencias']['alcistas'])}")
    if datos['divergencias']['bajistas']:
        print(f"Divergencias bajistas RSI: {len(datos['divergencias']['bajistas'])}")
    
    # Graficar
    graficar_analisis(datos, mostrar_avanzado=True)


def ejemplo_multiple_timeframes():
    """Ejemplo: analizar el mismo activo en múltiples timeframes"""
    
    print("\n" + "="*60)
    print("EJEMPLO MULTI-TIMEFRAME - BTC en diferentes intervalos")
    print("="*60)
    
    binance = BinanceClient()
    symbol = 'BTCUSDT'
    timeframes = ['15m', '1h', '4h']
    
    for tf in timeframes:
        print(f"\n--- Analizando {symbol} en {tf} ---")
        
        df = binance.history(symbol, tf, 500)
        fecha = df.index[-1].strftime('%Y-%m-%d')
        comision = binance.calcular_comision(symbol)
        
        datos = calcular_indicadores(symbol, fecha, tf, df, comision)
        
        print(f"Tendencia {tf}: {datos['tendencia']}")
        print(f"Volatilidad {tf}: {datos['volatilidad']:.2f}%")
        
        # Graficar cada timeframe
        graficar_analisis(datos, mostrar_avanzado=False)


def ejemplo_comparacion_activos():
    """Ejemplo: comparar múltiples activos"""
    
    print("\n" + "="*60)
    print("EJEMPLO COMPARACIÓN - Múltiples activos")
    print("="*60)
    
    binance = BinanceClient()
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    interval = '1h'
    
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
    
    # Mostrar comparación
    print("\n" + "="*60)
    print("RESUMEN COMPARATIVO")
    print("="*60)
    print(f"{'Símbolo':<12} {'Precio':<12} {'Tendencia':<12} {'Volatilidad':<12} {'Cambio':<12}")
    print("-" * 60)
    
    for r in resultados:
        print(f"{r['symbol']:<12} ${r['precio']:<11.2f} {r['tendencia']:<12} "
              f"{r['volatilidad']:<11.2f}% {r['cambio']:+11.2f}%")
    
    # Graficar comparación
    graficar_comparacion(datos_list, symbols)


def ejemplo_con_credenciales():
    """Ejemplo: usar credenciales para ver balance y crear órdenes"""
    
    print("\n" + "="*60)
    print("EJEMPLO CON CREDENCIALES - Trading real")
    print("="*60)
    
    try:
        # Cargar credenciales desde variables de entorno
        api_key, api_secret = load_secrets()
        binance = BinanceClient(api_key, api_secret)
        
        print("✅ Credenciales cargadas correctamente")
        
        # Ver balance
        print("\n📊 Balance de cuenta:")
        balances = binance.get_account_balance()
        for balance in balances[:10]:  # Mostrar solo primeros 10
            if balance['total'] > 0:
                print(f"  {balance['asset']:<8} {balance['total']:>15.8f}")
        
        # Obtener precio actual
        symbol = 'BTCUSDT'
        precio = binance.get_current_price(symbol)
        print(f"\n💰 Precio actual de {symbol}: ${precio:,.2f}")
        
        # Ver estadísticas 24h
        stats = binance.get_ticker_24h(symbol)
        print(f"\n📈 Estadísticas 24h:")
        print(f"  Cambio: {stats['price_change_percent']:+.2f}%")
        print(f"  Máximo: ${stats['high']:,.2f}")
        print(f"  Mínimo: ${stats['low']:,.2f}")
        print(f"  Volumen: {stats['volume']:,.2f}")
        
        # Ver órdenes abiertas
        ordenes = binance.get_open_orders(symbol)
        print(f"\n📋 Órdenes abiertas para {symbol}: {len(ordenes)}")
        
        # ADVERTENCIA: No crear órdenes en este ejemplo
        print("\n⚠️  Para crear órdenes, descomentar el código en el script")
        
        # Ejemplo de cómo crear una orden (COMENTADO para seguridad):
        # orden = binance.create_order(
        #     symbol='BTCUSDT',
        #     side='BUY',
        #     order_type='LIMIT',
        #     quantity=0.001,
        #     price=precio * 0.98  # 2% por debajo del precio actual
        # )
        # print(f"Orden creada: {orden}")
        
    except ValueError as e:
        print(f"\n❌ {e}")
        print("\n💡 Configura las variables de entorno:")
        print("\n   Linux/Mac:")
        print("   export BINANCE_API_KEY='tu_key'")
        print("   export BINANCE_API_SECRET='tu_secret'")
        print("\n   Windows PowerShell:")
        print("   $env:BINANCE_API_KEY='tu_key'")
        print("   $env:BINANCE_API_SECRET='tu_secret'")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")


def ejemplo_analisis_profundo():
    """Ejemplo: análisis técnico completo con todos los detalles"""
    
    print("\n" + "="*60)
    print("EJEMPLO ANÁLISIS PROFUNDO")
    print("="*60)
    
    binance = BinanceClient()
    symbol = 'ETHUSDT'
    
    # Obtener más datos para análisis profundo
    df = binance.history(symbol, '1h', 1000)
    fecha = df.index[-1].strftime('%Y-%m-%d')
    comision = binance.calcular_comision(symbol)
    
    # Calcular todos los indicadores
    datos = calcular_indicadores(symbol, fecha, '1h', df, comision, display=1)
    
    # Análisis detallado
    print(f"\n{'='*60}")
    print(f"ANÁLISIS TÉCNICO COMPLETO - {symbol}")
    print(f"{'='*60}")
    
    # Información básica
    print(f"\n📊 INFORMACIÓN GENERAL:")
    print(f"  Precio actual: ${datos['precio_actual']:.2f}")
    print(f"  Tendencia: {datos['tendencia']}")
    print(f"  Volatilidad: {datos['volatilidad']:.2f}%")
    print(f"  Cambio período: {datos['cambio_porcentual']:+.2f}%")
    print(f"  ATR: ${datos['ultimo_atr']:.2f}")
    
    # Soportes y resistencias
    print(f"\n🎯 SOPORTES Y RESISTENCIAS:")
    print(f"  Resistencias detectadas: {len(datos['resistance_levels'])}")
    if datos['resistance_levels']:
        print(f"    Niveles: {[f'${x:.2f}' for x in sorted(datos['resistance_levels'][-5:])]}")
    print(f"  Soportes detectados: {len(datos['support_levels'])}")
    if datos['support_levels']:
        print(f"    Niveles: {[f'${x:.2f}' for x in sorted(datos['support_levels'][-5:])]}")
    
    # Niveles de Fibonacci
    print(f"\n📐 NIVELES DE FIBONACCI:")
    for nivel, precio in datos['niveles_fibonacci'].items():
        if 'nivel_' in nivel:
            porcentaje = nivel.split('_')[1]
            print(f"  {porcentaje}%: ${precio:.2f}")
    
    # Señales de trading
    dfX = datos['dfX']
    senales_compra = (dfX['senal'] == 1).sum()
    senales_venta = (dfX['senal'] == -1).sum()
    
    print(f"\n🎲 SEÑALES DE TRADING:")
    print(f"  Señales de COMPRA: {senales_compra}")
    print(f"  Señales de VENTA: {senales_venta}")
    print(f"  Última señal: ", end="")
    ultima_senal = dfX['senal'].iloc[-1]
    if ultima_senal == 1:
        print("🟢 COMPRA")
    elif ultima_senal == -1:
        print("🔴 VENTA")
    else:
        print("⚪ NEUTRAL")
    
    # Gestión de riesgo
    print(f"\n💼 GESTIÓN DE RIESGO:")
    print(f"  Para posición LONG:")
    print(f"    Entry: ${datos['precio_actual']:.2f}")
    print(f"    Stop Loss: ${datos['sl_long']:.2f} ({((datos['sl_long']/datos['precio_actual']-1)*100):.2f}%)")
    print(f"    Take Profit: ${datos['tp_long']:.2f} ({((datos['tp_long']/datos['precio_actual']-1)*100):.2f}%)")
    
    print(f"  Para posición SHORT:")
    print(f"    Entry: ${datos['precio_actual']:.2f}")
    print(f"    Stop Loss: ${datos['sl_short']:.2f} ({((datos['sl_short']/datos['precio_actual']-1)*100):.2f}%)")
    print(f"    Take Profit: ${datos['tp_short']:.2f} ({((datos['tp_short']/datos['precio_actual']-1)*100):.2f}%)")
    
    # Patrones detectados
    print(f"\n🕯️  PATRONES DE VELAS:")
    for patron, indices in datos['patrones_velas'].items():
        if indices:
            print(f"  {patron.replace('_', ' ').title()}: {len(indices)} detectados")
    
    # Divergencias
    print(f"\n📉 DIVERGENCIAS RSI:")
    print(f"  Alcistas: {len(datos['divergencias']['alcistas'])}")
    print(f"  Bajistas: {len(datos['divergencias']['bajistas'])}")
    
    # Graficar con todo el análisis avanzado
    graficar_analisis(datos, mostrar_avanzado=True, guardar=True)


def ejemplo_wrapper_recalc():
    """Ejemplo: usar la función wrapper recalc para compatibilidad"""
    
    print("\n" + "="*60)
    print("EJEMPLO USANDO RECALC (wrapper)")
    print("="*60)
    
    binance = BinanceClient()
    symbol = 'SOLUSDT'
    
    df = binance.history(symbol, '4h', 300)
    fecha = df.index[-1].strftime('%Y-%m-%d')
    comision = binance.calcular_comision(symbol)
    
    # Usar recalc (función wrapper que hace todo)
    resultados = recalc(symbol, fecha, '4h', df, comision, display=0)
    
    print("\nResultados de recalc:")
    print(f"  Tendencia: {resultados['tendencia']}")
    print(f"  Volatilidad: {resultados['volatilidad']:.2f}%")
    print(f"  Precio actual: ${resultados['precio_actual']:.2f}")
    print(f"  Resistencias: {len(resultados['resistance_levels'])}")
    print(f"  Soportes: {len(resultados['support_levels'])}")


# ========================================
# MENÚ PRINCIPAL
# ========================================

def main():
    """Menú principal para ejecutar ejemplos"""
    
    print("\n" + "="*60)
    print("SISTEMA DE ANÁLISIS TÉCNICO DE TRADING")
    print("="*60)
    print("\nSelecciona un ejemplo para ejecutar:")
    print("\n1. Ejemplo básico (recomendado para empezar)")
    print("2. Análisis multi-timeframe")
    print("3. Comparación de múltiples activos")
    print("4. Con credenciales (balance y trading)")
    print("5. Análisis profundo completo")
    print("6. Usar wrapper recalc")
    print("7. Ejecutar todos los ejemplos")
    print("\n0. Salir")
    
    try:
        opcion = input("\nOpción: ").strip()
        
        if opcion == '1':
            ejemplo_basico()
        elif opcion == '2':
            ejemplo_multiple_timeframes()
        elif opcion == '3':
            ejemplo_comparacion_activos()
        elif opcion == '4':
            ejemplo_con_credenciales()
        elif opcion == '5':
            ejemplo_analisis_profundo()
        elif opcion == '6':
            ejemplo_wrapper_recalc()
        elif opcion == '7':
            ejemplo_basico()
            input("\nPresiona Enter para continuar...")
            ejemplo_multiple_timeframes()
            input("\nPresiona Enter para continuar...")
            ejemplo_comparacion_activos()
            input("\nPresiona Enter para continuar...")
            ejemplo_analisis_profundo()
        elif opcion == '0':
            print("\n¡Hasta luego!")
            return
        else:
            print("\n❌ Opción inválida")
            
    except KeyboardInterrupt:
        print("\n\n¡Programa interrumpido!")
    except Exception as e:
        logger.error(f"Error en la ejecución: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    # Ejecutar el ejemplo básico directamente (comentar para usar menú)
    ejemplo_basico()
    
    # O descomentar para usar el menú interactivo
    # main()