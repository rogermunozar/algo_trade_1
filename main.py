# main.py
"""
Script principal con sistema de Smart Exit y Capital Management
Versión actualizada con gestión de capital real y configuración centralizada
"""
from clients.binance_client import BinanceClient, load_secrets
from analysis.indicators import calcular_indicadores
from analysis.risk_management import apply_smart_exit, calcular_metricas_trailing
from analysis.strategies import apply_strategy, list_strategies, compare_strategies  # 🆕
from visualization.chart_plotter import graficar_analisis, graficar_comparacion
import config  # 🆕 Importar configuración centralizada
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


def ejemplo_con_smart_exit(symbol=None, 
                           interval=None, 
                           limit=None,
                           initial_capital=None,
                           risk_per_trade=None,
                           leverage=None,
                           profile=None): 
    """
    Ejemplo con Smart Exit y gestión de capital real
    
    Args:
        symbol: Par de trading (usa config.DEFAULT_SYMBOL si None)
        interval: Intervalo temporal (usa config.DEFAULT_INTERVAL si None)
        limit: Cantidad de velas (usa config.DEFAULT_LIMIT si None)
        initial_capital: Capital inicial (usa config.INITIAL_CAPITAL si None)
        risk_per_trade: % de riesgo (usa config.RISK_PER_TRADE si None)
        leverage: Apalancamiento (usa config.LEVERAGE si None)
        profile: Usar perfil predefinido ('conservador', 'balanceado', 'agresivo')
    """
    #  Aplicar valores del config.py si no se especifican
    symbol = symbol or config.DEFAULT_SYMBOL
    interval = interval or config.DEFAULT_INTERVAL
    limit = limit or config.DEFAULT_LIMIT
    initial_capital = initial_capital or config.INITIAL_CAPITAL
    risk_per_trade = risk_per_trade or config.RISK_PER_TRADE
    leverage = leverage or config.LEVERAGE
    
    # Si se especifica un perfil, usar esos valores
    if profile:
        profile_config = config.get_profile(profile)
        risk_per_trade = profile_config['risk']
        leverage = profile_config['leverage']
        atr_mult = profile_config['atr_mult']
        tp_mult = profile_config['tp_mult']
        print(f"\n🎯 Usando perfil: {profile.upper()}")
        print(f"   {profile_config['description']}")
    else:
        atr_mult = config.ATR_MULTIPLIER
        tp_mult = config.TP_MULTIPLIER
    
    logger.info(f"Ejecutando Smart Exit para {symbol}")
    
    print("\n" + "="*70)
    print(f"🎯 SMART EXIT CON CAPITAL MANAGEMENT - {symbol}")
    print("="*70)
    print(f"💰 Capital inicial: ${initial_capital:,.2f}")
    print(f"📊 Riesgo por trade: {risk_per_trade}%")
    print(f"⚡ Apalancamiento: {leverage}x")
    print("="*70)
    
    # Inicializar cliente
    binance = BinanceClient()
    
    # Obtener datos
    print(f"\n🔍 Obteniendo {limit} velas de {symbol} ({interval})...")
    df = binance.history(symbol, interval, limit)
    
    # Calcular indicadores
    print("🔧 Calculando indicadores técnicos...")
    fecha = df.index[-1].strftime('%Y-%m-%d')
    
    # 🔑 OBTENER COMISIÓN REAL DE BINANCE
    comision = binance.calcular_comision(symbol)
    print(f"💸 Comisión de Binance: {comision*100}%")
    
    # 🆕 Seleccionar modo de señales
    if config.SIGNAL_MODE == 'strategy':
        signal_mode = 'normal'  # Para indicators.py
        print(f"🎯 Usando estrategia: {config.STRATEGY_NAME}")
    else:
        signal_mode = config.SIGNAL_STRICTNESS
        print(f"🎯 Señales mejoradas - Modo: {signal_mode}")
    
    datos = calcular_indicadores(symbol, fecha, interval, df, comision, signal_mode=signal_mode)
    
    dfX = datos['dfX']
    
    # 🆕 Si se usa estrategia específica, reemplazar señales
    if config.SIGNAL_MODE == 'strategy':
        print(f"   Aplicando estrategia: {config.STRATEGY_NAME}...")
        dfX['senal'] = apply_strategy(dfX, strategy_name=config.STRATEGY_NAME)
        print(f"   Señales generadas: {(dfX['senal'] != 0).sum()}")
    
    # Aplicar Smart Exit CON CAPITAL MANAGEMENT
    print("🎯 Aplicando sistema de Smart Exit con gestión de capital...")
    df_with_exit = apply_smart_exit(
        dfX,
        entry_col='senal',
        price_col='close',
        atr_col='ATR',
        ema_short_col='EMA_12',
        ema_long_col='EMA_26',
        atr_mult=atr_mult,
        tp_multiplier=tp_mult,
        break_even_pct=config.BREAK_EVEN_PCT,
        support_short=config.SUPPORT_SHORT,
        # 🆕 PARÁMETROS DE CAPITAL
        initial_capital=initial_capital,
        risk_per_trade_pct=risk_per_trade,
        commission=comision,  # 🔑 Usar comisión real de Binance
        leverage=leverage,
        log_trades=True
    )
    
    # Calcular métricas
    metricas = calcular_metricas_trailing(df_with_exit)
    
    # Mostrar resultados
    _mostrar_resultados_basicos(datos, symbol)
    _mostrar_metricas_capital(metricas)
    _mostrar_senal_actual(dfX, df_with_exit, symbol)
    
    # Graficar
    print("\n📈 Generando gráfico...")
    graficar_analisis(datos, mostrar_avanzado=True)
    
    return df_with_exit, metricas


def ejemplo_comparacion_estrategias(symbol=None,
                                    interval=None,
                                    limit=None,
                                    initial_capital=None):
    """
    🆕 Compara todas las estrategias disponibles
    
    Args:
        symbol: Par de trading
        interval: Intervalo temporal
        limit: Cantidad de velas
        initial_capital: Capital inicial
    """
    # Usar valores del config si no se especifican
    symbol = symbol or config.DEFAULT_SYMBOL
    interval = interval or config.DEFAULT_INTERVAL
    limit = limit or config.DEFAULT_LIMIT
    initial_capital = initial_capital or config.INITIAL_CAPITAL
    
    print("\n" + "="*70)
    print(f"📊 COMPARACIÓN DE ESTRATEGIAS - {symbol}")
    print("="*70)
    
    binance = BinanceClient()
    
    # Obtener datos
    print(f"\n🔍 Obteniendo {limit} velas...")
    df = binance.history(symbol, interval, limit)
    fecha = df.index[-1].strftime('%Y-%m-%d')
    comision = binance.calcular_comision(symbol)
    
    # Calcular indicadores
    datos = calcular_indicadores(symbol, fecha, interval, df, comision)
    dfX = datos['dfX']
    
    # Obtener lista de estrategias
    strategies_info = list_strategies()
    
    print(f"\n🧪 Testeando {len(strategies_info)} estrategias...\n")
    
    resultados = []
    
    for strategy_name, description in strategies_info.items():
        print(f"Testeando: {strategy_name}...")
        
        # Aplicar estrategia
        dfX_copy = dfX.copy()
        dfX_copy['senal'] = apply_strategy(dfX_copy, strategy_name=strategy_name)
        
        # Aplicar smart exit
        df_test = apply_smart_exit(
            dfX_copy,
            entry_col='senal',
            initial_capital=initial_capital,
            risk_per_trade_pct=config.RISK_PER_TRADE,
            commission=comision,
            leverage=config.LEVERAGE,
            log_trades=False
        )
        
        # Calcular métricas
        metricas = calcular_metricas_trailing(df_test)
        
        if metricas and metricas.get('total_trades', 0) > 0:
            resultados.append({
                'estrategia': strategy_name,
                'descripcion': description,
                'trades': metricas['total_trades'],
                'win_rate': metricas['win_rate'],
                'roi': metricas['roi_total'],
                'capital_final': metricas['capital_final'],
                'profit_factor': metricas['profit_factor'],
                'max_dd': metricas['max_drawdown']
            })
        else:
            resultados.append({
                'estrategia': strategy_name,
                'descripcion': description,
                'trades': 0,
                'win_rate': 0,
                'roi': 0,
                'capital_final': initial_capital,
                'profit_factor': 0,
                'max_dd': 0
            })
    
    # Mostrar resultados
    _mostrar_resultados_comparacion_estrategias(resultados, symbol, initial_capital)


def _mostrar_resultados_comparacion_estrategias(resultados, symbol, capital_inicial):
    """Muestra tabla de comparación de estrategias"""
    print(f"\n{'='*100}")
    print(f"COMPARACIÓN DE ESTRATEGIAS - {symbol}")
    print(f"Capital inicial: ${capital_inicial:,.2f}")
    print(f"{'='*100}")
    print(f"{'Estrategia':<20} {'Trades':<8} {'Win%':<8} {'ROI%':<10} {'PF':<8} {'Final':<12}")
    print("-" * 100)
    
    for r in resultados:
        print(f"{r['estrategia']:<20} {r['trades']:<8} {r['win_rate']:<8.1f} "
              f"{r['roi']:<10.2f} {r['profit_factor']:<8.2f} ${r['capital_final']:<11,.2f}")
    
    print(f"{'='*100}")
    
    # Mejor estrategia por ROI
    mejores = [r for r in resultados if r['trades'] > 0]
    if mejores:
        mejor_roi = max(mejores, key=lambda x: x['roi'])
        mejor_winrate = max(mejores, key=lambda x: x['win_rate'])
        mejor_pf = max(mejores, key=lambda x: x['profit_factor'])
        
        print(f"\n🏆 MEJORES ESTRATEGIAS:")
        print(f"  Por ROI: {mejor_roi['estrategia']} ({mejor_roi['roi']:+.2f}%)")
        print(f"  Por Win Rate: {mejor_winrate['estrategia']} ({mejor_winrate['win_rate']:.1f}%)")
        print(f"  Por Profit Factor: {mejor_pf['estrategia']} ({mejor_pf['profit_factor']:.2f})")
    else:
        print(f"\n⚠️  Ninguna estrategia generó trades en este período")


#def ejemplo_basico(symbol='BTCUSDT', interval='1h', limit=500):


def ejemplo_optimizacion_capital(symbol='BTCUSDT', 
                                 interval='1h', 
                                 limit=1000,
                                 initial_capital=10000.0):
    """
    Optimización de parámetros CON capital management
    Prueba diferentes configuraciones de riesgo y apalancamiento
    
    Args:
        symbol: Par de trading
        interval: Intervalo temporal
        limit: Cantidad de velas
        initial_capital: Capital inicial
    """
    logger.info(f"Optimizando parámetros con capital para {symbol}")
    
    print("\n" + "="*70)
    print(f"⚙️  OPTIMIZACIÓN CON CAPITAL MANAGEMENT - {symbol}")
    print("="*70)
    
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
            'risk': 1.0,
            'atr_mult': 2.0,
            'tp_mult': 3.0,
            'leverage': 1.0
        },
        {
            'nombre': 'Moderado',
            'risk': 2.0,
            'atr_mult': 1.5,
            'tp_mult': 2.0,
            'leverage': 2.0
        },
        {
            'nombre': 'Agresivo',
            'risk': 3.0,
            'atr_mult': 1.0,
            'tp_mult': 1.5,
            'leverage': 3.0
        },
    ]
    
    print(f"\n🧪 Probando {len(configuraciones)} configuraciones...\n")
    
    resultados = []
    
    for config in configuraciones:
        print(f"Testeando: {config['nombre']}...")
        
        df_test = apply_smart_exit(
            dfX.copy(),
            entry_col='senal',
            atr_mult=config['atr_mult'],
            tp_multiplier=config['tp_mult'],
            initial_capital=initial_capital,
            risk_per_trade_pct=config['risk'],
            commission=comision,
            leverage=config['leverage'],
            log_trades=False
        )
        
        metricas = calcular_metricas_trailing(df_test)
        
        if metricas and metricas.get('total_trades', 0) > 0:
            resultados.append({
                'nombre': config['nombre'],
                'risk': config['risk'],
                'leverage': config['leverage'],
                'trades': metricas['total_trades'],
                'win_rate': metricas['win_rate'],
                'roi': metricas['roi_total'],
                'capital_final': metricas['capital_final'],
                'max_dd': metricas['max_drawdown'],
                'profit_factor': metricas['profit_factor']
            })
    
    # Mostrar resultados
    _mostrar_resultados_optimizacion_capital(resultados, symbol, initial_capital)


def _mostrar_resultados_basicos(datos, symbol):
    """Muestra resultados básicos del análisis"""
    print(f"\n{'='*70}")
    print(f"📊 ANÁLISIS DE {symbol}")
    print(f"{'='*70}")
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


def _mostrar_metricas_capital(metricas):
    """Muestra métricas con información de capital real"""
    if not metricas or metricas.get('total_trades', 0) == 0:
        print(f"\n⚠️  No se detectaron trades en el período")
        return
    
    print(f"\n{'='*70}")
    print(f"💰 MÉTRICAS DE CAPITAL MANAGEMENT")
    print(f"{'='*70}")
    
    print(f"\n📊 CAPITAL:")
    print(f"   ├─ Inicial: ${metricas['capital_inicial']:,.2f}")
    print(f"   ├─ Final: ${metricas['capital_final']:,.2f}")
    print(f"   └─ ROI Total: {metricas['roi_total']:+.2f}%")
    
    print(f"\n📈 TRADES:")
    print(f"   ├─ Total: {metricas['total_trades']}")
    print(f"   ├─ Por Stop: {metricas['exits_por_stop']}")
    print(f"   └─ Por TP: {metricas['exits_por_tp']}")
    
    print(f"\n💰 Performance:")
    print(f"   ├─ Ganadores: {metricas['winning_trades']} "
          f"({metricas['win_rate']:.1f}%)")
    print(f"   ├─ Perdedores: {metricas['losing_trades']}")
    print(f"   └─ Profit Factor: {metricas['profit_factor']:.2f}")
    
    print(f"\n💵 P&L (USD):")
    print(f"   ├─ Total: ${metricas['pnl_total']:+,.2f}")
    print(f"   ├─ Promedio: ${metricas['pnl_medio']:+,.2f}")
    print(f"   ├─ Avg Win: ${metricas['avg_win']:+,.2f}")
    print(f"   ├─ Avg Loss: ${metricas['avg_loss']:+,.2f}")
    print(f"   └─ Max Drawdown: ${metricas['max_drawdown']:,.2f}")
    
    print(f"\n💸 Comisiones pagadas: ${metricas['total_commission']:,.2f}")
    
    # Evaluación
    print(f"\n🎯 EVALUACIÓN:")
    if metricas['roi_total'] >= 10:
        print(f"   ✅ ROI excelente (≥10%)")
    elif metricas['roi_total'] >= 5:
        print(f"   ⚠️  ROI aceptable (5-10%)")
    elif metricas['roi_total'] > 0:
        print(f"   ⚠️  ROI positivo pero bajo (<5%)")
    else:
        print(f"   ❌ ROI negativo")
    
    if metricas['win_rate'] >= 60:
        print(f"   ✅ Win Rate excelente (≥60%)")
    elif metricas['win_rate'] >= 50:
        print(f"   ⚠️  Win Rate aceptable (50-60%)")
    else:
        print(f"   ❌ Win Rate bajo (<50%)")


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
        print(f"   🔴 VENTA - Considerar salida o SHORT en {symbol}")
    else:
        print(f"   ⚪ NEUTRAL - Esperar mejor oportunidad en {symbol}")


def _mostrar_resultados_optimizacion_capital(resultados, symbol, capital_inicial):
    """Muestra tabla de resultados de optimización con capital"""
    if not resultados:
        print("\n⚠️  No se obtuvieron resultados válidos")
        return
    
    print(f"\n{'='*90}")
    print(f"RESULTADOS DE OPTIMIZACIÓN - {symbol}")
    print(f"Capital inicial: ${capital_inicial:,.2f}")
    print(f"{'='*90}")
    print(f"{'Config':<15} {'Risk%':<8} {'Lev':<6} {'Trades':<8} {'Win%':<10} "
          f"{'ROI%':<10} {'Final':<12}")
    print("-" * 90)
    
    for r in resultados:
        print(f"{r['nombre']:<15} {r['risk']:<8.1f} {r['leverage']:<6.1f} "
              f"{r['trades']:<8} {r['win_rate']:<10.1f} {r['roi']:<10.2f} "
              f"${r['capital_final']:<11,.2f}")
    
    # Mejor configuración por ROI
    mejor = max(resultados, key=lambda x: x['roi'])
    print(f"\n🏆 Mejor configuración (por ROI): {mejor['nombre']}")
    print(f"   ROI: {mejor['roi']:.2f}%")
    print(f"   Capital Final: ${mejor['capital_final']:,.2f}")
    print(f"   Win Rate: {mejor['win_rate']:.1f}%")


def parse_args():
    """Parser de argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Sistema de Trading con Smart Exit y Capital Management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py                                    # Menú interactivo
  python main.py --ejemplo 1 --symbol ETHUSDT      # Smart Exit con capital
  python main.py -s BTCUSDT --capital 5000 --risk 1.5  # Capital personalizado
  python main.py --ejemplo 2 --leverage 3          # Con apalancamiento
        """
    )
    
    parser.add_argument('--symbol', '-s', type=str, default='BTCUSDT',
                       help='Par de trading (default: BTCUSDT)')
    parser.add_argument('--interval', '-i', type=str, default='1h',
                       help='Intervalo (default: 1h)')
    parser.add_argument('--limit', '-l', type=int, default=500,
                       help='Cantidad de velas (default: 500)')
    parser.add_argument('--ejemplo', '-e', type=int, choices=[1, 2, 3],
                       help='Ejecutar ejemplo (1=Smart Exit, 2=Optimización, 3=Comparar estrategias)')
    
    # 🆕 PARÁMETROS DE CAPITAL (con valores del config.py)
    parser.add_argument('--capital', '-c', type=float, default=None,
                       help=f'Capital inicial en USD (default: {config.INITIAL_CAPITAL})')
    parser.add_argument('--risk', '-r', type=float, default=None,
                       help=f'Riesgo por trade en %% (default: {config.RISK_PER_TRADE})')
    parser.add_argument('--leverage', '-lev', type=float, default=None,
                       help=f'Apalancamiento (default: {config.LEVERAGE})')
    parser.add_argument('--profile', '-p', type=str, 
                       choices=['conservador', 'balanceado', 'agresivo', 'scalper'],
                       help='Usar perfil predefinido (sobreescribe risk y leverage)')
    parser.add_argument('--strategy', '-st', type=str,
                       choices=['mean_reversion', 'breakout', 'trend_following', 'rsi_bb', 'ema_crossover'],
                       help='Estrategia específica a usar')
    
    return parser.parse_args()


def menu_interactivo(args):
    """Menú interactivo para seleccionar ejemplos"""
    print("\n" + "="*70)
    print("🚀 SISTEMA DE TRADING CON CAPITAL MANAGEMENT")
    print("="*70)
    
    # Mostrar configuración actual
    capital = args.capital or config.INITIAL_CAPITAL
    risk = args.risk or config.RISK_PER_TRADE
    leverage = args.leverage or config.LEVERAGE
    
    print(f"\nConfiguración:")
    print(f"  Par: {args.symbol} | Intervalo: {args.interval} | Velas: {args.limit}")
    print(f"  💰 Capital: ${capital:,.2f}")
    print(f"  📊 Riesgo/trade: {risk}%")
    print(f"  ⚡ Apalancamiento: {leverage}x")
    
    # Mostrar análisis de riesgo rápido
    max_loss = capital * (risk / 100)
    print(f"  💸 Pérdida máxima por trade: ${max_loss:.2f}")
    
    print("\nEJEMPLOS DISPONIBLES:")
    print("\n1. Smart Exit con Capital Management ⭐")
    print("2. Optimización de parámetros con Capital")
    print("3. Mostrar análisis de riesgo detallado")
    print("4. Cambiar a perfil predefinido")
    print("5. Comparar todas las estrategias 🆕")
    print("6. Cambiar estrategia de señales 🆕")
    print("\n9. Salir")
    
    try:
        opcion = input("\nSelecciona una opción: ").strip()
        
        if opcion == '1':
            ejemplo_con_smart_exit(
                args.symbol, args.interval, args.limit,
                args.capital, args.risk, args.leverage,
                profile=args.profile if hasattr(args, 'profile') else None
            )
        elif opcion == '2':
            capital = args.capital or config.INITIAL_CAPITAL
            ejemplo_optimizacion_capital(
                args.symbol, args.interval, args.limit, capital
            )
        elif opcion == '3':
            # Mostrar análisis de riesgo
            capital = args.capital or config.INITIAL_CAPITAL
            risk = args.risk or config.RISK_PER_TRADE
            config.show_risk_analysis(capital, risk)
            input("\nPresiona Enter para continuar...")
            menu_interactivo(args)  # Volver al menú
        elif opcion == '4':
            # Cambiar perfil
            print("\nPerfiles disponibles:")
            for i, (name, profile) in enumerate(config.PROFILES.items(), 1):
                print(f"{i}. {name.upper()} - {profile['description']}")
            
            profile_choice = input("\nSelecciona perfil (1-4): ").strip()
            profile_names = list(config.PROFILES.keys())
            if profile_choice.isdigit() and 1 <= int(profile_choice) <= len(profile_names):
                args.profile = profile_names[int(profile_choice) - 1]
                profile_config = config.get_profile(args.profile)
                args.risk = profile_config['risk']
                args.leverage = profile_config['leverage']
                print(f"\n✅ Perfil cambiado a: {args.profile.upper()}")
                input("Presiona Enter para continuar...")
            menu_interactivo(args)  # Volver al menú
        
        elif opcion == '5':
            # 🆕 Comparar estrategias
            capital = args.capital or config.INITIAL_CAPITAL
            ejemplo_comparacion_estrategias(
                args.symbol, args.interval, args.limit, capital
            )
        
        elif opcion == '6':
            # 🆕 Cambiar estrategia
            print("\nEstrategias disponibles:")
            strategies = list_strategies()
            for i, (name, desc) in enumerate(strategies.items(), 1):
                current = " ⭐" if name == config.STRATEGY_NAME else ""
                print(f"{i}. {name}{current}")
                print(f"   {desc}")
            
            strat_choice = input("\nSelecciona estrategia (1-5): ").strip()
            strat_names = list(strategies.keys())
            if strat_choice.isdigit() and 1 <= int(strat_choice) <= len(strat_names):
                config.STRATEGY_NAME = strat_names[int(strat_choice) - 1]
                config.SIGNAL_MODE = 'strategy'
                print(f"\n✅ Estrategia cambiada a: {config.STRATEGY_NAME}")
                input("Presiona Enter para continuar...")
            menu_interactivo(args)  # Volver al menú
        
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
        if args.ejemplo == 1:
            ejemplo_con_smart_exit(
                args.symbol, args.interval, args.limit,
                args.capital, args.risk, args.leverage
            )
        elif args.ejemplo == 2:
            ejemplo_optimizacion_capital(
                args.symbol, args.interval, args.limit, args.capital
            )
    else:
        # Modo interactivo
        menu_interactivo(args)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Modo CLI
        main()
    else:
        # 🎯 EJECUCIÓN POR DEFECTO - USA VALORES DE config.py
        print(f"\n💡 Usando configuración de config.py")
        print(f"   Para personalizar, edita config.py o usa argumentos CLI")
        print(f"   Ejemplo: python main.py --capital 5000 --risk 1.0\n")
        
        ejemplo_con_smart_exit()  # Usa todos los defaults de config.py