# main.py
"""
Ejemplo de uso de los módulos organizados
"""
from clients.binance_client import BinanceClient, load_secrets
from analysis.indicators import calcular_indicadores
from visualization.chart_plotter import graficar_analisis, recalc


def ejemplo_uso_basico():
    """Ejemplo básico: obtener datos, calcular indicadores y graficar"""
    
    # 1. Cargar credenciales (opcional para datos públicos)
    # api_key, api_secret = load_secrets()
    # binance = BinanceClient(api_key, api_secret)
    
    # Sin credenciales (solo datos públicos)
    binance = BinanceClient()
    
    # 2. Obtener datos históricos
    symbol = 'BNBUSDT'
    interval = '1m'
    limit = 500
    
    print(f"Obteniendo datos de {symbol}...")
    df = binance.history(symbol=symbol, interval=interval, limit=limit)
    print(f"Datos obtenidos: {len(df)} velas")
    
    # 3. Calcular indicadores
    print("Calculando indicadores...")
    
    # Obtener fecha del último timestamp del DataFrame
    fecha_str = df.index[-1].strftime('%Y-%m-%d')
    
    # Calcular comisión específica de Binance (ahora es un método del cliente)
    comision = binance.calcular_comision(symbol)
    
    datos = calcular_indicadores(symbol, fecha_str, interval, df, comision, display=0)
    
    # 4. Graficar
    print("Generando gráfico...")
    graficar_analisis(datos)
    
    # Mostrar resultados
    print(f"\nNiveles de resistencia: {datos['resistance_levels']}")
    print(f"Niveles de soporte: {datos['support_levels']}")
    print(f"Pendiente resistencia: {datos['slope_resistance']}")
    print(f"Pendiente soporte: {datos['slope_support']}")


def ejemplo_uso_wrapper():
    """Ejemplo usando la función wrapper recalc (compatibilidad con código anterior)"""
    
    binance = BinanceClient()
    
    symbol = 'ETHUSDT'
    interval = '4h'
    limit = 300
    
    df = binance.history(symbol=symbol, interval=interval, limit=limit)
    
    # Obtener fecha del último timestamp
    fecha_str = df.index[-1].strftime('%Y-%m-%d')
    
    # Calcular comisión (ahora es un método del cliente)
    comision = binance.calcular_comision(symbol)
    
    # Usando la función wrapper que hace todo en uno
    resultados = recalc(symbol, "2024-12", interval, df, comision, display=1)
    
    print(f"Resistencias: {resultados['resistance_levels']}")
    print(f"Soportes: {resultados['support_levels']}")


def ejemplo_con_resample():
    """Ejemplo con resampling de datos a diferentes timeframes"""
    
    binance = BinanceClient()
    
    # Obtener datos en 1h
    symbol = 'BNBUSDT'
    df = binance.history(symbol=symbol, interval='1h', limit=1000)
    
    # Resamplear a 4h
    df_4h = binance.resample(df, '4H')
    
    # Obtener fecha del último timestamp
    fecha_str = df_4h.index[-1].strftime('%Y-%m-%d')
    
    # Calcular comisión (ahora es un método del cliente)
    comision = binance.calcular_comision(symbol)
    
    # Calcular y graficar
    datos = calcular_indicadores(symbol, fecha_str, "4h", df_4h, comision)
    graficar_analisis(datos)


def ejemplo_con_trading():
    """Ejemplo con operaciones de trading (requiere API keys)"""
    
    # Cargar credenciales
    api_key, api_secret = load_secrets()
    binance = BinanceClient(api_key, api_secret)
    
    # Ver balance
    print("Balance de cuenta:")
    balances = binance.get_account_balance()
    for balance in balances:
        print(f"  {balance['asset']}: {balance['total']}")
    
    # Ver precio actual
    symbol = 'BTCUSDT'
    precio_actual = binance.get_current_price(symbol)
    print(f"\nPrecio actual de {symbol}: {precio_actual}")
    
    # Ver órdenes abiertas
    ordenes = binance.get_open_orders(symbol)
    print(f"\nÓrdenes abiertas: {len(ordenes)}")
    
    # Crear una orden LIMIT de compra (ejemplo - ¡CUIDADO!)
    # cantidad = 0.001
    # precio_limite = precio_actual * 0.98  # 2% por debajo del precio actual
    # orden = binance.create_order(symbol, 'BUY', 'LIMIT', cantidad, precio_limite)
    # print(f"Orden creada: {orden}")


def ejemplo_datos_csv():
    """
    Ejemplo mostrando que el análisis es agnóstico del broker.
    Puedes usar datos de cualquier fuente (CSV, otro broker, etc.)
    """
    import pandas as pd
    
    # Simular carga de CSV (podría ser de cualquier broker)
    # df = pd.read_csv('datos_trading.csv', index_col='fecha', parse_dates=True)
    
    # O crear datos de ejemplo
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    df = pd.DataFrame({
        'open': [100 + i*0.1 for i in range(100)],
        'high': [101 + i*0.1 for i in range(100)],
        'low': [99 + i*0.1 for i in range(100)],
        'close': [100.5 + i*0.1 for i in range(100)],
        'volume': [1000 + i*10 for i in range(100)]
    }, index=dates)
    
    # Obtener fecha del último timestamp
    fecha_str = df.index[-1].strftime('%Y-%m-%d')
    
    # El análisis funciona exactamente igual
    symbol = 'CUSTOM_DATA'
    comision = 0.002  # 0.2% genérica
    
    datos = calcular_indicadores(symbol, fecha_str, "1h", df, comision)
    graficar_analisis(datos)
    
    print("\n✅ Este ejemplo muestra que el análisis es completamente agnóstico del broker!")


if __name__ == "__main__":
    # Ejecutar el ejemplo que desees:
    
    ejemplo_uso_basico()
    
    # ejemplo_uso_wrapper()
    
    # ejemplo_con_resample()
    
    # ejemplo_con_trading()  # Descomentar solo si tienes API keys configuradas
    
    # ejemplo_datos_csv()  # Demuestra que es agnóstico del broker