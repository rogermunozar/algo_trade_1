# clients/binance_client.py
"""
Cliente para interactuar con Binance API
Maneja conexión, obtención de datos y operaciones de trading
"""
from binance.spot import Spot
import pandas as pd
import json


def load_secrets(filepath="secrets/binance.secrets.json"):
    """Carga las credenciales de API desde archivo JSON"""
    with open(filepath) as f:
        secrets = json.load(f)
    return secrets["api_key"], secrets["api_secret"]


class BinanceClient:
    def __init__(self, api_key=None, api_secret=None):
        """
        Inicializa la conexión con Binance
        api_key y api_secret son opcionales para datos públicos
        """
        self.client = Spot(api_key=api_key, api_secret=api_secret) if api_key else Spot()
        self.api_key = api_key
        self.api_secret = api_secret
    
    def calcular_comision(self, symbol):
        """
        Calcula la comisión específica de Binance para un símbolo
        
        Parámetros:
        - symbol: str (ej: 'BTCUSDT', 'BNBUSDT')
        
        Retorna:
        - float: comisión en decimal (ej: 0.002 para 0.2%)
        """
        binance_comision_base = 0.1  # 0.1%
        comision = binance_comision_base / 100 * 2  # Entrada y salida
        
        # Descuento si se opera con BNB
        if symbol == "BNBUSDT":
            comision = comision * 0.75
        
        return comision
    
    def history(self, symbol, interval, limit, start_time=None, end_time=None):
        """
        Obtiene datos históricos de Binance y retorna un DataFrame
        
        Parámetros:
        - symbol: str (ej: 'BTCUSDT')
        - interval: str (ej: '1h', '15m', '1d')
        - limit: int (cantidad de velas)
        - start_time: timestamp en milisegundos (opcional)
        - end_time: timestamp en milisegundos (opcional)
        
        Retorna:
        - DataFrame con columnas: open, high, low, close, volume e índice fecha
        """
        # Obtener datos de Binance
        if start_time is None:
            klines = self.client.klines(symbol=symbol, interval=interval, 
                                       limit=limit, endTime=end_time)
        else:
            klines = self.client.klines(symbol=symbol, interval=interval, 
                                       limit=limit, startTime=start_time, 
                                       endTime=end_time)
        
        # Crear DataFrame
        df = pd.DataFrame(klines)
        df = df[[0, 1, 2, 3, 4, 5]]  # Solo las primeras 6 columnas
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Convertir a numérico
        df['timestamp'] = df['timestamp']
        df['open'] = pd.to_numeric(df['open'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        # Ajustar timestamp y convertir a fecha
        df['timestamp'] = df['timestamp'] - 10800000
        df['fecha'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('fecha')
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        
        return df
    
    def resample(self, df, interval):
        """
        Resampling del DataFrame a un intervalo diferente
        
        Parámetros:
        - df: DataFrame con datos OHLCV
        - interval: str (ej: '1H', '4H', '1D')
        
        Retorna:
        - DataFrame resampleado
        """
        dfX = df.resample(interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        return dfX
    
    def get_account_balance(self):
        """
        Obtiene el balance de la cuenta
        Requiere API key y secret
        """
        if not self.api_key:
            raise ValueError("Se requieren credenciales de API para obtener el balance")
        
        account_info = self.client.account()
        balances = account_info['balances']
        
        # Filtrar solo balances con cantidad > 0
        active_balances = [
            {
                'asset': b['asset'],
                'free': float(b['free']),
                'locked': float(b['locked']),
                'total': float(b['free']) + float(b['locked'])
            }
            for b in balances
            if float(b['free']) > 0 or float(b['locked']) > 0
        ]
        
        return active_balances
    
    def get_current_price(self, symbol):
        """Obtiene el precio actual de un símbolo"""
        ticker = self.client.ticker_price(symbol=symbol)
        return float(ticker['price'])
    
    def create_order(self, symbol, side, order_type, quantity, price=None):
        """
        Crea una orden en Binance
        Requiere API key y secret
        
        Parámetros:
        - symbol: str (ej: 'BTCUSDT')
        - side: str ('BUY' o 'SELL')
        - order_type: str ('LIMIT', 'MARKET', etc)
        - quantity: float
        - price: float (requerido para LIMIT orders)
        """
        if not self.api_key:
            raise ValueError("Se requieren credenciales de API para crear órdenes")
        
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity
        }
        
        if order_type == 'LIMIT':
            if price is None:
                raise ValueError("Se requiere precio para órdenes LIMIT")
            params['price'] = price
            params['timeInForce'] = 'GTC'  # Good Till Cancel
        
        order = self.client.new_order(**params)
        return order
    
    def cancel_order(self, symbol, order_id):
        """
        Cancela una orden
        Requiere API key y secret
        """
        if not self.api_key:
            raise ValueError("Se requieren credenciales de API para cancelar órdenes")
        
        result = self.client.cancel_order(symbol=symbol, orderId=order_id)
        return result
    
    def get_open_orders(self, symbol=None):
        """
        Obtiene órdenes abiertas
        Requiere API key y secret
        
        Parámetros:
        - symbol: str opcional (si no se provee, retorna todas las órdenes abiertas)
        """
        if not self.api_key:
            raise ValueError("Se requieren credenciales de API para ver órdenes")
        
        if symbol:
            orders = self.client.get_open_orders(symbol=symbol)
        else:
            orders = self.client.get_open_orders()
        
        return orders