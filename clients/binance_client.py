# clients/binance_client.py
"""
Cliente mejorado para interactuar con Binance API
Incluye: manejo de errores, rate limiting, retry logic, y más funcionalidades
"""
from binance.spot import Spot
from binance.error import ClientError, ServerError
import pandas as pd
import json
import time
from typing import Optional, Dict, List, Tuple
from functools import wraps
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retry_on_error(max_retries=3, delay=1):
    """Decorador para reintentar en caso de errores de red"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (ServerError, ConnectionError) as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Intento {attempt + 1} falló: {e}. Reintentando en {delay}s...")
                    time.sleep(delay * (attempt + 1))
            return func(*args, **kwargs)
        return wrapper
    return decorator


def load_secrets_from_file(filepath="secrets/binance.secrets.json"):
    """
    Carga las credenciales de API desde archivo JSON (legacy)
    
    DEPRECADO: Usar load_secrets() que lee de variables de entorno
    
    Args:
        filepath: Ruta al archivo de credenciales
        
    Returns:
        Tuple[str, str]: (api_key, api_secret)
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        KeyError: Si faltan las claves en el JSON
    """
    logger.warning("load_secrets_from_file está deprecado. Usa load_secrets() con variables de entorno")
    try:
        with open(filepath) as f:
            secrets = json.load(f)
        
        if "api_key" not in secrets or "api_secret" not in secrets:
            raise KeyError("El archivo debe contener 'api_key' y 'api_secret'")
            
        return secrets["api_key"], secrets["api_secret"]
    except FileNotFoundError:
        logger.error(f"Archivo de credenciales no encontrado: {filepath}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error al parsear JSON: {filepath}")
        raise


def load_secrets():
    """
    Carga las credenciales de API desde variables de entorno
    
    Variables requeridas:
    - BINANCE_API_KEY
    - BINANCE_API_SECRET
    
    Returns:
        Tuple[str, str]: (api_key, api_secret)
        
    Raises:
        ValueError: Si no se encuentran las variables de entorno
        
    Example:
        # En terminal (Linux/Mac):
        export BINANCE_API_KEY="tu_key_aqui"
        export BINANCE_API_SECRET="tu_secret_aqui"
        
        # En terminal (Windows CMD):
        set BINANCE_API_KEY=tu_key_aqui
        set BINANCE_API_SECRET=tu_secret_aqui
        
        # En terminal (Windows PowerShell):
        $env:BINANCE_API_KEY="tu_key_aqui"
        $env:BINANCE_API_SECRET="tu_secret_aqui"
        
        # En Python:
        from clients.binance_client import BinanceClient, load_secrets
        api_key, api_secret = load_secrets()
        binance = BinanceClient(api_key, api_secret)
    """
    import os
    
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        raise ValueError(
            "No se encontraron las credenciales en variables de entorno.\n"
            "Debes configurar:\n"
            "  - BINANCE_API_KEY\n"
            "  - BINANCE_API_SECRET\n\n"
            "Ejemplo (Linux/Mac):\n"
            "  export BINANCE_API_KEY='tu_key'\n"
            "  export BINANCE_API_SECRET='tu_secret'\n\n"
            "Ejemplo (Windows PowerShell):\n"
            "  $env:BINANCE_API_KEY='tu_key'\n"
            "  $env:BINANCE_API_SECRET='tu_secret'"
        )
    
    logger.info("Credenciales cargadas desde variables de entorno")
    return api_key, api_secret


class BinanceClient:
    """Cliente mejorado para Binance con manejo robusto de errores"""
    
    # Mapeo de intervalos válidos
    VALID_INTERVALS = {
        '1m', '3m', '5m', '15m', '30m',
        '1h', '2h', '4h', '6h', '8h', '12h',
        '1d', '3d', '1w', '1M'
    }
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 testnet: bool = False):
        """
        Inicializa la conexión con Binance
        
        Args:
            api_key: API key (opcional para datos públicos)
            api_secret: API secret (opcional para datos públicos)
            testnet: Si True, usa testnet en lugar de producción
        """
        # Definir base_url correctamente
        if testnet:
            base_url = "https://testnet.binance.vision"
        else:
            base_url = "https://api.binance.com"  # URL de producción
        
        if api_key and api_secret:
            self.client = Spot(api_key=api_key, api_secret=api_secret, base_url=base_url)
            logger.info("Cliente inicializado con credenciales")
        else:
            self.client = Spot(base_url=base_url)
            logger.info("Cliente inicializado en modo público (solo lectura)")
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._exchange_info = None  # Cache para info del exchange
    
    def calcular_comision(self, symbol: str) -> float:
        """
        Calcula la comisión específica de Binance para un símbolo
        
        Args:
            symbol: Símbolo del par (ej: 'BTCUSDT', 'BNBUSDT')
        
        Returns:
            float: Comisión en decimal (ej: 0.002 para 0.2%)
        """
        binance_comision_base = 0.1  # 0.1%
        comision = binance_comision_base / 100 * 2  # Entrada y salida
        
        # Descuento si se opera con BNB
        if symbol.startswith("BNB"):
            comision = comision * 0.75
        
        return comision
    
    @retry_on_error(max_retries=3)
    def history(self, symbol: str, interval: str, limit: int = 500, 
                start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Obtiene datos históricos de Binance y retorna un DataFrame
        
        Args:
            symbol: Símbolo del par (ej: 'BTCUSDT')
            interval: Intervalo temporal (ej: '1h', '15m', '1d')
            limit: Cantidad de velas (máx 1000)
            start_time: Timestamp en milisegundos (opcional)
            end_time: Timestamp en milisegundos (opcional)
        
        Returns:
            pd.DataFrame: DataFrame con columnas OHLCV e índice fecha
            
        Raises:
            ValueError: Si el intervalo no es válido o limit > 1000
            ClientError: Si hay error en la petición a Binance
        """
        # Validaciones
        if interval not in self.VALID_INTERVALS:
            raise ValueError(f"Intervalo inválido. Usa: {self.VALID_INTERVALS}")
        
        if limit > 1000:
            logger.warning("Binance limita a 1000 velas. Ajustando limit a 1000.")
            limit = 1000
        
        try:
            # Obtener datos de Binance
            if start_time is None:
                klines = self.client.klines(
                    symbol=symbol, 
                    interval=interval, 
                    limit=limit, 
                    endTime=end_time
                )
            else:
                klines = self.client.klines(
                    symbol=symbol, 
                    interval=interval, 
                    limit=limit, 
                    startTime=start_time, 
                    endTime=end_time
                )
            
            if not klines:
                raise ValueError(f"No se obtuvieron datos para {symbol}")
            
            # Crear DataFrame
            df = pd.DataFrame(klines)
            df = df[[0, 1, 2, 3, 4, 5]]
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # Convertir a numérico
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Ajustar timestamp (Argentina GMT-3)
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df['timestamp'] = df['timestamp'] - 10800000  # Ajuste de zona horaria
            df['fecha'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Establecer fecha como índice
            df = df.set_index('fecha')
            df = df[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # Eliminar filas con NaN
            df = df.dropna()
            
            logger.info(f"Obtenidos {len(df)} datos para {symbol} ({interval})")
            return df
            
        except ClientError as e:
            logger.error(f"Error de Binance API: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al obtener históricos: {e}")
            raise
    
    def history_multiple(self, symbol: str, interval: str, total_candles: int) -> pd.DataFrame:
        """
        Obtiene más de 1000 velas haciendo múltiples peticiones
        
        Args:
            symbol: Símbolo del par
            interval: Intervalo temporal
            total_candles: Total de velas deseadas
            
        Returns:
            pd.DataFrame: DataFrame con todas las velas
        """
        all_data = []
        remaining = total_candles
        end_time = None
        
        while remaining > 0:
            limit = min(remaining, 1000)
            df_chunk = self.history(symbol, interval, limit, end_time=end_time)
            
            if df_chunk.empty:
                break
            
            all_data.append(df_chunk)
            remaining -= len(df_chunk)
            
            # Actualizar end_time para la siguiente petición
            end_time = int(df_chunk.index[0].timestamp() * 1000) - 1
            
            if len(df_chunk) < limit:  # No hay más datos disponibles
                break
            
            time.sleep(0.1)  # Rate limiting
        
        if not all_data:
            return pd.DataFrame()
        
        # Combinar y ordenar
        df_final = pd.concat(all_data)
        df_final = df_final.sort_index()
        df_final = df_final[~df_final.index.duplicated(keep='first')]
        
        logger.info(f"Total de velas obtenidas: {len(df_final)}")
        return df_final
    
    def resample(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """
        Resampling del DataFrame a un intervalo diferente
        
        Args:
            df: DataFrame con datos OHLCV
            interval: Nuevo intervalo (ej: '1H', '4H', '1D')
        
        Returns:
            pd.DataFrame: DataFrame resampleado
        """
        dfX = df.resample(interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Eliminar filas con NaN
        dfX = dfX.dropna()
        
        logger.info(f"Resampling: {len(df)} -> {len(dfX)} velas ({interval})")
        return dfX
    
    @retry_on_error(max_retries=2)
    def get_account_balance(self) -> List[Dict]:
        """
        Obtiene el balance de la cuenta
        Requiere API key y secret
        
        Returns:
            List[Dict]: Lista de balances con asset, free, locked, total
            
        Raises:
            ValueError: Si no hay credenciales configuradas
        """
        if not self.api_key:
            raise ValueError("Se requieren credenciales de API para obtener el balance")
        
        try:
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
        except ClientError as e:
            logger.error(f"Error al obtener balance: {e}")
            raise
    
    @retry_on_error(max_retries=2)
    def get_current_price(self, symbol: str) -> float:
        """
        Obtiene el precio actual de un símbolo
        
        Args:
            symbol: Símbolo del par
            
        Returns:
            float: Precio actual
        """
        try:
            ticker = self.client.ticker_price(symbol=symbol)
            return float(ticker['price'])
        except ClientError as e:
            logger.error(f"Error al obtener precio de {symbol}: {e}")
            raise
    
    def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """
        Obtiene el libro de órdenes (orderbook)
        
        Args:
            symbol: Símbolo del par
            limit: Profundidad del orderbook (5, 10, 20, 50, 100, 500, 1000)
            
        Returns:
            Dict: Orderbook con bids y asks
        """
        try:
            orderbook = self.client.depth(symbol=symbol, limit=limit)
            return {
                'bids': [(float(price), float(qty)) for price, qty in orderbook['bids']],
                'asks': [(float(price), float(qty)) for price, qty in orderbook['asks']],
                'timestamp': orderbook['lastUpdateId']
            }
        except ClientError as e:
            logger.error(f"Error al obtener orderbook: {e}")
            raise
    
    def get_ticker_24h(self, symbol: str) -> Dict:
        """
        Obtiene estadísticas de 24h para un símbolo
        
        Args:
            symbol: Símbolo del par
            
        Returns:
            Dict: Estadísticas de 24h
        """
        try:
            ticker = self.client.ticker_24hr(symbol=symbol)
            return {
                'symbol': ticker['symbol'],
                'price_change': float(ticker['priceChange']),
                'price_change_percent': float(ticker['priceChangePercent']),
                'high': float(ticker['highPrice']),
                'low': float(ticker['lowPrice']),
                'volume': float(ticker['volume']),
                'quote_volume': float(ticker['quoteVolume']),
                'open': float(ticker['openPrice']),
                'close': float(ticker['lastPrice']),
                'trades': int(ticker['count'])
            }
        except ClientError as e:
            logger.error(f"Error al obtener ticker 24h: {e}")
            raise
    
    @retry_on_error(max_retries=2)
    def create_order(self, symbol: str, side: str, order_type: str, 
                    quantity: float, price: Optional[float] = None, 
                    stop_price: Optional[float] = None,
                    time_in_force: str = 'GTC') -> Dict:
        """
        Crea una orden en Binance
        Requiere API key y secret
        
        Args:
            symbol: Símbolo del par (ej: 'BTCUSDT')
            side: 'BUY' o 'SELL'
            order_type: 'LIMIT', 'MARKET', 'STOP_LOSS_LIMIT', etc
            quantity: Cantidad a operar
            price: Precio límite (requerido para LIMIT orders)
            stop_price: Precio de activación (para órdenes STOP)
            time_in_force: 'GTC', 'IOC', 'FOK'
        
        Returns:
            Dict: Información de la orden creada
            
        Raises:
            ValueError: Si faltan parámetros requeridos
        """
        if not self.api_key:
            raise ValueError("Se requieren credenciales de API para crear órdenes")
        
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': quantity
        }
        
        if order_type.upper() == 'LIMIT':
            if price is None:
                raise ValueError("Se requiere precio para órdenes LIMIT")
            params['price'] = price
            params['timeInForce'] = time_in_force
        
        if 'STOP' in order_type.upper():
            if stop_price is None:
                raise ValueError("Se requiere stop_price para órdenes STOP")
            params['stopPrice'] = stop_price
        
        try:
            order = self.client.new_order(**params)
            logger.info(f"Orden creada: {order['orderId']} - {side} {quantity} {symbol}")
            return order
        except ClientError as e:
            logger.error(f"Error al crear orden: {e}")
            raise
    
    def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """
        Cancela una orden
        Requiere API key y secret
        
        Args:
            symbol: Símbolo del par
            order_id: ID de la orden a cancelar
            
        Returns:
            Dict: Información de la orden cancelada
        """
        if not self.api_key:
            raise ValueError("Se requieren credenciales de API para cancelar órdenes")
        
        try:
            result = self.client.cancel_order(symbol=symbol, orderId=order_id)
            logger.info(f"Orden cancelada: {order_id}")
            return result
        except ClientError as e:
            logger.error(f"Error al cancelar orden: {e}")
            raise
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Obtiene órdenes abiertas
        Requiere API key y secret
        
        Args:
            symbol: Símbolo del par (opcional, si no se provee retorna todas)
        
        Returns:
            List[Dict]: Lista de órdenes abiertas
        """
        if not self.api_key:
            raise ValueError("Se requieren credenciales de API para ver órdenes")
        
        try:
            if symbol:
                orders = self.client.get_open_orders(symbol=symbol)
            else:
                orders = self.client.get_open_orders()
            
            return orders
        except ClientError as e:
            logger.error(f"Error al obtener órdenes abiertas: {e}")
            raise
    
    def get_order_history(self, symbol: str, limit: int = 500) -> List[Dict]:
        """
        Obtiene historial de órdenes
        
        Args:
            symbol: Símbolo del par
            limit: Cantidad de órdenes (máx 1000)
            
        Returns:
            List[Dict]: Lista de órdenes históricas
        """
        if not self.api_key:
            raise ValueError("Se requieren credenciales de API")
        
        try:
            orders = self.client.get_orders(symbol=symbol, limit=limit)
            return orders
        except ClientError as e:
            logger.error(f"Error al obtener historial de órdenes: {e}")
            raise
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Obtiene información del símbolo (filters, precisión, etc)
        
        Args:
            symbol: Símbolo del par
            
        Returns:
            Dict: Información del símbolo
        """
        if self._exchange_info is None:
            self._exchange_info = self.client.exchange_info()
        
        for s in self._exchange_info['symbols']:
            if s['symbol'] == symbol:
                return s
        
        raise ValueError(f"Símbolo {symbol} no encontrado")
    
    def get_min_notional(self, symbol: str) -> float:
        """
        Obtiene el valor mínimo de operación para un símbolo
        
        Args:
            symbol: Símbolo del par
            
        Returns:
            float: Valor mínimo notional
        """
        info = self.get_symbol_info(symbol)
        for f in info['filters']:
            if f['filterType'] == 'NOTIONAL':
                return float(f['minNotional'])
        return 0.0