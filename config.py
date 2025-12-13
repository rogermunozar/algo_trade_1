# config.py
"""
Configuración centralizada del sistema de trading
Ajusta estos valores según tu perfil de riesgo
"""

# ========================================
# CONFIGURACIÓN DE SEÑALES
# ========================================

# Modo de generación de señales
# Opciones:
#   'improved'  → Sistema mejorado con filtros (default) ⭐
#   'strategy'  → Usar estrategia específica del módulo strategies.py
SIGNAL_MODE = 'improved'  # o 'strategy'

# Si SIGNAL_MODE='improved', nivel de estrictez:
#   'strict'     → Muy conservador, señales de alta calidad (min_score=5)
#   'normal'     → Balanceado (min_score=4) ⭐
#   'aggressive' → Más señales, menor calidad (min_score=3)
SIGNAL_STRICTNESS = 'normal'

# Si SIGNAL_MODE='strategy', estrategia a usar:
# Opciones: 'mean_reversion', 'breakout', 'trend_following', 'rsi_bb', 'ema_crossover'
STRATEGY_NAME = 'mean_reversion'


# ========================================
# CONFIGURACIÓN DE CAPITAL
# ========================================

# Capital inicial en USD
INITIAL_CAPITAL = 10000.0

# Riesgo por trade (porcentaje del capital)
# Recomendaciones:
#   0.5-1.0%  → Conservador (profesional)
#   1.5-2.0%  → Balanceado (recomendado) ⭐
#   2.5-3.0%  → Agresivo (experiencia requerida)
#   5.0%+     → Muy agresivo (peligroso) ⚠️
RISK_PER_TRADE = 0.25

# Apalancamiento
# Recomendaciones:
#   1x   → Spot / Sin apalancamiento (más seguro) ⭐
#   2-3x → Apalancamiento moderado
#   5x+  → Alto riesgo (solo expertos) ⚠️
LEVERAGE = 0.25


# ========================================
# CONFIGURACIÓN DE TRADING
# ========================================

# Par de trading por defecto
DEFAULT_SYMBOL = 'BNBUSDT'

# Intervalo temporal por defecto
# Opciones: '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'
DEFAULT_INTERVAL = '1m'

# Cantidad de velas para análisis
DEFAULT_LIMIT = 500


# ========================================
# PARÁMETROS DE SMART EXIT
# ========================================

# Multiplicador de ATR para Stop Loss
# Valores típicos: 1.0 (tight), 1.5 (medio), 2.0 (amplio)
ATR_MULTIPLIER = 1.5

# Multiplicador para Take Profit (relación riesgo/recompensa)
# Ejemplo: 2.0 = riesgo 1:2 (si arriesgas $100, buscas ganar $200)
TP_MULTIPLIER = 2.0

# Porcentaje del camino a TP para activar Break-Even
# 0.9 = 90% del camino (recomendado)
BREAK_EVEN_PCT = 0.9

# Permitir operaciones SHORT
SUPPORT_SHORT = True


# ========================================
# PERFILES PREDEFINIDOS
# ========================================

PROFILES = {
    'conservador': {
        'risk': 0.5,
        'leverage': 1.0,
        'atr_mult': 2.0,
        'tp_mult': 3.0,
        'description': 'Bajo riesgo, crecimiento lento pero estable'
    },
    'balanceado': {
        'risk': 1.5,
        'leverage': 2.0,
        'atr_mult': 1.5,
        'tp_mult': 2.0,
        'description': 'Balance riesgo/recompensa óptimo (RECOMENDADO)'
    },
    'agresivo': {
        'risk': 3.0,
        'leverage': 3.0,
        'atr_mult': 1.0,
        'tp_mult': 1.5,
        'description': 'Alto riesgo, requiere experiencia'
    },
    'scalper': {
        'risk': 2.0,
        'leverage': 5.0,
        'atr_mult': 0.8,
        'tp_mult': 1.2,
        'description': 'Operaciones rápidas, alta frecuencia'
    }
}


# ========================================
# FUNCIONES HELPER
# ========================================

def get_profile(profile_name='balanceado'):
    """
    Obtiene la configuración de un perfil predefinido
    
    Args:
        profile_name: 'conservador', 'balanceado', 'agresivo', 'scalper'
    
    Returns:
        dict: Configuración del perfil
    """
    if profile_name not in PROFILES:
        print(f"⚠️  Perfil '{profile_name}' no encontrado. Usando 'balanceado'")
        profile_name = 'balanceado'
    
    return PROFILES[profile_name]


def calculate_max_loss_streak(capital, risk_pct):
    """
    Calcula cuántas pérdidas consecutivas puede soportar la cuenta
    antes de perder el 50% del capital
    
    Args:
        capital: Capital inicial
        risk_pct: Riesgo por trade en %
    
    Returns:
        int: Número de trades perdedores consecutivos
    """
    current = capital
    target = capital * 0.5
    streak = 0
    
    while current > target:
        loss = current * (risk_pct / 100)
        current -= loss
        streak += 1
    
    return streak


def show_risk_analysis(capital=INITIAL_CAPITAL, risk=RISK_PER_TRADE):
    """
    Muestra un análisis de riesgo basado en la configuración
    """
    print("\n" + "="*70)
    print("📊 ANÁLISIS DE RIESGO")
    print("="*70)
    print(f"💰 Capital: ${capital:,.2f}")
    print(f"📊 Riesgo por trade: {risk}%")
    print(f"💸 Pérdida máxima por trade: ${capital * (risk/100):,.2f}")
    print()
    
    # Calcular rachas de pérdidas
    streak_10 = capital
    for _ in range(10):
        streak_10 -= streak_10 * (risk/100)
    
    streak_20 = capital
    for _ in range(20):
        streak_20 -= streak_20 * (risk/100)
    
    loss_10 = ((streak_10 / capital) - 1) * 100
    loss_20 = ((streak_20 / capital) - 1) * 100
    
    print("🔻 Impacto de rachas perdedoras:")
    print(f"   10 pérdidas consecutivas: ${streak_10:,.2f} ({loss_10:+.1f}%)")
    print(f"   20 pérdidas consecutivas: ${streak_20:,.2f} ({loss_20:+.1f}%)")
    print()
    
    # Trades necesarios para duplicar
    win_rate = 0.6  # Asumiendo 60% win rate
    avg_rr = TP_MULTIPLIER  # Risk:Reward ratio
    
    expectancy = (win_rate * avg_rr * risk) - ((1 - win_rate) * risk)
    trades_to_double = 100 / expectancy if expectancy > 0 else float('inf')
    
    print(f"📈 Con {win_rate*100:.0f}% win rate y RR 1:{avg_rr}:")
    print(f"   Expectativa por trade: {expectancy:+.2f}%")
    print(f"   Trades para duplicar cuenta: {trades_to_double:.0f}")
    print()
    
    # Evaluación
    if risk <= 1.0:
        print("✅ CONSERVADOR: Crecimiento lento pero muy seguro")
    elif risk <= 2.0:
        print("✅ BALANCEADO: Buen equilibrio riesgo/recompensa")
    elif risk <= 3.0:
        print("⚠️  AGRESIVO: Requiere experiencia y disciplina")
    else:
        print("❌ MUY AGRESIVO: Alto riesgo de pérdida significativa")
    
    print("="*70)


def validate_config():
    """
    Valida que la configuración sea razonable
    """
    warnings = []
    
    if RISK_PER_TRADE > 5.0:
        warnings.append(f"⚠️  Risk muy alto ({RISK_PER_TRADE}%). Recomendado: ≤3%")
    
    if LEVERAGE > 5.0:
        warnings.append(f"⚠️  Leverage muy alto ({LEVERAGE}x). Recomendado: ≤3x")
    
    if RISK_PER_TRADE * LEVERAGE > 10:
        warnings.append(f"❌ PELIGRO: Risk × Leverage = {RISK_PER_TRADE * LEVERAGE}. ¡Muy arriesgado!")
    
    if ATR_MULTIPLIER < 1.0:
        warnings.append(f"⚠️  ATR multiplier bajo ({ATR_MULTIPLIER}). Stops muy ajustados")
    
    if TP_MULTIPLIER < 1.5:
        warnings.append(f"⚠️  TP multiplier bajo ({TP_MULTIPLIER}). Risk:Reward desfavorable")
    
    return warnings


# ========================================
# EJECUTAR AL IMPORTAR
# ========================================

if __name__ == "__main__":
    # Si ejecutas este archivo directamente, muestra el análisis
    print("\n🔧 CONFIGURACIÓN ACTUAL:")
    print(f"   Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"   Risk: {RISK_PER_TRADE}%")
    print(f"   Leverage: {LEVERAGE}x")
    print(f"   Symbol: {DEFAULT_SYMBOL}")
    print(f"   Interval: {DEFAULT_INTERVAL}")
    
    # Validar configuración
    warnings = validate_config()
    if warnings:
        print("\n⚠️  ADVERTENCIAS:")
        for warning in warnings:
            print(f"   {warning}")
    else:
        print("\n✅ Configuración validada correctamente")
    
    # Mostrar análisis de riesgo
    show_risk_analysis()
    
    # Mostrar perfiles disponibles
    print("\n📋 PERFILES DISPONIBLES:")
    for name, profile in PROFILES.items():
        print(f"\n   {name.upper()}:")
        print(f"      Risk: {profile['risk']}%")
        print(f"      Leverage: {profile['leverage']}x")
        print(f"      {profile['description']}")