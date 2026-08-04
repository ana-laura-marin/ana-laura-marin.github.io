"""Configuracion central del proyecto: rutas, contrato de esquema y taxonomia de gasto."""

from pathlib import Path

# --- Rutas -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"

# Dataset de Kaggle
KAGGLE_DATASET = "priyamchoksi/credit-card-transactions-dataset"
RAW_CSV = DATA_RAW / "credit_card_transactions.csv"
SYNTHETIC_CSV = DATA_RAW / "synthetic_transactions.csv"

RANDOM_STATE = 42

# --- Contrato de esquema ---------------------------------------------------
# Columnas que el pipeline realmente consume. El CSV real trae 24; las extras
# (first, last, street, trans_num, unix_time, merch_lat, merch_long, ...) se
# ignoran para no cargar 354 MB de datos que no usamos.
COLUMNAS_REQUERIDAS = [
    "trans_date_trans_time",  # timestamp de la transaccion
    "cc_num",                 # identificador del tarjetahabiente
    "merchant",               # comercio
    "category",               # categoria de gasto
    "amt",                    # monto
    "gender",                 # genero del tarjetahabiente
    "city",
    "state",
    "city_pop",               # poblacion de la ciudad (proxy de urbanizacion)
    "job",                    # ocupacion (proxy de ingreso)
    "dob",                    # fecha de nacimiento -> edad
    "is_fraud",               # marca de fraude
]

DTYPES = {
    "cc_num": "int64",
    "merchant": "category",
    "category": "category",
    "amt": "float64",
    "gender": "category",
    "city": "category",
    "state": "category",
    "city_pop": "int32",
    "job": "category",
    "is_fraud": "int8",
}

# --- Taxonomia de categorias ----------------------------------------------
# Las 14 categorias de Sparkov. El sufijo _net/_pos distingue canal online vs
# presencial, lo que da una dimension de canal ademas de la de tipo de gasto.
CATEGORIAS = [
    "entertainment", "food_dining", "gas_transport", "grocery_net",
    "grocery_pos", "health_fitness", "home", "kids_pets", "misc_net",
    "misc_pos", "personal_care", "shopping_net", "shopping_pos", "travel",
]

ETIQUETAS_ES = {
    "entertainment": "Entretenimiento",
    "food_dining": "Restaurantes",
    "gas_transport": "Combustible y transporte",
    "grocery_net": "Supermercado (online)",
    "grocery_pos": "Supermercado (presencial)",
    "health_fitness": "Salud y bienestar",
    "home": "Hogar",
    "kids_pets": "Ninos y mascotas",
    "misc_net": "Varios (online)",
    "misc_pos": "Varios (presencial)",
    "personal_care": "Cuidado personal",
    "shopping_net": "Compras (online)",
    "shopping_pos": "Compras (presencial)",
    "travel": "Viajes",
}

# Clasificacion hedonico / utilitario: es el eje conductual del analisis.
# El gasto hedonico responde a impulso y estado de animo; el utilitario es
# recurrente y poco elastico. La hipotesis del proyecto es que la propension
# hedonica varia sistematicamente con demografia, canal y momento del dia.
HEDONICO = {
    "entertainment", "food_dining", "travel",
    "shopping_net", "shopping_pos", "personal_care",
}
UTILITARIO = {
    "grocery_net", "grocery_pos", "gas_transport",
    "home", "health_fitness", "kids_pets",
}
MIXTO = {"misc_net", "misc_pos"}

CANAL_ONLINE = {"grocery_net", "misc_net", "shopping_net"}


def tipo_gasto(categoria: str) -> str:
    if categoria in HEDONICO:
        return "hedonico"
    if categoria in UTILITARIO:
        return "utilitario"
    return "mixto"


def canal(categoria: str) -> str:
    return "online" if categoria in CANAL_ONLINE else "presencial"


# --- Segmentacion ----------------------------------------------------------
K_MIN, K_MAX = 2, 9          # rango de k a evaluar por silueta
K_FIJO = None                # si se define un entero, se salta la busqueda
MIN_TXN_POR_CLIENTE = 30     # descarta clientes con historial insuficiente
