"""Rutas, supuestos economicos y parametros del generador del competidor.

Todo lo que es un supuesto (y no un dato) vive en este archivo, para que el
lector del reporte pueda auditarlo en un solo lugar.
"""

from __future__ import annotations

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CRUDOS = RAIZ / "data" / "raw"
DERIVADOS = RAIZ / "data" / "derivados"
FIGURAS = RAIZ / "data" / "figuras"

SEMILLA = 20260916

# Ventana de analisis: 2017 solo llega al 15 de agosto, se corta en julio para
# no mezclar meses parciales en las comparaciones interanuales.
INICIO = "2013-01-01"
FIN = "2017-07-31"

# ---------------------------------------------------------------- supuestos
# El dataset publica VOLUMEN (unidades), no dinero: no trae precios ni margenes.
# Estos son precios unitarios promedio en USD para Ecuador (economia dolarizada)
# hacia 2016, fijados por orden de magnitud a partir del tipo de producto de
# cada familia. Son el supuesto mas fuerte del proyecto y el reporte lo somete
# a analisis de sensibilidad.
PRECIO_UNITARIO = {
    "GROCERY I": 1.60, "BEVERAGES": 1.10, "PRODUCE": 1.35, "CLEANING": 2.40,
    "DAIRY": 1.70, "BREAD/BAKERY": 1.20, "POULTRY": 4.50, "MEATS": 6.50,
    "PERSONAL CARE": 3.20, "DELI": 5.00, "HOME CARE": 2.80, "EGGS": 2.60,
    "FROZEN FOODS": 3.50, "PREPARED FOODS": 3.80, "LIQUOR,WINE,BEER": 6.50,
    "SEAFOOD": 7.00, "GROCERY II": 2.20, "HOME AND KITCHEN I": 8.00,
    "HOME AND KITCHEN II": 9.50, "CELEBRATION": 4.50, "LINGERIE": 9.00,
    "LADIESWEAR": 14.00, "PLAYERS AND ELECTRONICS": 45.00, "AUTOMOTIVE": 7.50,
    "LAWN AND GARDEN": 12.00, "PET SUPPLIES": 5.50, "BEAUTY": 6.00,
    "SCHOOL AND OFFICE SUPPLIES": 2.50, "MAGAZINES": 3.00, "HARDWARE": 6.00,
    "HOME APPLIANCES": 85.00, "BABY CARE": 7.00, "BOOKS": 12.00,
}

# Margen bruto por familia: abarrote y bebidas son trafico de bajo margen,
# el fresco y el no-alimentario sostienen la contribucion.
MARGEN_BRUTO = {
    "GROCERY I": 0.18, "BEVERAGES": 0.20, "PRODUCE": 0.30, "CLEANING": 0.26,
    "DAIRY": 0.22, "BREAD/BAKERY": 0.34, "POULTRY": 0.16, "MEATS": 0.18,
    "PERSONAL CARE": 0.32, "DELI": 0.35, "HOME CARE": 0.28, "EGGS": 0.15,
    "FROZEN FOODS": 0.26, "PREPARED FOODS": 0.38, "LIQUOR,WINE,BEER": 0.28,
    "SEAFOOD": 0.20, "GROCERY II": 0.24, "HOME AND KITCHEN I": 0.36,
    "HOME AND KITCHEN II": 0.36, "CELEBRATION": 0.40, "LINGERIE": 0.45,
    "LADIESWEAR": 0.45, "PLAYERS AND ELECTRONICS": 0.22, "AUTOMOTIVE": 0.34,
    "LAWN AND GARDEN": 0.35, "PET SUPPLIES": 0.30, "BEAUTY": 0.42,
    "SCHOOL AND OFFICE SUPPLIES": 0.33, "MAGAZINES": 0.25, "HARDWARE": 0.32,
    "HOME APPLIANCES": 0.20, "BABY CARE": 0.30, "BOOKS": 0.30,
}

# Rango de sensibilidad aplicado a todos los precios a la vez.
SENSIBILIDAD_PRECIO = (0.70, 1.30)

# Familias con volumen tan bajo que la brecha no es accionable (ruido).
UMBRAL_VOLUMEN_MINIMO = 0.001  # 0.1 % del volumen total

# -------------------------------------------------- parametros del competidor
# "Andina" es una cadena ficticia. Su participacion se modela en escala logit:
#   logit(share_andina) = afinidad_familia + fuerza_plaza + deriva * t + ruido AR(1)
# Los valores estan en log-odds: 0 = paridad, negativo = Favorita domina.

# Afinidad por tipo de familia: Andina nace como cadena de frescos y se le
# dificulta el abarrote de alta rotacion.
AFINIDAD_FAMILIA = {
    "PRODUCE": 0.95, "MEATS": 0.80, "POULTRY": 0.75, "SEAFOOD": 0.70,
    "BREAD/BAKERY": 0.55, "DELI": 0.45, "EGGS": 0.35, "DAIRY": 0.10,
    "PREPARED FOODS": 0.30, "FROZEN FOODS": -0.10, "GROCERY I": -0.85,
    "BEVERAGES": -0.70, "CLEANING": -0.45, "HOME CARE": -0.30,
    "PERSONAL CARE": -0.20, "LIQUOR,WINE,BEER": -0.15,
}
AFINIDAD_POR_DEFECTO = -0.25  # resto de familias no alimentarias

# Fuerza por plaza: Andina es competitiva donde Favorita tiene poca densidad de
# tiendas, y marginal en Quito, donde Favorita esta instalada hace decadas.
FUERZA_PLAZA = {
    "Quito": -1.20, "Guayaquil": -0.35, "Cuenca": -0.10, "Ambato": 0.25,
    "Santo Domingo": 0.15, "Machala": 0.30, "Manta": 0.45, "Loja": 0.35,
    "Latacunga": 0.20, "Cayambe": -0.40, "Daule": 0.10, "Babahoyo": 0.25,
    "Esmeraldas": 0.40, "Ibarra": 0.30, "Riobamba": 0.20, "Quevedo": 0.35,
    "Salinas": 0.50, "Libertad": 0.45, "Playas": 0.55, "Puyo": 0.30,
    "Guaranda": 0.25, "El Carmen": 0.20,
}
FUERZA_POR_DEFECTO = 0.10

# Deriva mensual en log-odds: Andina gana terreno lentamente en fresco y lo
# pierde en abarrote (mide ~1.1 pp de share al ano en el centro de la curva).
DERIVA_FRESCO = 0.0045
DERIVA_RESTO = -0.0015
FAMILIAS_FRESCO = ("PRODUCE", "MEATS", "POULTRY", "SEAFOOD", "BREAD/BAKERY",
                   "DELI", "EGGS", "PREPARED FOODS")

RUIDO_AR1 = 0.65      # persistencia del choque mensual
RUIDO_SIGMA = 0.16    # desviacion del choque en log-odds

# Plazas golpeadas por el terremoto del 16 de abril de 2016 (Manabi y Esmeraldas).
PLAZAS_TERREMOTO = ("Manta", "Esmeraldas", "El Carmen", "Playas")

# Paleta compartida con el resto del portafolio, validada con el validador de la
# guia de visualizacion (banda de luminosidad, piso de croma, separacion CVD y
# piso de vision normal). El verde queda por debajo de 3:1 contra la superficie
# clara, asi que las figuras que lo usan llevan etiquetas visibles.
PALETA = {
    "favorita": "#2a78d6", "andina": "#eb6834", "acento": "#1baf7a",
    "brecha": "#e34948",
}
