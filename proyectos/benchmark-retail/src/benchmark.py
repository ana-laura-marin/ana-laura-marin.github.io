"""Generador de la cadena competidora ficticia 'Andina'.

La participacion de Andina en cada plaza y familia se modela en log-odds contra
las ventas reales de Favorita:

    eta = afinidad_familia + fuerza_plaza + deriva*t + interaccion + shock + ruido
    unidades_andina = unidades_favorita * exp(eta)

de modo que share_andina = 1 / (1 + exp(-eta)). Todo lo que Andina "hace" esta
declarado en config.py o en INTERACCIONES: no hay nada aleatorio sin semilla.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

# Ventajas especificas de plaza-familia, cada una con su racional de negocio.
# Son la senal que el analisis debe recuperar: no basta con que Andina sea
# fuerte en una ciudad, la pregunta es donde lo es mas de lo que su propia
# posicion en esa ciudad explicaria.
INTERACCIONES = [
    # (ciudad, familia, intensidad log-odds, mes de inicio, racional)
    ("Cuenca", "DAIRY", 0.70, "2015-01",
     "Acuerdo de exclusividad con una planta lechera del Azuay"),
    ("Guayaquil", "LIQUOR,WINE,BEER", 0.60, None,
     "Distribuidora de licores propia en el puerto"),
    ("Ambato", "CLEANING", 0.55, "2016-01",
     "Centro de distribucion regional de no alimentario"),
    ("Quito", "PERSONAL CARE", 0.50, "2015-06",
     "Formato de conveniencia con seccion de farmacia"),
    ("Machala", "FROZEN FOODS", 0.45, "2015-09",
     "Cadena de frio instalada para exportacion bananera"),
    ("Manta", "BEVERAGES", -0.50, None,
     "Favorita conserva la exclusividad del embotellador local"),
]

SHOCK_TERREMOTO = -0.35  # Andina pierde share en la costa tras abril de 2016
MESES_SHOCK = ("2016-04", "2016-05", "2016-06")


def _deriva(familia: str) -> float:
    return cfg.DERIVA_FRESCO if familia in cfg.FAMILIAS_FRESCO else cfg.DERIVA_RESTO


def _ruido_ar1(n_series: int, n_meses: int, rng: np.random.Generator) -> np.ndarray:
    """Choques mensuales persistentes, una serie por par plaza-familia."""
    innov = rng.normal(0, cfg.RUIDO_SIGMA, size=(n_series, n_meses))
    ruido = np.empty_like(innov)
    ruido[:, 0] = innov[:, 0] / np.sqrt(1 - cfg.RUIDO_AR1 ** 2)
    for t in range(1, n_meses):
        ruido[:, t] = cfg.RUIDO_AR1 * ruido[:, t - 1] + innov[:, t]
    return ruido


def generar(plaza: pd.DataFrame, semilla: int | None = None) -> pd.DataFrame:
    """Devuelve el panel de plaza con las columnas del competidor anadidas.

    Espera el panel ciudad x familia x mes de data_io.panel_plaza().
    """
    rng = np.random.default_rng(cfg.SEMILLA if semilla is None else semilla)

    df = plaza.sort_values(["city", "family", "mes"]).reset_index(drop=True)
    meses = np.sort(df.mes.unique())
    idx_mes = pd.Series(range(len(meses)), index=meses)

    claves = df[["city", "family"]].drop_duplicates().reset_index(drop=True)
    ruido = _ruido_ar1(len(claves), len(meses), rng)
    mapa_ruido = {(c, f): ruido[i] for i, (c, f) in
                  enumerate(zip(claves.city, claves.family))}

    t = df.mes.map(idx_mes).to_numpy()
    eta = (df.family.map(cfg.AFINIDAD_FAMILIA).fillna(cfg.AFINIDAD_POR_DEFECTO).to_numpy()
           + df.city.map(cfg.FUERZA_PLAZA).fillna(cfg.FUERZA_POR_DEFECTO).to_numpy()
           + df.family.map(_deriva).to_numpy() * t)

    eta += np.array([mapa_ruido[(c, f)][i] for c, f, i in
                     zip(df.city, df.family, t)])

    mes_txt = df.mes.dt.strftime("%Y-%m")
    for ciudad, familia, fuerza, desde, _ in INTERACCIONES:
        activa = (df.city == ciudad) & (df.family == familia)
        if desde is not None:
            activa &= mes_txt >= desde
        eta += np.where(activa, fuerza, 0.0)

    golpeadas = df.city.isin(cfg.PLAZAS_TERREMOTO) & mes_txt.isin(MESES_SHOCK)
    eta += np.where(golpeadas, SHOCK_TERREMOTO, 0.0)

    df["eta"] = eta
    df["unidades_andina"] = df.unidades * np.exp(eta)
    df["ingreso_andina"] = df.unidades_andina * df.family.map(cfg.PRECIO_UNITARIO)
    df["mercado_unidades"] = df.unidades + df.unidades_andina
    df["mercado_ingreso"] = df.ingreso + df.ingreso_andina
    df["share"] = df.unidades / df.mercado_unidades.replace(0, np.nan)

    # Andina promociona mas fuerte justo donde es debil: da una palanca que el
    # analisis puede contrastar contra la propia intensidad de Favorita.
    debilidad = np.clip(-df.eta, 0, None)
    df["intensidad_promo_andina"] = np.clip(
        df.intensidad_promo * (1 + 0.45 * debilidad)
        + rng.normal(0, 0.04, len(df)), 0, 1)

    return df


def celdas_sembradas() -> set[tuple[str, str]]:
    """Pares plaza-familia con ventaja sembrada (para el control de falsos positivos)."""
    return {(c, f) for c, f, fuerza, _, _ in INTERACCIONES if fuerza > 0}
