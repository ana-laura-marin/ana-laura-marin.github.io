"""Limpieza y enriquecimiento a nivel de transaccion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

NOCHE_INICIO, NOCHE_FIN = 22, 6   # franja nocturna [22:00, 06:00)


def limpiar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica filtros de calidad y devuelve (df limpio, bitacora de descartes)."""
    log = {"filas_iniciales": len(df)}

    df = df.drop_duplicates(
        subset=["cc_num", "trans_date_trans_time", "amt", "merchant"]
    )
    log["duplicados"] = log["filas_iniciales"] - len(df)

    antes = len(df)
    df = df[df["amt"] > 0]
    log["monto_no_positivo"] = antes - len(df)

    antes = len(df)
    df = df[df["category"].astype(str).isin(cfg.CATEGORIAS)]
    log["categoria_desconocida"] = antes - len(df)

    # El fraude no es una decision de gasto del tarjetahabiente: contaminaria
    # los perfiles de consumo. Se separa en lugar de mezclarse.
    log["transacciones_fraude"] = int(df["is_fraud"].sum())
    df_fraude = df[df["is_fraud"] == 1].copy()
    df = df[df["is_fraud"] == 0].copy()

    edad = _edad_anios(df)
    antes = len(df)
    df = df[(edad >= 18) & (edad <= 100)]
    log["edad_implausible"] = antes - len(df)

    log["filas_finales"] = len(df)
    return df.reset_index(drop=True), {"log": log, "fraude": df_fraude}


def _edad_anios(df: pd.DataFrame) -> pd.Series:
    return (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25


def enriquecer(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega las variables temporales, de taxonomia y conductuales."""
    ts = df["trans_date_trans_time"]

    df["edad"] = _edad_anios(df)
    df["hora"] = ts.dt.hour
    df["dia_semana"] = ts.dt.dayofweek
    df["mes"] = ts.dt.month
    df["anio_mes"] = ts.dt.to_period("M").astype(str)
    df["es_finde"] = (df["dia_semana"] >= 5).astype("int8")
    df["es_nocturna"] = ((df["hora"] >= NOCHE_INICIO) | (df["hora"] < NOCHE_FIN)).astype("int8")

    cat = df["category"].astype(str)
    df["tipo_gasto"] = cat.map(cfg.tipo_gasto).astype("category")
    df["canal"] = cat.map(cfg.canal).astype("category")
    df["categoria_es"] = cat.map(cfg.ETIQUETAS_ES).astype("category")

    df["log_amt"] = np.log(df["amt"])
    df["log_city_pop"] = np.log(df["city_pop"].clip(lower=1))

    # Sesgo de numero redondo: marcador clasico de compra deliberada o
    # planificada frente a gasto por impulso con precio de mercado.
    centavos = np.round(df["amt"] * 100).astype("int64")
    df["monto_redondo"] = (centavos % 500 == 0).astype("int8")

    return df
