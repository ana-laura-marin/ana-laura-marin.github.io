"""Construccion del panel a nivel de tarjetahabiente.

La unidad de analisis del proyecto no es la transaccion sino el cliente: lo que
interesa es la *composicion* de su gasto, no cada compra suelta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

COLS_SHARE = [f"share_{c}" for c in cfg.CATEGORIAS]


def _hhi(shares: np.ndarray) -> np.ndarray:
    """Indice Herfindahl sobre las participaciones: 1 = todo en una categoria."""
    return (shares ** 2).sum(axis=1)


def construir_panel(df: pd.DataFrame, min_txn: int = cfg.MIN_TXN_POR_CLIENTE) -> pd.DataFrame:
    """Agrega transacciones a un registro por tarjetahabiente."""
    # --- Composicion del gasto por categoria (share sobre el monto) ---------
    monto_cat = (
        df.pivot_table(index="cc_num", columns="category", values="amt",
                       aggfunc="sum", observed=False)
        .reindex(columns=cfg.CATEGORIAS)
        .fillna(0.0)
    )
    shares = monto_cat.div(monto_cat.sum(axis=1), axis=0)
    shares.columns = COLS_SHARE

    # --- Metricas de intensidad y ritmo -------------------------------------
    g = df.groupby("cc_num", observed=True)
    base = g.agg(
        n_txn=("amt", "size"),
        gasto_total=("amt", "sum"),
        ticket_medio=("amt", "mean"),
        ticket_mediano=("amt", "median"),
        log_ticket_medio=("log_amt", "mean"),
        dispersion_log_ticket=("log_amt", "std"),
        share_finde=("es_finde", "mean"),
        share_nocturna=("es_nocturna", "mean"),
        share_redondo=("monto_redondo", "mean"),
        edad=("edad", "mean"),
        log_city_pop=("log_city_pop", "mean"),
        n_comercios=("merchant", "nunique"),
        primera_txn=("trans_date_trans_time", "min"),
        ultima_txn=("trans_date_trans_time", "max"),
    )

    # --- Composicion por tipo de gasto y canal ------------------------------
    for col, campo, valores in [
        ("share_hedonico", "tipo_gasto", "hedonico"),
        ("share_utilitario", "tipo_gasto", "utilitario"),
        ("share_online", "canal", "online"),
    ]:
        monto_grupo = df[df[campo].astype(str) == valores].groupby("cc_num", observed=True)["amt"].sum()
        base[col] = (monto_grupo / base["gasto_total"]).fillna(0.0)

    # --- Atributos demograficos (constantes por cliente) --------------------
    demo = g[["gender", "state", "job", "city"]].first()

    # --- Volatilidad mensual del gasto --------------------------------------
    mensual = df.groupby(["cc_num", "anio_mes"], observed=True)["amt"].sum()
    vol = mensual.groupby("cc_num", observed=True).agg(["mean", "std"])
    base["cv_gasto_mensual"] = (vol["std"] / vol["mean"]).fillna(0.0)

    panel = base.join(demo).join(shares)

    # --- Derivadas -----------------------------------------------------------
    meses = ((panel["ultima_txn"] - panel["primera_txn"]).dt.days / 30.44).clip(lower=1)
    panel["txn_por_mes"] = panel["n_txn"] / meses
    panel["gasto_por_mes"] = panel["gasto_total"] / meses
    panel["hhi_categorias"] = _hhi(panel[COLS_SHARE].to_numpy())
    panel["diversificacion"] = 1 - panel["hhi_categorias"]
    panel["lealtad_comercio"] = 1 - panel["n_comercios"] / panel["n_txn"]

    antes = len(panel)
    panel = panel[panel["n_txn"] >= min_txn].copy()
    panel.attrs["clientes_descartados"] = antes - len(panel)

    return panel


def clr(shares: pd.DataFrame) -> pd.DataFrame:
    """Transformacion log-ratio centrada.

    Las participaciones viven en el simplex: suman 1, asi que no son
    independientes y la distancia euclidiana entre ellas no tiene sentido
    geometrico. La CLR las manda a R^D, donde k-means si es valido. Los ceros
    se sustituyen de forma multiplicativa antes de tomar logaritmos.
    """
    X = shares.to_numpy(dtype=float)

    positivos = X[X > 0]
    delta = positivos.min() / 2 if positivos.size else 1e-6

    ceros = X == 0
    X = np.where(ceros, delta, X)
    # Reescala las partes no nulas para que la suma vuelva a ser 1.
    n_ceros = ceros.sum(axis=1, keepdims=True)
    X = np.where(ceros, X, X * (1 - n_ceros * delta))
    X = X / X.sum(axis=1, keepdims=True)

    logX = np.log(X)
    Z = logX - logX.mean(axis=1, keepdims=True)

    return pd.DataFrame(Z, index=shares.index,
                        columns=[c.replace("share_", "clr_") for c in shares.columns])
