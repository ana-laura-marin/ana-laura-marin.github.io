"""Segmentacion de tarjetahabientes por composicion de gasto."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config as cfg
from .features import COLS_SHARE, clr


def elegir_k(Z: np.ndarray, k_min: int = cfg.K_MIN, k_max: int = cfg.K_MAX,
             semilla: int = cfg.RANDOM_STATE) -> tuple[int, pd.DataFrame]:
    """Barre k y elige por silueta. Devuelve (k elegido, tabla de diagnostico)."""
    filas = []
    # La silueta sobre muchas observaciones es cara; se estima en una muestra.
    idx = np.arange(len(Z))
    if len(Z) > 8000:
        idx = np.random.default_rng(semilla).choice(len(Z), 8000, replace=False)

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=semilla)
        etiquetas = km.fit_predict(Z)
        filas.append({
            "k": k,
            "inercia": km.inertia_,
            "silueta": silhouette_score(Z[idx], etiquetas[idx]),
            "calinski_harabasz": calinski_harabasz_score(Z, etiquetas),
        })

    diag = pd.DataFrame(filas)
    mejor = int(diag.loc[diag["silueta"].idxmax(), "k"])
    return mejor, diag


def segmentar(panel: pd.DataFrame, k: int | None = cfg.K_FIJO,
              semilla: int = cfg.RANDOM_STATE) -> dict:
    """Corre la segmentacion completa sobre las participaciones CLR."""
    Z_df = clr(panel[COLS_SHARE])
    Z = StandardScaler().fit_transform(Z_df.to_numpy())

    if k is None:
        k, diag = elegir_k(Z, semilla=semilla)
    else:
        diag = None

    km = KMeans(n_clusters=k, n_init=20, random_state=semilla)
    etiquetas = km.fit_predict(Z)

    pca = PCA(n_components=2, random_state=semilla)
    coords = pca.fit_transform(Z)

    panel = panel.copy()
    panel["segmento"] = etiquetas
    panel["pc1"], panel["pc2"] = coords[:, 0], coords[:, 1]

    return {
        "panel": panel,
        "k": k,
        "diagnostico_k": diag,
        "varianza_pca": pca.explained_variance_ratio_,
        "modelo": km,
        "clr": Z_df,
    }


def perfilar(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Caracteriza cada segmento frente al promedio poblacional."""
    g = panel.groupby("segmento", observed=True)

    # Indice de sobre/sub representacion: 100 = igual al promedio general.
    medias = g[COLS_SHARE].mean()
    lift = (medias / panel[COLS_SHARE].mean() * 100).round(1)
    lift.columns = [cfg.ETIQUETAS_ES[c.replace("share_", "")] for c in lift.columns]

    metricas = [
        "gasto_por_mes", "ticket_medio", "txn_por_mes", "edad", "log_city_pop",
        "share_hedonico", "share_utilitario", "share_online", "share_finde",
        "share_nocturna", "share_redondo", "diversificacion",
        "lealtad_comercio", "cv_gasto_mensual",
    ]
    resumen = g[metricas].mean().round(3)
    resumen.insert(0, "n_clientes", g.size())
    resumen.insert(1, "pct_cartera", (g["gasto_total"].sum() / panel["gasto_total"].sum() * 100).round(1))

    composicion = pd.DataFrame({
        "pct_femenino": g["gender"].apply(lambda s: (s.astype(str) == "F").mean() * 100).round(1),
        "estado_top": g["state"].apply(lambda s: s.astype(str).mode().iat[0] if len(s) else None),
        "ocupacion_top": g["job"].apply(lambda s: s.astype(str).mode().iat[0] if len(s) else None),
    })

    return {
        "resumen": resumen.join(composicion),
        "lift_categorias": lift,
        "shares_medios": medias.round(4),
    }


def nombrar(lift: pd.DataFrame, resumen: pd.DataFrame) -> dict[int, str]:
    """Etiqueta cada segmento con las dos categorias donde mas sobre-indexa."""
    nombres = {}
    for seg in lift.index:
        top = lift.loc[seg].nlargest(2).index.tolist()
        intensidad = "alto gasto" if resumen.loc[seg, "gasto_por_mes"] > resumen["gasto_por_mes"].median() else "gasto moderado"
        nombres[seg] = f"{top[0]} + {top[1]} ({intensidad})"
    return nombres
