"""Determinantes de la composicion del gasto: logit fraccional.

La variable dependiente es una participacion en [0, 1], no un monto. Un MCO
sobre una proporcion predice fuera del soporte y asume efectos constantes en
todo el rango. El logit fraccional (Papke y Wooldridge, 1996) estima por
cuasi-maxima verosimilitud con link logit, respeta el soporte y es consistente
aunque la distribucion no sea binomial; por eso los errores estandar van
siempre robustos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from . import config as cfg
from .features import COLS_SHARE

# Especificacion base. Las continuas entran centradas para que la constante y
# los efectos parciales se lean en el cliente promedio.
CONTROLES = (
    "edad_c + I(edad_c**2) + C(gender) + log_city_pop_c "
    "+ log_gasto_mes_c + txn_por_mes_c + C(state)"
)

REGRESORES_CLAVE = ["edad_c", "C(gender)[T.M]", "log_city_pop_c", "log_gasto_mes_c"]


def preparar(panel: pd.DataFrame) -> pd.DataFrame:
    d = panel.copy()
    d["gender"] = d["gender"].astype(str)
    d["state"] = d["state"].astype(str)
    d["log_gasto_mes"] = np.log(d["gasto_por_mes"].clip(lower=1e-6))

    for col in ["edad", "log_city_pop", "log_gasto_mes", "txn_por_mes"]:
        d[f"{col}_c"] = d[col] - d[col].mean()

    # El logit fraccional admite 0 y 1, pero los extremos exactos hacen fragil
    # la convergencia; se comprimen minimamente hacia el interior.
    for c in COLS_SHARE + ["share_hedonico", "share_utilitario", "share_online"]:
        d[c] = d[c].clip(1e-6, 1 - 1e-6)

    return d


def logit_fraccional(datos: pd.DataFrame, y: str, controles: str = CONTROLES):
    """Estima el modelo y devuelve (resultados, efectos parciales promedio)."""
    modelo = smf.glm(
        formula=f"{y} ~ {controles}",
        data=datos,
        family=sm.families.Binomial(),
    )
    res = modelo.fit(cov_type="HC1")

    # Efecto parcial promedio: para link logit, dE[y]/dx = beta * mean(p(1-p)).
    p = res.fittedvalues
    escala = float(np.mean(p * (1 - p)))
    ape = res.params * escala
    ape_se = res.bse * escala

    efectos = pd.DataFrame({
        "coef": res.params,
        "ape": ape,
        "ape_se": ape_se,
        "z": res.params / res.bse,
        "p_valor": res.pvalues,
    })
    return res, efectos


def tabla_por_categoria(datos: pd.DataFrame,
                        regresores: list[str] = REGRESORES_CLAVE) -> pd.DataFrame:
    """Corre la misma especificacion en las 14 categorias y apila los EPP.

    Leer los coeficientes categoria por categoria es lo que permite decir que
    la edad no 'baja el gasto', sino que lo *recompone*: cae en unas partidas y
    sube en otras, con la suma de efectos igual a cero por construccion.

    Devuelve formato largo (una fila por categoria x regresor).
    """
    filas = []
    for cat in cfg.CATEGORIAS:
        col = f"share_{cat}"
        try:
            _, ef = logit_fraccional(datos, col)
        except Exception as exc:  # categoria degenerada o sin variacion
            print(f"  ! {cat}: no converge ({type(exc).__name__})")
            continue
        for r in regresores:
            if r not in ef.index:
                continue
            filas.append({
                "categoria": cfg.ETIQUETAS_ES[cat],
                "tipo": cfg.tipo_gasto(cat),
                "regresor": r,
                "ape": ef.loc[r, "ape"],
                "ape_se": ef.loc[r, "ape_se"],
                "p_valor": ef.loc[r, "p_valor"],
                "signif": _estrellas(ef.loc[r, "p_valor"]),
            })

    return pd.DataFrame(filas)


def formatear_tabla(largo: pd.DataFrame) -> pd.DataFrame:
    """Pivota la tabla larga a formato de presentacion (EPP con estrellas)."""
    if largo.empty:
        return largo
    d = largo.copy()
    d["celda"] = d["ape"].map(lambda v: f"{v:+.4f}") + d["signif"]
    return d.pivot(index=["tipo", "categoria"], columns="regresor", values="celda")


def _estrellas(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def contraste_conductual(df_txn: pd.DataFrame) -> pd.DataFrame:
    """Contrastes descriptivos entre gasto hedonico y utilitario.

    Tres marcadores conductuales: concentracion nocturna, sesgo de fin de
    semana y frecuencia de montos redondos (proxy de compra deliberada).
    """
    g = df_txn.groupby("tipo_gasto", observed=True)
    out = g.agg(
        n_transacciones=("amt", "size"),
        ticket_medio=("amt", "mean"),
        ticket_mediano=("amt", "median"),
        pct_nocturna=("es_nocturna", lambda s: s.mean() * 100),
        pct_finde=("es_finde", lambda s: s.mean() * 100),
        pct_monto_redondo=("monto_redondo", lambda s: s.mean() * 100),
        hora_mediana=("hora", "median"),
    ).round(2)
    return out
