"""Descomposicion de la brecha competitiva y valoracion de la oportunidad.

La idea central: que Favorita tenga poco share en una plaza no es, por si solo,
una oportunidad. Puede ser una plaza dificil. La pregunta accionable es si una
familia rinde por debajo de lo que esa misma plaza logra en el resto de su
surtido y de lo que esa misma familia logra en el resto del pais.

Se ajusta un modelo aditivo en log-odds

    logit(share_cf) = mu + alpha_ciudad + beta_familia + residuo_cf

ponderado por tamano de mercado. El residuo negativo es la brecha accionable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoide(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def ventana(bench: pd.DataFrame, meses: int = 12) -> pd.DataFrame:
    """Ultimos `meses` del panel, agregados a ciudad x familia."""
    corte = bench.mes.max() - pd.DateOffset(months=meses - 1)
    v = bench[bench.mes >= corte]
    out = (v.groupby(["city", "family"], observed=True)
            .agg(unidades=("unidades", "sum"),
                 unidades_andina=("unidades_andina", "sum"),
                 mercado_unidades=("mercado_unidades", "sum"),
                 ingreso=("ingreso", "sum"),
                 mercado_ingreso=("mercado_ingreso", "sum"),
                 tiendas=("tiendas", "max"),
                 intensidad_promo=("intensidad_promo", "mean"),
                 intensidad_promo_andina=("intensidad_promo_andina", "mean"))
            .reset_index())
    out["share"] = out.unidades / out.mercado_unidades.replace(0, np.nan)
    return out.dropna(subset=["share"])


def filtrar_material(df: pd.DataFrame, columna: str = "mercado_ingreso") -> pd.DataFrame:
    """Descarta celdas cuyo mercado es demasiado chico para ser accionable."""
    umbral = cfg.UMBRAL_VOLUMEN_MINIMO * df[columna].sum()
    return df[(df[columna] >= umbral) & (df.unidades > 0)].copy()


def descomponer(df: pd.DataFrame, peso: str = "mercado_ingreso") -> pd.DataFrame:
    """Ajusta mu + alpha_ciudad + beta_familia por minimos cuadrados ponderados.

    Anade share_esperado, residuo y la oportunidad valorizada.
    """
    d = df.copy()
    ciudades = sorted(d.city.unique())
    familias = sorted(d.family.unique())

    X = np.column_stack([
        np.ones(len(d)),
        *[(d.city == c).to_numpy(float) for c in ciudades[1:]],
        *[(d.family == f).to_numpy(float) for f in familias[1:]],
    ])
    y = _logit(d.share.to_numpy())
    w = np.sqrt(d[peso].to_numpy())

    coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    ajuste = X @ coef

    d["share_esperado"] = _sigmoide(ajuste)
    d["residuo"] = y - ajuste
    d["brecha_pp"] = 100 * (d.share_esperado - d.share)

    # Oportunidad: cerrar la brecha con el desempeno esperado, con el mercado dado.
    objetivo = np.maximum(d.share_esperado - d.share, 0)
    d["oport_unidades"] = objetivo * d.mercado_unidades
    precio = d.family.map(cfg.PRECIO_UNITARIO)
    margen = d.family.map(cfg.MARGEN_BRUTO)
    d["oport_ingreso"] = d.oport_unidades * precio
    d["oport_contribucion"] = d.oport_ingreso * margen

    efectos = dict(zip(["intercepto"] + [f"city::{c}" for c in ciudades[1:]]
                       + [f"family::{f}" for f in familias[1:]], coef))
    d.attrs["efectos"] = efectos
    d.attrs["r2"] = 1 - np.sum(w ** 2 * (y - ajuste) ** 2) / np.sum(
        w ** 2 * (y - np.average(y, weights=d[peso])) ** 2)
    return d


def residuos_mensuales(bench: pd.DataFrame, meses: int = 24) -> pd.DataFrame:
    """Repite la descomposicion mes a mes para medir persistencia."""
    corte = bench.mes.max() - pd.DateOffset(months=meses - 1)
    filas = []
    for mes, grupo in bench[bench.mes >= corte].groupby("mes", observed=True):
        g = filtrar_material(grupo.assign(
            share=grupo.unidades / grupo.mercado_unidades.replace(0, np.nan)
        ).dropna(subset=["share"]))
        if g.city.nunique() < 3 or g.family.nunique() < 3:
            continue
        res = descomponer(g)[["mes", "city", "family", "residuo", "brecha_pp"]].copy()
        res.attrs = {}  # los efectos del ajuste mensual no se propagan al concat
        filas.append(res)
    return pd.concat(filas, ignore_index=True)


def persistencia(res_mens: pd.DataFrame) -> pd.DataFrame:
    """Que tan estable en el tiempo es la brecha de cada celda.

    El t de una media sobre residuos autocorrelacionados exagera la evidencia;
    se corrige el tamano de muestra por la autocorrelacion de primer orden.
    """
    def _fila(g: pd.DataFrame) -> pd.Series:
        r = g.sort_values("mes").residuo.to_numpy()
        n = len(r)
        rho = 0.0
        if n > 3 and r.std() > 0:
            rho = float(np.corrcoef(r[:-1], r[1:])[0, 1])
            rho = np.clip(rho, -0.95, 0.95)
        n_ef = max(n * (1 - rho) / (1 + rho), 2.0)
        ee = r.std(ddof=1) / np.sqrt(n_ef) if r.std(ddof=1) > 0 else np.nan
        return pd.Series({
            "residuo_medio": r.mean(),
            "meses": n,
            "pct_meses_negativo": float((r < 0).mean()),
            "autocorrelacion": rho,
            "t_ajustado": r.mean() / ee if ee and not np.isnan(ee) else np.nan,
        })

    conteo = res_mens.groupby(["city", "family"], observed=True).residuo.transform("size")
    return (res_mens[conteo >= 6].groupby(["city", "family"], observed=True)
            .apply(_fila, include_groups=False).reset_index())


def elasticidad_promo(panel: pd.DataFrame, desde: str = "2015-01-01",
                      min_obs: int = 200) -> pd.DataFrame:
    """Elasticidad del volumen al numero de items en promocion, por familia.

    Regresion log-log con efectos fijos de tienda y de mes dentro de cada familia:
    el coeficiente se lee como el aumento porcentual de unidades ante un aumento
    de 1 % en los items promocionados, ya descontados el tamano de la tienda y la
    estacionalidad propia de esa familia. La muestra arranca en 2015 porque el
    programa promocional no existe antes de abril de 2014 y recien se generaliza
    despues.

    Sigue siendo una asociacion y no un efecto causal: la promocion se decide
    donde se espera que funcione, y ese sesgo empuja la elasticidad hacia arriba.
    """
    p = panel[panel.abierta & (panel.unidades > 0) & (panel.mes >= desde)].copy()
    p["y"] = np.log(p.unidades)
    p["x"] = np.log1p(p.items_promo)

    filas = []
    for familia, g in p.groupby("family", observed=True):
        if len(g) < min_obs or g.x.std() < 0.05:
            continue
        y = (g.y - g.groupby("store_nbr").y.transform("mean")
             - g.groupby("mes").y.transform("mean") + g.y.mean())
        x = (g.x - g.groupby("store_nbr").x.transform("mean")
             - g.groupby("mes").x.transform("mean") + g.x.mean())
        if x.std() < 1e-6:
            continue
        beta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
        resid = y - beta * x
        ee = float(np.sqrt(np.var(resid, ddof=1) / (np.var(x, ddof=1) * (len(g) - 2))))
        filas.append({"family": familia, "elasticidad": beta, "ee": ee,
                      "t": beta / ee if ee > 0 else np.nan,
                      # efecto de duplicar los items en promocion
                      "lift_pct": 100 * (2 ** beta - 1), "obs": len(g)})
    return pd.DataFrame(filas).sort_values("elasticidad", ascending=False)


def ranking(oport: pd.DataFrame, persist: pd.DataFrame, elast: pd.DataFrame,
            residuo_minimo: float = 0.20,
            elasticidad_util: float = 0.10) -> pd.DataFrame:
    """Une valor, persistencia y palanca promocional en una sola tabla.

    Tres criterios, y ninguno alcanza por si solo: el dinero sin persistencia
    premia a las plazas grandes por ser grandes, y la persistencia sin dinero
    llena la lista de celdas irrelevantes.

    El tamano de la brecha se mide en log-odds y no en puntos de share, porque un
    punto de share no vale lo mismo en todas partes: donde Favorita tiene 75 % del
    mercado, perder tres puntos es un deterioro mayor que perder tres puntos donde
    tiene 35 %.
    """
    r = (oport.merge(persist, on=["city", "family"], how="left")
              .merge(elast[["family", "elasticidad", "lift_pct", "t"]].rename(
                  columns={"t": "t_promo"}), on="family", how="left"))
    r["material"] = r.oport_contribucion > 0
    r["brecha_relevante"] = r.residuo <= -residuo_minimo
    r["persistente"] = (r.t_ajustado < -1.96) & (r.pct_meses_negativo >= 0.7)
    # No basta con que la elasticidad sea distinta de cero: una respuesta de 0,03
    # es real y economicamente inservible como palanca.
    r["palanca_promo"] = (r.elasticidad >= elasticidad_util) & (r.t_promo > 2)
    r["prioritaria"] = r.material & r.brecha_relevante & r.persistente
    return r.sort_values("oport_contribucion", ascending=False)


def embudo(r: pd.DataFrame) -> pd.DataFrame:
    """Cuantas celdas y cuanto dinero sobreviven a cada criterio."""
    etapas = [
        ("Share por debajo del esperado", r.material),
        ("+ brecha de al menos 0,20 log-odds", r.material & r.brecha_relevante),
        ("+ persistente en el tiempo", r.material & r.brecha_relevante & r.persistente),
        ("+ con palanca promocional",
         r.material & r.brecha_relevante & r.persistente & r.palanca_promo),
    ]
    return pd.DataFrame([{
        "criterio": nombre,
        "celdas": int(m.sum()),
        "contribucion_usd": float(r.loc[m, "oport_contribucion"].sum()),
    } for nombre, m in etapas])


def evaluar_recuperacion(r: pd.DataFrame, sembradas: set[tuple[str, str]]) -> pd.DataFrame:
    """Contrasta las celdas priorizadas contra las ventajas realmente sembradas.

    Como el competidor es simulado, se conoce la respuesta correcta: se puede
    medir cuantas ventajas reales encuentra el metodo y cuantas inventa.
    """
    marcadas = set(map(tuple, r.loc[r.prioritaria, ["city", "family"]].to_numpy()))
    verdaderos = marcadas & sembradas
    return pd.DataFrame([{
        "sembradas": len(sembradas),
        "priorizadas": len(marcadas),
        "aciertos": len(verdaderos),
        "falsos_positivos": len(marcadas - sembradas),
        "no_detectadas": len(sembradas - marcadas),
        "precision": len(verdaderos) / len(marcadas) if marcadas else np.nan,
        "cobertura": len(verdaderos) / len(sembradas) if sembradas else np.nan,
    }])
