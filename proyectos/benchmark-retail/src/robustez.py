"""Cuanto de la conclusion depende de los supuestos.

Dos preguntas distintas:

1. Los precios son inventados. Si estuvieran mal, cambiaria el ranking?
2. El competidor es simulado. Si se cambia la semilla del generador, el metodo
   sigue encontrando las mismas ventajas?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import benchmark, brechas
from . import config as cfg


def sensibilidad_precios(oport: pd.DataFrame, n: int = 400,
                         dispersion: float = 0.30,
                         semilla: int | None = None) -> pd.DataFrame:
    """Perturba el precio de cada familia y mide cuanto se mueve la oportunidad.

    No escala todos los precios a la vez (eso solo multiplica el total): sortea
    un error independiente por familia, que es lo que de verdad puede reordenar
    el ranking.
    """
    rng = np.random.default_rng(cfg.SEMILLA if semilla is None else semilla)
    familias = sorted(oport.family.unique())
    base = np.array([cfg.PRECIO_UNITARIO[f] for f in familias])
    margen = np.array([cfg.MARGEN_BRUTO[f] for f in familias])
    idx = oport.family.map({f: i for i, f in enumerate(familias)}).to_numpy()
    unidades = oport.oport_unidades.to_numpy()

    sigma = np.sqrt(np.log(1 + dispersion ** 2))
    totales = np.empty(n)
    posiciones = np.zeros((n, len(oport)), dtype=int)
    for i in range(n):
        precios = base * rng.lognormal(-0.5 * sigma ** 2, sigma, len(base))
        contrib = unidades * precios[idx] * margen[idx]
        totales[i] = contrib.sum()
        posiciones[i] = (-contrib).argsort().argsort()

    res = oport[["city", "family", "oport_contribucion"]].copy()
    res["posicion_mediana"] = np.median(posiciones, axis=0) + 1
    res["posicion_p10"] = np.percentile(posiciones, 10, axis=0) + 1
    res["posicion_p90"] = np.percentile(posiciones, 90, axis=0) + 1
    res.attrs["total_base"] = float(oport.oport_contribucion.sum())
    res.attrs["total_p10"] = float(np.percentile(totales, 10))
    res.attrs["total_p90"] = float(np.percentile(totales, 90))
    return res.sort_values("oport_contribucion", ascending=False)


def estabilidad_semillas(plaza: pd.DataFrame, panel: pd.DataFrame,
                         semillas: int = 15) -> pd.DataFrame:
    """Repite todo el analisis con distintas realizaciones del competidor.

    Las ventajas sembradas son las mismas en todas; lo que cambia es el ruido.
    Una celda que solo aparece en algunas semillas es ruido que paso el filtro.
    """
    elast = brechas.elasticidad_promo(panel)
    conteo: dict[tuple[str, str], int] = {}
    filas = []
    for k in range(semillas):
        b = benchmark.generar(plaza, semilla=cfg.SEMILLA + k)
        d = brechas.descomponer(brechas.filtrar_material(brechas.ventana(b, 12)))
        r = brechas.ranking(d, brechas.persistencia(
            brechas.residuos_mensuales(b, 24)), elast)
        ev = brechas.evaluar_recuperacion(r, benchmark.celdas_sembradas())
        ev.insert(0, "semilla", k)
        filas.append(ev)
        for celda in map(tuple, r.loc[r.prioritaria, ["city", "family"]].to_numpy()):
            conteo[celda] = conteo.get(celda, 0) + 1

    resumen = pd.concat(filas, ignore_index=True)
    frec = (pd.DataFrame([{"city": c, "family": f, "veces": v,
                           "frecuencia": v / semillas} for (c, f), v in conteo.items()])
            .sort_values("veces", ascending=False))
    frec["sembrada"] = [(c, f) in benchmark.celdas_sembradas()
                        for c, f in zip(frec.city, frec.family)]
    resumen.attrs["frecuencia"] = frec
    return resumen


def curva_umbral(oport: pd.DataFrame, persist: pd.DataFrame, elast: pd.DataFrame,
                 cortes=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)) -> pd.DataFrame:
    """Precision y cobertura del filtro segun donde se ponga el corte."""
    sembradas = benchmark.celdas_sembradas()
    filas = []
    for corte in cortes:
        r = brechas.ranking(oport, persist, elast, residuo_minimo=corte)
        ev = brechas.evaluar_recuperacion(r, sembradas).iloc[0]
        filas.append({"corte": corte, "priorizadas": ev.priorizadas,
                      "precision": ev.precision, "cobertura": ev.cobertura,
                      "contribucion": float(
                          r.loc[r.prioritaria, "oport_contribucion"].sum())})
    return pd.DataFrame(filas)
