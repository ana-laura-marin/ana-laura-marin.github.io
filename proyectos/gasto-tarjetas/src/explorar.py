"""Perfilado de la base: qué hay adentro, antes de analizar nada.

Es el paso cero. Responde qué columnas existen, de qué tipo, cuánto falta,
cuánta variedad tiene cada una y en qué rangos se mueve — sin supuestos sobre
lo que el análisis vaya a necesitar después.

    python -m src.explorar
    python -m src.explorar --nrows 200000
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as cfg


def perfil_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por columna: tipo, faltantes, cardinalidad, rango y ejemplo."""
    filas = []
    for col in df.columns:
        s = df[col]
        n_nulos = int(s.isna().sum())

        if pd.api.types.is_numeric_dtype(s):
            rango = f"{s.min():,.2f} a {s.max():,.2f}"
        elif pd.api.types.is_datetime64_any_dtype(s):
            rango = f"{s.min():%Y-%m-%d} a {s.max():%Y-%m-%d}"
        else:
            rango = ""

        no_nulos = s.dropna()
        ejemplo = str(no_nulos.iloc[0])[:38] if len(no_nulos) else ""

        filas.append({
            "columna": col,
            "tipo": str(s.dtype),
            "nulos": n_nulos,
            "pct_nulos": round(n_nulos / len(df) * 100, 2),
            "unicos": int(s.nunique(dropna=True)),
            "rango": rango,
            "ejemplo": ejemplo,
        })
    return pd.DataFrame(filas).set_index("columna")


def resumen_montos(df: pd.DataFrame) -> pd.Series:
    """Percentiles del monto: dónde está la masa y cuán larga es la cola."""
    q = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 0.999]
    out = df["amt"].quantile(q)
    out.index = [f"p{int(v * 1000) / 10:g}" for v in q]
    out["media"] = df["amt"].mean()
    out["max"] = df["amt"].max()
    return out.round(2)


def resumen_categorias(df: pd.DataFrame) -> pd.DataFrame:
    """Volumen y valor por categoría, que es la dimensión del proyecto."""
    g = df.groupby("category", observed=True)["amt"]
    out = pd.DataFrame({
        "transacciones": g.size(),
        "monto_total": g.sum(),
        "ticket_medio": g.mean(),
        "ticket_mediano": g.median(),
    })
    out["pct_transacciones"] = out["transacciones"] / out["transacciones"].sum() * 100
    out["pct_monto"] = out["monto_total"] / out["monto_total"].sum() * 100
    out.index = [cfg.ETIQUETAS_ES.get(str(i), str(i)) for i in out.index]
    return out.sort_values("pct_monto", ascending=False).round(2)


def resumen_clientes(df: pd.DataFrame) -> pd.Series:
    """Cuántas observaciones hay por tarjetahabiente: define qué se puede estimar."""
    por_cliente = df.groupby("cc_num", observed=True).size()
    return pd.Series({
        "clientes": por_cliente.size,
        "txn_min": por_cliente.min(),
        "txn_p25": por_cliente.quantile(0.25),
        "txn_mediana": por_cliente.median(),
        "txn_p75": por_cliente.quantile(0.75),
        "txn_max": por_cliente.max(),
        "clientes_con_menos_de_30": int((por_cliente < 30).sum()),
    }).round(1)


def integridad(df: pd.DataFrame) -> pd.Series:
    """Chequeos que deciden si hay que limpiar antes de analizar."""
    dup = df.duplicated(subset=["cc_num", "trans_date_trans_time", "amt", "merchant"]).sum()
    edad = (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25
    cat_desconocidas = (~df["category"].astype(str).isin(cfg.CATEGORIAS)).sum()

    return pd.Series({
        "filas": len(df),
        "duplicados_exactos": int(dup),
        "montos_no_positivos": int((df["amt"] <= 0).sum()),
        "categorias_fuera_de_taxonomia": int(cat_desconocidas),
        "edad_menor_18": int((edad < 18).sum()),
        "edad_mayor_100": int((edad > 100).sum()),
        "transacciones_fraude": int(df["is_fraud"].sum()),
        "pct_fraude": round(df["is_fraud"].mean() * 100, 3),
    })


def cobertura_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Transacciones por mes: detecta huecos y estacionalidad antes de modelar."""
    m = df.set_index("trans_date_trans_time").resample("MS").agg(
        transacciones=("amt", "size"), monto=("amt", "sum")
    )
    m.index = m.index.strftime("%Y-%m")
    return m.round(0)


def informe(df: pd.DataFrame, origen: str = "") -> None:
    def bloque(titulo: str) -> None:
        print(f"\n{'-' * 74}\n{titulo}\n{'-' * 74}")

    print(f"\n{'=' * 74}\nPERFIL DE LA BASE{f'  ({origen})' if origen else ''}\n{'=' * 74}")
    print(f"{len(df):,} filas  x  {df.shape[1]} columnas  "
          f"|  {df.memory_usage(deep=True).sum() / 1e6:,.0f} MB en memoria")

    bloque("Columnas")
    print(perfil_columnas(df).to_string())

    bloque("Integridad")
    print(integridad(df).to_string())

    bloque("Distribucion del monto")
    print(resumen_montos(df).to_string())

    bloque("Categorias de gasto")
    print(resumen_categorias(df).to_string())

    bloque("Tarjetahabientes")
    print(resumen_clientes(df).to_string())

    bloque("Cobertura temporal")
    print(cobertura_temporal(df).to_string())


def main() -> None:
    from . import data_io

    ap = argparse.ArgumentParser()
    ap.add_argument("--sintetico", action="store_true")
    ap.add_argument("--nrows", type=int, default=None)
    args = ap.parse_args()

    ruta, origen = data_io.resolver_fuente(preferir_sintetico=args.sintetico)
    df = data_io.cargar(ruta, nrows=args.nrows)
    informe(df, origen=f"fuente: {origen}")


if __name__ == "__main__":
    main()
