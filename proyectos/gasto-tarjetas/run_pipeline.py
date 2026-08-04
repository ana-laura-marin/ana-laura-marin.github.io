"""Orquestador del pipeline completo.

    python run_pipeline.py                 # usa Kaggle si hay token, si no sinteticos
    python run_pipeline.py --sintetico     # fuerza datos sinteticos
    python run_pipeline.py --nrows 200000  # submuestra para iterar rapido
    python run_pipeline.py --k 5           # fija el numero de segmentos
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from src import config as cfg
from src import data_io, econometrics as econ, features, prep, segment, viz


def _titulo(txt: str) -> None:
    print(f"\n{'=' * 66}\n{txt}\n{'=' * 66}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintetico", action="store_true",
                    help="Usa datos sinteticos aunque haya CSV de Kaggle")
    ap.add_argument("--nrows", type=int, default=None,
                    help="Limita filas leidas del CSV")
    ap.add_argument("--k", type=int, default=cfg.K_FIJO,
                    help="Numero de segmentos (por defecto se elige por silueta)")
    ap.add_argument("--min-txn", type=int, default=cfg.MIN_TXN_POR_CLIENTE)
    args = ap.parse_args()

    t0 = time.time()
    cfg.TABLES.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- datos
    _titulo("1. Datos")
    ruta, origen = data_io.resolver_fuente(preferir_sintetico=args.sintetico)
    print(f"  Fuente: {origen}  ->  {ruta.name}")
    df = data_io.cargar(ruta, nrows=args.nrows)
    print(f"  {len(df):,} transacciones, {df['cc_num'].nunique():,} tarjetahabientes")

    # -------------------------------------------------------------- limpieza
    _titulo("2. Limpieza y enriquecimiento")
    df, extra = prep.limpiar(df)
    for k, v in extra["log"].items():
        print(f"  {k:.<32} {v:,}")
    df = prep.enriquecer(df)
    print(f"  Rango temporal: {df['trans_date_trans_time'].min():%Y-%m-%d} "
          f"a {df['trans_date_trans_time'].max():%Y-%m-%d}")

    # ------------------------------------------------------- panel de clientes
    _titulo("3. Panel de tarjetahabientes")
    panel = features.construir_panel(df, min_txn=args.min_txn)
    print(f"  {len(panel):,} clientes con >= {args.min_txn} transacciones "
          f"({panel.attrs['clientes_descartados']:,} descartados)")
    print(f"  Gasto mensual mediano: {panel['gasto_por_mes'].median():,.0f}")
    print(f"  Participacion hedonica media: {panel['share_hedonico'].mean():.1%}")

    # ---------------------------------------------------------- segmentacion
    _titulo("4. Segmentacion")
    res = segment.segmentar(panel, k=args.k)
    panel = res["panel"]
    if res["diagnostico_k"] is not None:
        print(res["diagnostico_k"].round(3).to_string(index=False))
    print(f"\n  k elegido: {res['k']}   "
          f"varianza CP1+CP2: {res['varianza_pca'][:2].sum():.1%}")

    perfil = segment.perfilar(panel)
    nombres = segment.nombrar(perfil["lift_categorias"], perfil["resumen"])
    print("\n  Perfil de segmentos:")
    print(perfil["resumen"].to_string())
    print("\n  Especializacion (100 = promedio de la cartera):")
    print(perfil["lift_categorias"].to_string())

    # Si los datos son sinteticos, se puede auditar contra la verdad conocida.
    if "_segmento_real" in df.columns:
        real = df.groupby("cc_num", observed=True)["_segmento_real"].first()
        cruce = pd.crosstab(
            pd.Series(panel.index.map(real), index=panel.index, name="segmento_latente"),
            panel["segmento"],
        )
        # Pureza: fraccion de clientes en la clase latente dominante de su cluster.
        pureza = cruce.max(axis=0).sum() / cruce.to_numpy().sum()
        print(f"\n  Validacion contra segmentos latentes reales (pureza = {pureza:.1%}):")
        print(cruce.to_string())
        if res["k"] < cruce.shape[0]:
            print(f"  Nota: la silueta eligio k={res['k']} frente a {cruce.shape[0]} "
                  f"grupos latentes. Es el sesgo tipico de la silueta hacia\n"
                  f"  particiones gruesas: separa el eje dominante y colapsa los "
                  f"matices. Correr con --k {cruce.shape[0]} para comparar.")

    # ---------------------------------------------------------- econometria
    _titulo("5. Determinantes del gasto (logit fraccional)")
    datos = econ.preparar(panel)
    res_hed, ef_hed = econ.logit_fraccional(datos, "share_hedonico")
    print(f"  Modelo principal: share_hedonico ~ demografia + intensidad + EF estado")
    print(f"  n = {int(res_hed.nobs):,}   pseudo-R2 (dev) = "
          f"{1 - res_hed.deviance / res_hed.null_deviance:.3f}")
    print("\n  Efectos parciales promedio (puntos de participacion):")
    print(ef_hed.loc[[i for i in econ.REGRESORES_CLAVE if i in ef_hed.index]]
          .round(5).to_string())

    ape_largo = econ.tabla_por_categoria(datos)
    tabla_fmt = econ.formatear_tabla(ape_largo)
    print("\n  Efectos parciales por categoria:")
    print(tabla_fmt.to_string())

    print("\n  Contrastes conductuales:")
    conductual = econ.contraste_conductual(df)
    print(conductual.to_string())

    # ------------------------------------------------------------- salidas
    _titulo("6. Salidas")
    panel.to_parquet(cfg.DATA_PROCESSED / "panel_clientes.parquet")
    perfil["resumen"].to_csv(cfg.TABLES / "perfil_segmentos.csv")
    perfil["lift_categorias"].to_csv(cfg.TABLES / "lift_segmentos.csv")
    ape_largo.to_csv(cfg.TABLES / "efectos_parciales.csv", index=False)
    tabla_fmt.to_csv(cfg.TABLES / "efectos_parciales_formato.csv")
    conductual.to_csv(cfg.TABLES / "contraste_conductual.csv")

    rutas = viz.generar_todas(
        df, panel, perfil, res["diagnostico_k"], res["k"],
        res["varianza_pca"], nombres, ape_largo,
    )
    for r in rutas:
        print(f"  figura -> {r}")
    print(f"  tablas -> {cfg.TABLES}")
    print(f"\nListo en {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
