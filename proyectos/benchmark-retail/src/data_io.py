"""Carga de los CSV de Favorita y construccion del panel mensual.

El panel base es mes x tienda x familia; de ahi se derivan las vistas por plaza
(ciudad x familia x mes) que usa el benchmark.
"""

from __future__ import annotations

import pandas as pd

from . import config as cfg


def _ruta_cache(nombre: str):
    cfg.DERIVADOS.mkdir(parents=True, exist_ok=True)
    return cfg.DERIVADOS / f"{nombre}.parquet"


def cargar_tiendas() -> pd.DataFrame:
    return pd.read_csv(cfg.CRUDOS / "stores.csv")


def cargar_feriados() -> pd.DataFrame:
    return pd.read_csv(cfg.CRUDOS / "holidays_events.csv", parse_dates=["date"])


def cargar_petroleo() -> pd.DataFrame:
    oil = pd.read_csv(cfg.CRUDOS / "oil.csv", parse_dates=["date"])
    oil = oil.set_index("date").asfreq("D").ffill().bfill().reset_index()
    return oil.rename(columns={"dcoilwtico": "wti"})


def cargar_transacciones() -> pd.DataFrame:
    return pd.read_csv(cfg.CRUDOS / "transactions.csv", parse_dates=["date"])


def cargar_ventas_diarias() -> pd.DataFrame:
    """train.csv crudo, recortado a la ventana de analisis."""
    tr = pd.read_csv(cfg.CRUDOS / "train.csv", parse_dates=["date"])
    return tr[(tr.date >= cfg.INICIO) & (tr.date <= cfg.FIN)].copy()


def construir_panel(forzar: bool = False) -> pd.DataFrame:
    """Panel mensual tienda x familia con volumen, promocion e ingreso.

    Se cachea en parquet porque agregar 3 millones de filas toma varios segundos.
    """
    cache = _ruta_cache("panel_tienda")
    if cache.exists() and not forzar:
        return pd.read_parquet(cache)

    tr = cargar_ventas_diarias()
    tr["mes"] = tr.date.values.astype("datetime64[M]")

    panel = (tr.groupby(["mes", "store_nbr", "family"], observed=True)
               .agg(unidades=("sales", "sum"),
                    dias=("sales", "size"),
                    dias_promo=("onpromotion", lambda s: int((s > 0).sum())),
                    items_promo=("onpromotion", "sum"))
               .reset_index())

    # Una tienda-mes sin una sola venta en ninguna familia es tienda cerrada
    # (aperturas posteriores a 2013): se marca para excluirla de los promedios.
    abierta = (panel.groupby(["mes", "store_nbr"], observed=True).unidades
                    .transform("sum") > 0)
    panel["abierta"] = abierta

    panel = panel.merge(cargar_tiendas(), on="store_nbr", how="left")
    panel["precio"] = panel.family.map(cfg.PRECIO_UNITARIO)
    panel["margen"] = panel.family.map(cfg.MARGEN_BRUTO)
    panel["ingreso"] = panel.unidades * panel.precio
    panel["contribucion"] = panel.ingreso * panel.margen
    panel["intensidad_promo"] = panel.dias_promo / panel.dias

    panel.to_parquet(cache, index=False)
    return panel


def panel_plaza(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Agrega el panel a ciudad x familia x mes (la unidad del benchmark)."""
    panel = construir_panel() if panel is None else panel
    p = panel[panel.abierta]
    plaza = (p.groupby(["mes", "city", "family"], observed=True)
              .agg(unidades=("unidades", "sum"),
                   ingreso=("ingreso", "sum"),
                   contribucion=("contribucion", "sum"),
                   tiendas=("store_nbr", "nunique"),
                   intensidad_promo=("intensidad_promo", "mean"))
              .reset_index())
    return plaza
