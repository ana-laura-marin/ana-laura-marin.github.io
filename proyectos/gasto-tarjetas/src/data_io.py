"""Obtencion y carga de datos.

Resuelve la fuente en este orden: CSV real de Kaggle si ya esta descargado ->
descarga via API de Kaggle si hay credenciales -> datos sinteticos como
respaldo. Asi el pipeline corre siempre, con o sin token.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from . import config as cfg


def hay_credenciales_kaggle() -> bool:
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def descargar_de_kaggle(force: bool = False) -> Path | None:
    """Descarga el dataset con el CLI de Kaggle. Devuelve None si no se puede."""
    if cfg.RAW_CSV.exists() and not force:
        return cfg.RAW_CSV
    if not hay_credenciales_kaggle():
        return None
    if shutil.which("kaggle") is None:
        print("  ! El CLI de kaggle no esta instalado: pip install kaggle")
        return None

    cfg.DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"  Descargando {cfg.KAGGLE_DATASET} (~354 MB)...")
    res = subprocess.run(
        ["kaggle", "datasets", "download", "-d", cfg.KAGGLE_DATASET,
         "-p", str(cfg.DATA_RAW)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        print(f"  ! Fallo la descarga: {res.stderr.strip()[:300]}")
        return None

    for z in cfg.DATA_RAW.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(cfg.DATA_RAW)
        z.unlink()

    return cfg.RAW_CSV if cfg.RAW_CSV.exists() else None


def resolver_fuente(preferir_sintetico: bool = False) -> tuple[Path, str]:
    """Devuelve (ruta, origen) donde origen es 'kaggle' o 'sintetico'."""
    if not preferir_sintetico:
        ruta = descargar_de_kaggle()
        if ruta is not None:
            return ruta, "kaggle"
        print("  Sin CSV de Kaggle ni credenciales: se usan datos sinteticos.")

    if not cfg.SYNTHETIC_CSV.exists():
        from . import synthetic
        synthetic.escribir()
    return cfg.SYNTHETIC_CSV, "sintetico"


def cargar(ruta: Path, nrows: int | None = None) -> pd.DataFrame:
    """Carga solo las columnas del contrato, con dtypes economicos en memoria."""
    disponibles = pd.read_csv(ruta, nrows=0).columns.tolist()

    faltantes = [c for c in cfg.COLUMNAS_REQUERIDAS if c not in disponibles]
    if faltantes:
        raise ValueError(
            f"El archivo {ruta.name} no tiene las columnas requeridas: {faltantes}. "
            f"Columnas encontradas: {disponibles}"
        )

    usecols = [c for c in cfg.COLUMNAS_REQUERIDAS if c in disponibles]
    if "_segmento_real" in disponibles:   # solo presente en los sinteticos
        usecols.append("_segmento_real")

    df = pd.read_csv(
        ruta,
        usecols=usecols,
        dtype={k: v for k, v in cfg.DTYPES.items() if k in usecols},
        parse_dates=["trans_date_trans_time", "dob"],
        nrows=nrows,
    )
    return df
