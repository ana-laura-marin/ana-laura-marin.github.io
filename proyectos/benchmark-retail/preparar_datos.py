"""Extrae y valida el ZIP de la competencia Store Sales (Corporacion Favorita).

    python preparar_datos.py                       # busca el ZIP en data/raw y en Descargas
    python preparar_datos.py "C:/ruta/al/store.zip" # ruta explicita

Deja los CSV en data/raw/ e imprime un resumen para verificar que estan completos.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
CRUDOS = RAIZ / "data" / "raw"

ESPERADOS = {
    "train.csv": ["id", "date", "store_nbr", "family", "sales", "onpromotion"],
    "stores.csv": ["store_nbr", "city", "state", "type", "cluster"],
    "transactions.csv": ["date", "store_nbr", "transactions"],
    "oil.csv": ["date", "dcoilwtico"],
    "holidays_events.csv": ["date", "type", "locale", "locale_name",
                            "description", "transferred"],
}


def localizar_zip(argumento: str | None) -> Path:
    """Devuelve la ruta del ZIP: el argumento, o el candidato mas reciente."""
    if argumento:
        ruta = Path(argumento).expanduser()
        if not ruta.exists():
            raise SystemExit(f"No existe el archivo: {ruta}")
        return ruta

    candidatos: list[Path] = list(CRUDOS.glob("*.zip"))
    descargas = Path.home() / "Downloads"
    if descargas.exists():
        candidatos += [p for p in descargas.glob("*.zip")
                       if "store" in p.name.lower() or "sales" in p.name.lower()
                       or "favorita" in p.name.lower()]
    if not candidatos:
        raise SystemExit(
            "No encontre el ZIP. Descargalo de\n"
            "  https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data\n"
            f"y dejalo en {CRUDOS} o pasa la ruta como argumento."
        )
    return max(candidatos, key=lambda p: p.stat().st_mtime)


def extraer(zip_path: Path) -> None:
    CRUDOS.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        nombres = [n for n in z.namelist() if n.lower().endswith(".csv")]
        print(f"ZIP: {zip_path}")
        print(f"CSV dentro: {', '.join(sorted(nombres))}\n")
        for n in nombres:
            destino = CRUDOS / Path(n).name
            with z.open(n) as origen, open(destino, "wb") as salida:
                salida.write(origen.read())


def resumir() -> None:
    faltantes = []
    for archivo, columnas in ESPERADOS.items():
        ruta = CRUDOS / archivo
        if not ruta.exists():
            faltantes.append(archivo)
            continue
        df = pd.read_csv(ruta)
        cols_faltantes = [c for c in columnas if c not in df.columns]
        print(f"{archivo:22s} {len(df):>10,} filas  {list(df.columns)}")
        if cols_faltantes:
            print(f"  ! columnas ausentes: {cols_faltantes}")
        if "date" in df.columns:
            print(f"  periodo: {df['date'].min()} a {df['date'].max()}")

    if faltantes:
        print(f"\nFaltan archivos: {', '.join(faltantes)}")
    else:
        print("\nTodos los archivos esperados estan presentes.")


if __name__ == "__main__":
    extraer(localizar_zip(sys.argv[1] if len(sys.argv) > 1 else None))
    resumir()
