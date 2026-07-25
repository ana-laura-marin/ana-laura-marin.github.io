"""Cliente ligero para descargar precios de Yahoo Finance y calcular retornos.

No depende de ``yfinance``: consume directamente la API pública *chart* de
Yahoo Finance mediante ``requests``. Devuelve retornos mensuales listos para
estimar betas por regresión, con la misma interfaz usada en el proyecto
"Costo de Capital con CAPM".
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests

_BASE = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    )
}


def _a_unix(fecha: str) -> int:
    """Convierte una fecha ``YYYY-MM-DD`` a timestamp Unix (UTC)."""
    return int(
        datetime.strptime(fecha, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )


def fetch_prices(ticker: str, start: str, end: str, interval: str = "1d") -> pd.Series:
    """Descarga el precio de cierre ajustado (dividendos y splits) de un ticker.

    Parameters
    ----------
    ticker : str
        Símbolo de Yahoo Finance (p. ej. ``"AAPL"`` o ``"^GSPC"``).
    start, end : str
        Rango de fechas en formato ``YYYY-MM-DD``.
    interval : str
        Frecuencia de los datos (``"1d"`` por defecto).

    Returns
    -------
    pandas.Series
        Serie de cierres ajustados indexada por fecha.
    """
    url = _BASE.format(ticker=ticker)
    params = {
        "period1": _a_unix(start),
        "period2": _a_unix(end),
        "interval": interval,
        "events": "div,split",
    }
    resp = requests.get(url, params=params, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    resultado = resp.json()["chart"]["result"][0]

    fechas = pd.to_datetime(resultado["timestamp"], unit="s")
    ajustado = resultado["indicators"]["adjclose"][0]["adjclose"]

    serie = pd.Series(ajustado, index=fechas, name="adjclose").dropna()
    serie.index.name = "date"
    return serie


def fetch_monthly_returns(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Retornos mensuales a partir del cierre ajustado de fin de mes.

    Resamplea los precios diarios a fin de mes y calcula la variación
    porcentual mes a mes.

    Returns
    -------
    pandas.DataFrame
        DataFrame con una columna ``return`` indexada por fin de mes.
    """
    precios = fetch_prices(ticker, start, end, interval="1d")
    mensual = precios.resample("ME").last()
    retornos = mensual.pct_change().dropna()

    df = retornos.to_frame("return")
    df.index.name = "date"
    return df


def fetch_dividends(ticker: str, start: str, end: str) -> pd.Series:
    """Descarga el histórico de dividendos por acción de un ticker.

    Consume el mismo endpoint *chart* de Yahoo Finance (``events=div``). Los
    montos vienen **ajustados por splits**, es decir, expresados en la base de
    acciones actual, por lo que la serie es directamente comparable a lo largo
    del tiempo aunque haya habido divisiones de acciones.

    Parameters
    ----------
    ticker : str
        Símbolo de Yahoo Finance (p. ej. ``"AAPL"``).
    start, end : str
        Rango de fechas en formato ``YYYY-MM-DD``.

    Returns
    -------
    pandas.Series
        Dividendo por acción indexado por fecha ex-dividendo. Serie vacía si el
        ticker no pagó dividendos en el rango.
    """
    url = _BASE.format(ticker=ticker)
    params = {
        "period1": _a_unix(start),
        "period2": _a_unix(end),
        "interval": "1d",
        "events": "div,split",
    }
    resp = requests.get(url, params=params, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    resultado = resp.json()["chart"]["result"][0]

    eventos = resultado.get("events", {}).get("dividends", {})
    if not eventos:
        return pd.Series(dtype="float64", name="dividend")

    registros = {
        pd.to_datetime(v["date"], unit="s"): v["amount"] for v in eventos.values()
    }
    serie = pd.Series(registros, name="dividend").sort_index()
    serie.index.name = "date"
    return serie
