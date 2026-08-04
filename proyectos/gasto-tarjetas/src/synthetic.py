"""Generador de datos sinteticos con el mismo esquema que el CSV de Kaggle.

Sirve para dos cosas: (1) poder correr y probar todo el pipeline antes de tener
el archivo real descargado, y (2) tener un caso donde la estructura latente es
conocida, asi se puede verificar que la segmentacion la recupera.

Se construyen 4 segmentos latentes con propensiones de gasto distintas y
demografia correlacionada, de modo que tanto el clustering como las regresiones
tengan senal real que encontrar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

# Perfiles latentes: peso relativo de cada categoria y demografia asociada.
SEGMENTOS = {
    "digitales_urbanos": {
        "peso": 0.28,
        "edad": (22, 34),
        "log_city_pop": (11.5, 1.0),
        "p_femenino": 0.52,
        "gasto_base": 3.6,          # media del log del monto
        "propension": {
            "shopping_net": 4.0, "misc_net": 2.5, "grocery_net": 2.2,
            "entertainment": 3.0, "food_dining": 2.8, "personal_care": 1.8,
        },
    },
    "familias_suburbanas": {
        "peso": 0.30,
        "edad": (34, 52),
        "log_city_pop": (9.0, 1.2),
        "p_femenino": 0.55,
        "gasto_base": 3.9,
        "propension": {
            "kids_pets": 4.2, "grocery_pos": 4.0, "home": 3.0,
            "gas_transport": 2.8, "health_fitness": 1.8, "shopping_pos": 2.0,
        },
    },
    "viajeros_alto_gasto": {
        "peso": 0.17,
        "edad": (38, 60),
        "log_city_pop": (12.2, 0.9),
        "p_femenino": 0.46,
        "gasto_base": 4.5,
        "propension": {
            "travel": 4.5, "food_dining": 3.4, "entertainment": 2.6,
            "shopping_pos": 2.4, "shopping_net": 2.0, "personal_care": 1.6,
        },
    },
    "conservadores_maduros": {
        "peso": 0.25,
        "edad": (55, 80),
        "log_city_pop": (8.2, 1.3),
        "p_femenino": 0.53,
        "gasto_base": 3.4,
        "propension": {
            "grocery_pos": 4.4, "gas_transport": 3.2, "health_fitness": 3.0,
            "home": 2.6, "misc_pos": 2.0, "kids_pets": 1.2,
        },
    },
}

# Monto tipico por categoria (multiplicador sobre el gasto base del cliente).
MULT_MONTO = {
    "travel": 4.0, "shopping_net": 1.6, "shopping_pos": 1.5, "home": 1.4,
    "entertainment": 1.0, "food_dining": 0.8, "grocery_pos": 1.1,
    "grocery_net": 1.0, "gas_transport": 0.7, "health_fitness": 1.2,
    "kids_pets": 0.9, "personal_care": 0.6, "misc_net": 1.0, "misc_pos": 0.9,
}

OCUPACIONES = [
    "Accountant", "Nurse", "Software engineer", "Teacher", "Sales manager",
    "Civil engineer", "Chef", "Electrician", "Financial analyst", "Pharmacist",
    "Architect", "Journalist", "Economist", "Retail assistant", "Truck driver",
]

ESTADOS = ["CA", "TX", "NY", "FL", "PA", "OH", "IL", "GA", "NC", "MI"]


def _propension_a_probabilidades(propension: dict, rng: np.random.Generator) -> np.ndarray:
    """Convierte pesos por categoria en un vector de probabilidad sobre las 14.

    Las categorias no listadas reciben un piso bajo para que ningun cliente
    tenga cero absoluto en una categoria (los ceros estructurales complican la
    transformacion log-ratio despues).
    """
    alpha = np.array([propension.get(c, 0.35) for c in cfg.CATEGORIAS], dtype=float)
    # Dirichlet: introduce heterogeneidad individual dentro de cada segmento.
    return rng.dirichlet(alpha * 3.0)


def generar(n_clientes: int = 600, txn_por_cliente: int = 260,
            semilla: int = cfg.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)

    nombres_seg = list(SEGMENTOS)
    pesos = np.array([SEGMENTOS[s]["peso"] for s in nombres_seg])
    pesos = pesos / pesos.sum()
    asignacion = rng.choice(len(nombres_seg), size=n_clientes, p=pesos)

    inicio = pd.Timestamp("2019-01-01")
    rango_seg = int((pd.Timestamp("2020-12-31") - inicio).total_seconds())

    filas = []
    for i in range(n_clientes):
        seg = SEGMENTOS[nombres_seg[asignacion[i]]]
        cc_num = 4_000_000_000_000_000 + i * 7919

        edad = rng.uniform(*seg["edad"])
        dob = pd.Timestamp("2020-01-01") - pd.Timedelta(days=int(edad * 365.25))
        genero = "F" if rng.random() < seg["p_femenino"] else "M"
        city_pop = int(np.clip(np.exp(rng.normal(*seg["log_city_pop"])), 150, 3_000_000))
        estado = ESTADOS[rng.integers(len(ESTADOS))]
        ciudad = f"{estado}_city_{rng.integers(1, 40)}"
        ocupacion = OCUPACIONES[rng.integers(len(OCUPACIONES))]

        probs = _propension_a_probabilidades(seg["propension"], rng)
        n_txn = max(15, int(rng.normal(txn_por_cliente, txn_por_cliente * 0.25)))
        cats = rng.choice(cfg.CATEGORIAS, size=n_txn, p=probs)

        # Monto: lognormal centrada en el gasto base del cliente, escalada por
        # el multiplicador de la categoria.
        mu = seg["gasto_base"] + np.log([MULT_MONTO[c] for c in cats])
        montos = np.round(np.exp(rng.normal(mu, 0.55)), 2)

        # Bunching en montos redondos: mas frecuente en gasto utilitario, que
        # se planifica y suele pagarse en cifras cerradas. Sin esto la metrica
        # de numero redondo sale plana y no habria nada que detectar.
        p_redondo = np.where(
            [cfg.tipo_gasto(c) == "utilitario" for c in cats], 0.18, 0.06
        )
        snap = rng.random(n_txn) < p_redondo
        montos = np.where(snap, np.maximum(np.round(montos / 5) * 5, 5.0), montos)

        # Hora del dia: el gasto hedonico se concentra en la tarde/noche.
        # La hora es circular, asi que se envuelve modulo 24 en vez de recortar:
        # recortar apila masa artificial en 00:00 y 23:59.
        es_hedonico = np.array([cfg.tipo_gasto(c) == "hedonico" for c in cats])
        horas = np.where(
            es_hedonico,
            rng.normal(19.5, 3.2, n_txn),
            rng.normal(13.5, 3.8, n_txn),
        ) % 24
        segundos = (rng.integers(0, rango_seg, n_txn) // 86400) * 86400 + (horas * 3600).astype(int)
        ts = inicio + pd.to_timedelta(segundos, unit="s")

        filas.append(pd.DataFrame({
            "trans_date_trans_time": ts,
            "cc_num": cc_num,
            "merchant": [f"fraud_{c}_{rng.integers(1, 12)}" for c in cats],
            "category": cats,
            "amt": montos,
            "gender": genero,
            "city": ciudad,
            "state": estado,
            "city_pop": city_pop,
            "job": ocupacion,
            "dob": dob,
            # Fraude raro y sesgado hacia montos altos, como en el dataset real.
            "is_fraud": (rng.random(n_txn) < 0.004 * (montos > np.exp(seg["gasto_base"]) * 3)).astype(int),
            "_segmento_real": nombres_seg[asignacion[i]],
        }))

    df = pd.concat(filas, ignore_index=True)
    return df.sort_values("trans_date_trans_time").reset_index(drop=True)


def escribir(path=cfg.SYNTHETIC_CSV, **kwargs) -> "pd.DataFrame":
    df = generar(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    d = escribir()
    print(f"Escritas {len(d):,} transacciones de {d.cc_num.nunique()} clientes -> {cfg.SYNTHETIC_CSV}")
