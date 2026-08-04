"""Reducción a 2D con UMAP como paso previo al clustering por densidad.

Por qué el paso intermedio: DBSCAN agrupa por densidad local, pero en alta
dimensión las distancias se concentran —todos los pares terminan casi
equidistantes— y la densidad deja de discriminar. Aplicado directo sobre las
~26 variables del panel, DBSCAN marca casi todo como ruido. UMAP preserva la
estructura local mejor que PCA, así que la densidad del embedding sí es
informativa.

Advertencia que conviene tener presente: el embedding 2D **distorsiona** la
densidad. UMAP optimiza la topología local, no las distancias, así que la
separación entre grupos en el mapa no es proporcional a su disimilitud real, y
un embedding a 2D puede tanto fabricar grupos como disolverlos. El mapa sirve
para encontrar y ver los grupos; la interpretación de las variables se hace
siempre sobre los datos originales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config as cfg
from .features import COLS_SHARE, clr

# Comportamiento e intensidad. Se excluyen a propósito share_hedonico,
# share_utilitario y share_online: son combinaciones lineales de las 14
# participaciones y entrarían pesando la composición dos veces. Igual
# hhi_categorias, que es el complemento exacto de diversificacion.
COLS_CONDUCTA = [
    "share_finde", "share_nocturna", "share_redondo",
    "diversificacion", "lealtad_comercio", "cv_gasto_mensual",
]
COLS_INTENSIDAD = [
    "log_gasto_mes", "log_ticket_medio", "dispersion_log_ticket", "txn_por_mes",
]
COLS_DEMO = ["edad", "log_city_pop"]


def construir_matriz(panel: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Arma X_scaled: CLR de las participaciones + conducta + intensidad + demografía.

    Las participaciones entran transformadas con CLR y no crudas: suman 1, así
    que viven en el símplex y la distancia euclidiana entre ellas —que es la
    que UMAP y DBSCAN van a usar— no tiene sentido geométrico sin la
    transformación.
    """
    Z = clr(panel[COLS_SHARE])

    otras = panel.copy()
    otras["log_gasto_mes"] = np.log(otras["gasto_por_mes"].clip(lower=1e-6))

    X = pd.concat([Z, otras[COLS_CONDUCTA + COLS_INTENSIDAD + COLS_DEMO]], axis=1)

    faltantes = X.columns[X.isna().any()].tolist()
    if faltantes:
        raise ValueError(f"Hay nulos en {faltantes}; UMAP no los admite.")

    # Estandarizar es obligatorio: sin esto, txn_por_mes (decenas) domina a las
    # participaciones (centésimas) en cualquier distancia euclidiana.
    X_scaled = StandardScaler().fit_transform(X.to_numpy(dtype=float))
    return X_scaled, X.columns.tolist()


def reducir_umap(
    X_scaled: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    metric: str = "euclidean",
    random_state: int = cfg.RANDOM_STATE,
) -> tuple[np.ndarray, "object"]:
    """Reduce a 2D con UMAP. Devuelve (X_2d, modelo ajustado).

    Parámetros elegidos pensando en que después viene DBSCAN, no en que el
    dibujo quede bonito:

    - ``min_dist=0.0`` — el default de 0.1 separa puntos artificialmente para
      que el mapa se lea mejor, y eso es exactamente lo que no se quiere antes
      de un algoritmo de densidad: infla los huecos dentro de un mismo grupo.
      Con 0.0 UMAP deja que los puntos se apilen y los grumos quedan compactos.
    - ``n_neighbors`` — controla el balance entre estructura local y global.
      Bajo (5) fragmenta en muchos grupitos; alto (50) tiende a una sola masa.
      15 es un punto de partida razonable, pero con pocas filas conviene
      probar varios y ver cuál da grumos estables.
    - ``random_state`` — fijo para que el embedding sea reproducible. Tiene un
      costo: UMAP desactiva el paralelismo cuando se fija la semilla, así que
      corre más lento. Vale la pena, porque sin esto cada corrida mueve los
      grupos y ningún ``eps`` de DBSCAN se sostiene entre ejecuciones.
    """
    import umap  # import diferido: arrastra numba y tarda en cargar

    reductor = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    X_2d = reductor.fit_transform(X_scaled)
    return X_2d, reductor


def verificar(X_2d: np.ndarray, X_scaled: np.ndarray) -> pd.Series:
    """Chequeos de forma antes de pasar a DBSCAN."""
    assert X_2d.ndim == 2, f"X_2d deberia ser 2D, es {X_2d.ndim}D"
    assert X_2d.shape[1] == 2, f"X_2d deberia tener 2 columnas, tiene {X_2d.shape[1]}"
    assert X_2d.shape[0] == X_scaled.shape[0], (
        f"X_2d tiene {X_2d.shape[0]} filas y los datos originales "
        f"{X_scaled.shape[0]}: se perdieron entidades en el camino"
    )
    assert np.isfinite(X_2d).all(), "X_2d contiene NaN o infinitos"

    return pd.Series({
        "filas": X_2d.shape[0],
        "columnas_originales": X_scaled.shape[1],
        "columnas_embedding": X_2d.shape[1],
        "rango_cp1": f"{X_2d[:, 0].min():.2f} a {X_2d[:, 0].max():.2f}",
        "rango_cp2": f"{X_2d[:, 1].min():.2f} a {X_2d[:, 1].max():.2f}",
    })


# --- Calibración de eps ----------------------------------------------------
def min_samples_sugerido(n_dim: int) -> int:
    """Regla práctica: 2 × dimensiones, con piso de 4.

    ``min_samples`` es la densidad mínima que se exige para declarar una región
    densa: cuántos vecinos tiene que haber dentro de ``eps`` para que un punto
    cuente como núcleo.

    Sobre un embedding 2D la regla da 4, y en estos datos eso **no funciona**:
    con k=4 el codo cae en un eps que parte los grumos reales en 20 pedazos.
    La regla nació pensando en el espacio original de las variables, donde la
    dimensión es alta; después de reducir a 2D deja de tener sentido atarla a
    la dimensión. Ver MIN_SAMPLES_DEFAULT.
    """
    return max(4, 2 * n_dim)


# El criterio que sí funciona: min_samples es el tamaño mínimo de grupo que uno
# está dispuesto a llamar grupo. Con ~600 entidades y segmentos de cartera que
# se esperan en el orden de las centenas, 20 descarta satélites sin tocar la
# estructura. Barriendo 4..30 sobre estos datos, el codo recupera los 4 grupos
# latentes recién a partir de min_samples=20 (ARI 0.99); por debajo fragmenta.
MIN_SAMPLES_DEFAULT = 20


def curva_k_distancias(X_2d: np.ndarray, min_samples: int) -> np.ndarray:
    """Distancia de cada punto a su k-ésimo vecino, ordenada de menor a mayor.

    El k debe ser el mismo ``min_samples`` que después recibe DBSCAN: la curva
    responde exactamente la pregunta "a qué distancia está el k-ésimo vecino",
    que es la que DBSCAN se hace para decidir si un punto es núcleo. Con otro k
    el gráfico describe una densidad distinta de la que se va a usar.

    Se ordena ascendente para que el codo quede arriba a la derecha: la parte
    plana son los puntos en zonas densas y la subida final, los aislados.
    """
    from sklearn.neighbors import NearestNeighbors

    # n_neighbors = min_samples porque el primer vecino que devuelve sklearn es
    # el propio punto (distancia 0); la columna k-ésima es entonces el k-ésimo
    # vecino real.
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X_2d)
    distancias, _ = nn.kneighbors(X_2d)
    return np.sort(distancias[:, -1])


def codo(distancias: np.ndarray) -> tuple[int, float]:
    """Encuentra el codo por máxima distancia a la cuerda.

    Se traza la recta que une el primer y el último punto de la curva y se
    busca el punto que más se aleja de ella. Es determinista y no necesita
    dependencias extra. Ambos ejes se normalizan a [0,1] primero: sin eso el
    resultado dependería de las unidades del embedding.
    """
    n = len(distancias)
    x = np.linspace(0.0, 1.0, n)
    rango = distancias[-1] - distancias[0]
    y = (distancias - distancias[0]) / (rango if rango else 1.0)

    # Distancia perpendicular de cada punto a la cuerda (0,y0)-(1,y1).
    dx, dy = 1.0, y[-1] - y[0]
    norma = np.hypot(dx, dy)
    perpendicular = np.abs(dy * x - dx * (y - y[0])) / norma

    i = int(perpendicular.argmax())
    return i, float(distancias[i])


def eps_candidatos(distancias: np.ndarray, eps_codo: float,
                   factores=(0.8, 1.0, 1.25)) -> pd.DataFrame:
    """Tres eps alrededor del codo, para probar cuál da grupos más sensatos.

    El codo es una sugerencia, no una respuesta. Se separan multiplicativamente
    y no por posición en la curva: cerca del codo la curva es casi vertical, y
    unos pocos puntos de índice a cada lado dan eps tan parecidos que producen
    exactamente el mismo clustering.

    ``pct_no_nucleo`` es la fracción de entidades sin ``min_samples`` vecinos
    dentro de ese eps. Es cota superior del ruido —algunas terminarán como
    puntos frontera de un grupo— pero da la magnitud del descarte antes de
    correr nada.
    """
    filas = []
    for f in factores:
        eps = eps_codo * f
        filas.append({
            "factor": f,
            "eps": round(float(eps), 4),
            "pct_no_nucleo": round(float((distancias > eps).mean() * 100), 1),
        })
    etiquetas = ["conservador", "codo", "permisivo"][:len(filas)]
    return pd.DataFrame(filas, index=etiquetas)


# --- DBSCAN ----------------------------------------------------------------
RUIDO = -1


def agrupar(X_2d: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Corre DBSCAN y devuelve las etiquetas.

    A diferencia de k-means no se le pasa el número de grupos: DBSCAN lo
    descubre según la densidad. Los puntos que no alcanzan densidad suficiente
    quedan con etiqueta -1, que **no es un grupo**: es la marca de que no
    pertenecen a ninguno. Tratarlo como un grupo más mete en el análisis casos
    que precisamente el algoritmo identificó como atípicos.
    """
    from sklearn.cluster import DBSCAN

    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_2d)


def resumen_grupos(etiquetas: np.ndarray) -> pd.Series:
    """Cuenta grupos y ruido por separado."""
    unicas = set(etiquetas)
    n_grupos = len(unicas - {RUIDO})          # el -1 se excluye del conteo
    n_ruido = int((etiquetas == RUIDO).sum())

    # dtype=object: sin esto pandas promueve todo a float por culpa de
    # pct_ruido y los conteos se muestran como "4.0" en vez de "4".
    return pd.Series({
        "n_grupos": n_grupos,
        "n_ruido": n_ruido,
        "pct_ruido": round(n_ruido / len(etiquetas) * 100, 1),
        "n_agrupados": len(etiquetas) - n_ruido,
    }, dtype=object)


def tamanos_grupos(etiquetas: np.ndarray) -> pd.DataFrame:
    """Tamaño de cada grupo, con el ruido en una fila aparte y rotulada."""
    s = pd.Series(etiquetas).value_counts().sort_index()
    idx = ["ruido (-1)" if i == RUIDO else f"grupo {i}" for i in s.index]
    out = pd.DataFrame({"entidades": s.to_numpy()}, index=idx)
    out["pct"] = (out["entidades"] / out["entidades"].sum() * 100).round(1)
    return out


def comparar_eps(X_2d: np.ndarray, valores_eps, min_samples: int,
                 y_true=None) -> pd.DataFrame:
    """Barre varios eps y compara estabilidad de la partición.

    Si hay etiquetas verdaderas se agrega el índice de Rand ajustado, calculado
    solo sobre los puntos agrupados: incluir el ruido lo distorsiona, porque el
    -1 no es una clase sino la ausencia de clase.
    """
    from sklearn.metrics import adjusted_rand_score

    filas = []
    for eps in valores_eps:
        etiquetas = agrupar(X_2d, eps, min_samples)
        r = resumen_grupos(etiquetas).to_dict()
        r["eps"] = round(float(eps), 4)

        if y_true is not None:
            asignados = etiquetas != RUIDO
            r["ari_agrupados"] = (
                round(adjusted_rand_score(np.asarray(y_true)[asignados],
                                          etiquetas[asignados]), 3)
                if asignados.sum() else np.nan
            )
        filas.append(r)

    cols = ["eps", "n_grupos", "n_agrupados", "n_ruido", "pct_ruido"]
    if y_true is not None:
        cols.append("ari_agrupados")
    return pd.DataFrame(filas)[cols]


# --- Atípicos --------------------------------------------------------------
# Variables sobre las que se describe a los atípicos. Son las del panel
# original, no las coordenadas UMAP: el embedding sirvió para *encontrar* los
# casos, pero no tiene unidades interpretables para describirlos.
COLS_PERFIL = [
    "gasto_por_mes", "ticket_medio", "txn_por_mes", "n_txn", "n_comercios",
    "edad", "log_city_pop",
    "share_hedonico", "share_utilitario", "share_online",
    "share_finde", "share_nocturna", "share_redondo",
    "diversificacion", "lealtad_comercio", "cv_gasto_mensual",
]

# La tabla se lee desde finanzas, no desde el código.
ETIQUETAS_PERFIL = {
    "gasto_por_mes": "Gasto mensual",
    "ticket_medio": "Ticket medio",
    "txn_por_mes": "Transacciones por mes",
    "n_txn": "Transacciones totales",
    "n_comercios": "Comercios distintos",
    "edad": "Edad",
    "log_city_pop": "Tamaño de ciudad (log)",
    "share_hedonico": "% gasto hedónico",
    "share_utilitario": "% gasto utilitario",
    "share_online": "% gasto en línea",
    "share_finde": "% en fin de semana",
    "share_nocturna": "% nocturno",
    "share_redondo": "% en montos redondos",
    "diversificacion": "Diversificación de la canasta",
    "lealtad_comercio": "Lealtad al comercio",
    "cv_gasto_mensual": "Volatilidad del gasto mensual",
}


def separar_outliers(df: pd.DataFrame, etiquetas: np.ndarray
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parte el DataFrame en atípicos (label -1) y agrupados.

    No se descartan: un punto de ruido es una entidad con perfil tan singular
    que no entró en ninguna región densa, y en la práctica suele ser el caso
    más informativo del análisis —el cliente de mayor valor, el fraude, el
    error de captura—. Se separan para mirarlos, no para eliminarlos.
    """
    if len(etiquetas) != len(df):
        raise ValueError(
            f"labels tiene {len(etiquetas)} elementos y el DataFrame {len(df)} "
            f"filas: deben venir en el mismo orden y cantidad."
        )
    es_ruido = np.asarray(etiquetas) == RUIDO
    return df.loc[es_ruido].copy(), df.loc[~es_ruido].copy()


def perfil_outliers(df: pd.DataFrame, etiquetas: np.ndarray,
                    columnas: list[str] | None = None) -> pd.DataFrame:
    """Contrasta el promedio de los atípicos contra el del resto.

    Comparar medias crudas entre variables con unidades distintas no dice en
    cuál se desvían *más*: 200 colones de diferencia en el ticket y 0.2 en una
    participación no son comparables. Por eso la columna que ordena la tabla es
    ``brecha_sd``, la diferencia expresada en desviaciones estándar del resto
    de la cartera, que sí es adimensional.

    La referencia excluye a los propios atípicos: incluirlos contamina el
    promedio contra el que se los compara.
    """
    columnas = columnas or [c for c in COLS_PERFIL if c in df.columns]
    outliers, resto = separar_outliers(df, etiquetas)

    if outliers.empty:
        return pd.DataFrame(columns=["Atípicos", "Resto", "Brecha (desv. est.)",
                                     "Brecha %", "Percentil medio"])

    mu_out = outliers[columnas].mean()
    mu_resto = resto[columnas].mean()
    sd_resto = resto[columnas].std().replace(0, np.nan)

    # Percentil promedio de los atipicos dentro de la distribucion completa.
    # Distingue dos cosas que la brecha de medias confunde: un atipico en el
    # percentil 2 o 98 es un caso extremo y merece revision individual; uno
    # cerca de 50 en todo es un caso fronterizo entre grupos, que es otra
    # historia y normalmente menos interesante.
    percentiles = df[columnas].rank(pct=True) * 100
    pct_out = percentiles.loc[outliers.index].mean()

    tabla = pd.DataFrame({
        "Atípicos": mu_out,
        "Resto": mu_resto,
        "Brecha (desv. est.)": (mu_out - mu_resto) / sd_resto,
        "Brecha %": (mu_out - mu_resto) / mu_resto.abs().replace(0, np.nan) * 100,
        "Percentil medio": pct_out,
    })
    tabla = tabla.reindex(
        tabla["Brecha (desv. est.)"].abs().sort_values(ascending=False).index)
    tabla.index = [ETIQUETAS_PERFIL.get(i, i) for i in tabla.index]
    tabla.index.name = "Variable"
    return tabla


# Nombres de negocio para los grupos hallados. Son una lectura del perfil, no
# una salida del algoritmo: DBSCAN devuelve 0,1,2,3 y el sentido lo pone quien
# analiza. Los ids son estables porque UMAP corre con semilla fija y DBSCAN es
# determinista; si se cambia eps, min_samples o la semilla hay que revisarlos
# contra la tabla de perfiles antes de reutilizarlos.
NOMBRES_GRUPOS = {
    0: "Rutina de mantenimiento",
    1: "Hogar con dependientes",
    2: "Alto valor experiencial",
    3: "Digital joven",
}


def perfilar_grupos(panel: pd.DataFrame, etiquetas: np.ndarray,
                    nombres: dict[int, str] | None = None) -> pd.DataFrame:
    """Tabla de negocio de los grupos: tamaño, valor y composición.

    Pensada para leerse desde finanzas y no desde el algoritmo: cuántos
    clientes, qué porción de la facturación explican, cuánto gastan y en qué.
    El ruido va como fila aparte, nunca sumado a un grupo.
    """
    d = panel.copy()
    d["_g"] = etiquetas
    total_gasto = d["gasto_total"].sum()

    filas = []
    for g, sub in d.groupby("_g", observed=True):
        etiqueta = ("Atípicos" if g == RUIDO
                    else (nombres or {}).get(g, f"Grupo {g}"))
        top = (sub[COLS_SHARE].mean()
               .rename(lambda c: cfg.ETIQUETAS_ES[c.replace("share_", "")])
               .nlargest(3))

        filas.append({
            "grupo": g,
            "Segmento": etiqueta,
            "Clientes": len(sub),
            "% clientes": round(len(sub) / len(d) * 100, 1),
            "% facturación": round(sub["gasto_total"].sum() / total_gasto * 100, 1),
            "Gasto mensual": round(sub["gasto_por_mes"].mean()),
            "Ticket medio": round(sub["ticket_medio"].mean()),
            "Edad media": round(sub["edad"].mean()),
            "% hedónico": round(sub["share_hedonico"].mean() * 100),
            "% online": round(sub["share_online"].mean() * 100),
            "% nocturno": round(sub["share_nocturna"].mean() * 100),
            "Categorías principales": " · ".join(top.index),
        })

    # Se omite el ritmo de transacciones: es ~11 al mes en los cuatro grupos,
    # asi que ocupa ancho sin distinguir nada.
    out = pd.DataFrame(filas).sort_values(
        ["grupo"], key=lambda s: s.replace(RUIDO, 10_000)   # el ruido al final
    )
    return out.set_index("Segmento").drop(columns="grupo")


def barrer_min_samples(X_2d: np.ndarray, valores, y_true=None) -> pd.DataFrame:
    """Barre min_samples recalculando su curva de k-distancias cada vez.

    k = min_samples no es una convención cosmética: la curva describe la
    densidad que DBSCAN va a usar. Cambiar min_samples y reusar el eps de un k
    distinto responde a otra pregunta, así que acá cada fila recalibra.
    """
    filas = []
    for ms in valores:
        d = curva_k_distancias(X_2d, ms)
        _, eps = codo(d)
        etiquetas = agrupar(X_2d, eps, ms)
        fila = {"min_samples": ms, "eps_codo": round(float(eps), 4)}
        fila.update(resumen_grupos(etiquetas).to_dict())

        if y_true is not None:
            from sklearn.metrics import adjusted_rand_score
            asignados = etiquetas != RUIDO
            fila["ari_agrupados"] = round(
                adjusted_rand_score(np.asarray(y_true)[asignados],
                                    etiquetas[asignados]), 3)
        filas.append(fila)

    return pd.DataFrame(filas).set_index("min_samples")


if __name__ == "__main__":
    import argparse

    from . import data_io, features, prep

    ap = argparse.ArgumentParser()
    ap.add_argument("--min-samples", type=int, default=MIN_SAMPLES_DEFAULT)
    args = ap.parse_args()

    ruta, origen = data_io.resolver_fuente()
    df = data_io.cargar(ruta)
    df, _ = prep.limpiar(df)
    df = prep.enriquecer(df)
    panel = features.construir_panel(df)

    X_scaled, columnas = construir_matriz(panel)
    print(f"Fuente: {origen}  |  X_scaled: {X_scaled.shape[0]} filas x "
          f"{X_scaled.shape[1]} columnas")

    X_2d, _ = reducir_umap(X_scaled)
    print(f"\n{verificar(X_2d, X_scaled).to_string()}")

    from . import viz
    viz.aplicar_estilo()
    print(f"\nfigura -> {viz.mapa_umap(X_2d)}")

    # --- Calibración de eps ------------------------------------------------
    min_samples = args.min_samples
    distancias = curva_k_distancias(X_2d, min_samples)
    i_codo, eps = codo(distancias)
    candidatos = eps_candidatos(distancias, eps)

    print(f"\nmin_samples = {min_samples}   "
          f"(la regla 2 x dimensiones daria {min_samples_sugerido(X_2d.shape[1])})")
    print(f"codo en la entidad {i_codo} de {len(distancias)} "
          f"({i_codo / len(distancias):.1%} de la curva)")
    print(f"eps sugerido = {eps:.4f}")
    print(f"\nCandidatos a probar:\n{candidatos.to_string()}")
    print(f"\nfigura -> {viz.grafico_k_distancias(distancias, i_codo, eps, min_samples, candidatos)}")

    # --- DBSCAN ------------------------------------------------------------
    y_true = None
    if "_segmento_real" in df.columns:
        real = df.groupby("cc_num", observed=True)["_segmento_real"].first()
        y_true = panel.index.map(real).to_numpy()

    print(f"\n{'=' * 66}\nDBSCAN\n{'=' * 66}")
    print(comparar_eps(X_2d, candidatos["eps"], min_samples, y_true).to_string(index=False))

    etiquetas = agrupar(X_2d, eps, min_samples)
    print(f"\nCon eps del codo ({eps:.4f}):")
    print(resumen_grupos(etiquetas).to_string())
    print(f"\n{tamanos_grupos(etiquetas).to_string()}")
    print(f"\nfigura -> {viz.mapa_clusters(X_2d, etiquetas, eps, min_samples)}")
