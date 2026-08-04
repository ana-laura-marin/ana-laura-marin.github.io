"""Figuras del proyecto.

Paleta validada con el validador de la guia de visualizacion (checks de banda
de luminosidad, piso de croma, separacion CVD y piso de vision normal en todos
los pares). El aqua queda por debajo de 3:1 contra la superficie clara, asi que
todas las figuras que lo usan llevan etiquetas visibles o su tabla equivalente
en reports/tables.
"""

from __future__ import annotations

import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from . import config as cfg
from .features import COLS_SHARE

# --- Paleta ----------------------------------------------------------------
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TINTA_MUTED = "#898781"
GRILLA = "#e1e0d9"
EJE = "#c3c2b7"

SERIE = {"1_azul": "#2a78d6", "2_naranja": "#eb6834", "3_aqua": "#1baf7a"}
COLOR_TIPO = {
    "hedonico": SERIE["2_naranja"],
    "utilitario": SERIE["1_azul"],
    "mixto": SERIE["3_aqua"],
}
# Divergente azul <-> rojo con punto medio gris neutro.
CMAP_DIV = LinearSegmentedColormap.from_list(
    "div_lift", ["#e34948", "#f0efec", "#2a78d6"]
)


def aplicar_estilo() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "text.color": TINTA,
        "axes.labelcolor": TINTA_2,
        "axes.edgecolor": EJE,
        "axes.linewidth": 0.8,
        "axes.titlesize": 13,
        "axes.titleweight": "600",
        "axes.titlecolor": TINTA,
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": TINTA_MUTED,
        "ytick.color": TINTA_MUTED,
        "xtick.labelcolor": TINTA_2,
        "ytick.labelcolor": TINTA_2,
        "grid.color": GRILLA,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "figure.dpi": 130,
        "savefig.bbox": "tight",
    })


def _guardar(fig, nombre: str) -> str:
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    ruta = cfg.FIGURES / f"{nombre}.png"
    fig.savefig(ruta)
    plt.close(fig)
    return str(ruta)


def _leyenda_tipos(ax) -> None:
    manijas = [
        mpl.patches.Patch(facecolor=COLOR_TIPO[t], label=t.capitalize())
        for t in ["hedonico", "utilitario", "mixto"]
    ]
    ax.legend(handles=manijas, loc="lower right", ncols=3)


# --- 1. Composicion agregada del gasto -------------------------------------
def composicion_categorias(df_txn: pd.DataFrame) -> str:
    monto = df_txn.groupby("category", observed=True)["amt"].sum()
    share = (monto / monto.sum() * 100).sort_values()

    etiquetas = [cfg.ETIQUETAS_ES[c] for c in share.index]
    colores = [COLOR_TIPO[cfg.tipo_gasto(c)] for c in share.index]

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.barh(etiquetas, share.values, color=colores, height=0.62)
    # Etiqueta directa en cada barra: cumple la regla de relieve del aqua.
    for y, v in enumerate(share.values):
        ax.text(v + 0.25, y, f"{v:.1f}%", va="center", fontsize=9, color=TINTA_2)

    ax.set_title("Composicion del gasto por categoria")
    ax.set_xlabel("% del monto total transado")
    ax.set_xlim(0, share.max() * 1.15)
    ax.xaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    _leyenda_tipos(ax)
    return _guardar(fig, "01_composicion_categorias")


# --- 2. Diagnostico de k ---------------------------------------------------
def diagnostico_k(diag: pd.DataFrame, k_elegido: int) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))

    for ax, col, titulo in [
        (axes[0], "silueta", "Coeficiente de silueta"),
        (axes[1], "inercia", "Inercia intra-cluster"),
    ]:
        ax.plot(diag["k"], diag[col], color=SERIE["1_azul"], marker="o",
                markersize=5, markerfacecolor=SUPERFICIE, markeredgewidth=1.8)
        y = diag.loc[diag["k"] == k_elegido, col].iat[0]
        ax.scatter([k_elegido], [y], s=110, color=SERIE["1_azul"], zorder=3,
                   edgecolor=SUPERFICIE, linewidth=2)
        ax.annotate(f"k = {k_elegido}", (k_elegido, y), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, color=TINTA)
        ax.set_title(titulo)
        ax.set_xlabel("Numero de segmentos (k)")
        ax.yaxis.grid(True, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)

    fig.suptitle("Seleccion del numero de segmentos", x=0.005, ha="left",
                 fontsize=13, fontweight="600", color=TINTA)
    fig.tight_layout()
    return _guardar(fig, "02_diagnostico_k")


# --- 3. Mapa de segmentos (small multiples) --------------------------------
def mapa_segmentos(panel: pd.DataFrame, varianza: np.ndarray,
                   nombres: dict[int, str]) -> str:
    """Un panel por segmento sobre el fondo de la poblacion.

    Se usan multiplos pequenos en vez de un scatter con k colores: mas alla de
    tres series simultaneas ningun orden de la paleta pasa el piso de vision
    normal en todos los pares.
    """
    segs = sorted(panel["segmento"].unique())
    ncols = min(3, len(segs))
    nrows = int(np.ceil(len(segs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.3 * nrows),
                             squeeze=False, sharex=True, sharey=True)

    for i, seg in enumerate(segs):
        ax = axes[i // ncols][i % ncols]
        ax.scatter(panel["pc1"], panel["pc2"], s=7, color=GRILLA, linewidth=0)
        sub = panel[panel["segmento"] == seg]
        ax.scatter(sub["pc1"], sub["pc2"], s=9, color=SERIE["1_azul"], linewidth=0)
        ax.set_title(f"Segmento {seg}  ·  n={len(sub)}", fontsize=10)
        # El nombre se parte en la conjuncion para no desbordar el panel.
        etiqueta = "\n".join(textwrap.wrap(nombres.get(seg, ""), width=30))
        ax.text(0.03, 0.04, etiqueta, transform=ax.transAxes, fontsize=7.5,
                color=TINTA_2, va="bottom", linespacing=1.35)
        ax.tick_params(length=0, labelsize=8)

    for j in range(len(segs), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.supxlabel(f"CP1 ({varianza[0]*100:.0f}% de la varianza)", fontsize=9, color=TINTA_2)
    fig.supylabel(f"CP2 ({varianza[1]*100:.0f}%)", fontsize=9, color=TINTA_2)
    fig.suptitle("Segmentos en el espacio log-ratio del gasto", x=0.005, ha="left",
                 fontsize=13, fontweight="600", color=TINTA)
    fig.tight_layout()
    return _guardar(fig, "03_mapa_segmentos")


# --- 4. Sobre/sub representacion por segmento ------------------------------
def heatmap_lift(lift: pd.DataFrame) -> str:
    M = lift.to_numpy()

    # El indice es una razon: 200 y 50 son inversos entre si y deben quedar a
    # la misma distancia del punto neutro. En escala lineal no lo estan, asi
    # que el color se mapea sobre log2(lift/100) y la celda muestra el indice.
    L = np.log2(np.clip(M, 1e-3, None) / 100)
    tope = max(abs(L).max(), 0.1)
    norma = TwoSlopeNorm(vmin=-tope, vcenter=0.0, vmax=tope)

    fig, ax = plt.subplots(figsize=(1.05 * len(lift.columns) + 2, 0.62 * len(lift) + 2.4))
    ax.imshow(L, cmap=CMAP_DIV, norm=norma, aspect="auto")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color=TINTA if abs(L[i, j]) < tope * 0.55 else SUPERFICIE)

    ax.set_xticks(range(len(lift.columns)), lift.columns, rotation=45,
                  ha="right", fontsize=9)
    ax.set_yticks(range(len(lift)), [f"Segmento {s}" for s in lift.index], fontsize=9)
    ax.set_title("Indice de especializacion por segmento  (100 = promedio de la cartera)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return _guardar(fig, "04_lift_segmentos")


# --- 5. Perfil horario hedonico vs utilitario ------------------------------
def perfil_horario(df_txn: pd.DataFrame) -> str:
    tabla = (
        df_txn.groupby(["tipo_gasto", "hora"], observed=True)
        .size().unstack(fill_value=0)
    )
    tabla = tabla.div(tabla.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for tipo in ["hedonico", "utilitario"]:
        if tipo not in tabla.index:
            continue
        y = tabla.loc[tipo]
        ax.plot(y.index, y.values, color=COLOR_TIPO[tipo], label=tipo.capitalize())
        # Etiqueta directa al final de la serie, ademas de la leyenda.
        ax.text(y.index[-1] + 0.25, y.values[-1], tipo.capitalize(),
                color=COLOR_TIPO[tipo], fontsize=9, va="center", fontweight="600")

    ax.set_title("Cuando se gasta: distribucion horaria por tipo de gasto")
    ax.set_xlabel("Hora del dia")
    ax.set_ylabel("% de las transacciones del tipo")
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlim(-0.5, 26)
    ax.yaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(loc="upper left")
    return _guardar(fig, "05_perfil_horario")


# --- 6. Efectos parciales de la edad ---------------------------------------
def efectos_edad(largo: pd.DataFrame, regresor: str = "edad_c") -> str | None:
    d = largo[largo["regresor"] == regresor].copy()
    if d.empty:
        return None
    d = d.sort_values("ape")

    # Polaridad: el signo es la informacion, asi que par divergente azul/rojo.
    # Los efectos no significativos van en gris: pintarles un signo sugiere una
    # direccion que los datos no sostienen.
    colores = np.where(d["p_valor"] >= 0.10, TINTA_MUTED,
                       np.where(d["ape"] >= 0, "#2a78d6", "#e34948"))

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.axvline(0, color=EJE, linewidth=1)
    ax.errorbar(d["ape"], range(len(d)), xerr=1.96 * d["ape_se"], fmt="none",
                ecolor=TINTA_MUTED, elinewidth=1.2, capsize=3)
    ax.scatter(d["ape"], range(len(d)), s=52, color=colores, zorder=3,
               edgecolor=SUPERFICIE, linewidth=1.5)

    ax.set_yticks(range(len(d)),
                  [f"{c}{s}" for c, s in zip(d["categoria"], d["signif"])])
    ax.set_title("Efecto de un ano adicional de edad sobre la participacion de cada categoria")
    ax.set_xlabel("Efecto parcial promedio (puntos de participacion por ano)\n"
                  "Barras: intervalo de confianza al 95% con errores robustos")
    ax.xaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    return _guardar(fig, "06_efecto_edad")


# --- Embedding UMAP --------------------------------------------------------
def mapa_umap(X_2d, nombre: str = "07_umap_2d") -> str:
    """Scatter crudo del embedding, antes de agrupar.

    Sin color: el punto es juzgar si hay grumos separados por regiones vacías
    —que es lo único que DBSCAN puede encontrar— sin que un coloreado previo
    sugiera estructura que no está.
    """
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    # Puntos pequeños y semitransparentes: en un scatter denso, marcas grandes
    # y opacas tapan justo las diferencias de densidad que se quieren ver.
    ax.scatter(X_2d[:, 0], X_2d[:, 1], s=11, color=SERIE["1_azul"],
               alpha=0.55, linewidth=0)

    ax.set_title(f"Embedding UMAP en 2D  ·  {len(X_2d):,} entidades")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    # Los ejes de UMAP no tienen unidades interpretables: solo importa la
    # posición relativa, no la escala.
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return _guardar(fig, nombre)


def diagrama_dbscan(nombre: str = "10_como_agrupa_dbscan") -> str:
    """Diagrama didáctico de cómo decide DBSCAN.

    Los puntos son inventados para la ilustración, pero la clasificación en
    núcleo / frontera / ruido la hace el DBSCAN de verdad: así el dibujo no
    puede contradecir al algoritmo que explica.
    """
    from sklearn.cluster import DBSCAN

    eps, min_samples = 0.62, 4
    rng = np.random.default_rng(7)

    def rejilla(cx, cy, nx, ny, paso=0.3):
        xs = (np.arange(nx) - (nx - 1) / 2) * paso + cx
        ys = (np.arange(ny) - (ny - 1) / 2) * paso + cy
        return np.array([[x, y] for x in xs for y in ys])

    # Geometria deliberada, no aleatoria: hace falta que existan los tres roles
    # para poder mostrarlos. Las fronteras se ubican a 0.55 de un nucleo (dentro
    # del radio) pero suficientemente solas como para no alcanzar min_samples.
    denso = rejilla(0, 0, 5, 5)
    segundo = rejilla(3.2, 1.3, 3, 3)
    fronteras = np.array([[1.17, 0.0], [2.32, 1.3]])
    aislados = np.array([[1.9, -1.45], [3.4, -0.95]])
    P = np.vstack([denso, segundo, fronteras, aislados])
    # El jitter tiene que ser menor que el margen geometrico mas ajustado
    # (~0.024) o los puntos frontera ganan un vecino y pasan a ser nucleo.
    P = P + rng.normal(0, 0.008, P.shape)

    modelo = DBSCAN(eps=eps, min_samples=min_samples).fit(P)
    etiquetas = modelo.labels_
    es_nucleo = np.zeros(len(P), bool)
    es_nucleo[modelo.core_sample_indices_] = True
    es_ruido = etiquetas == -1
    es_frontera = ~es_nucleo & ~es_ruido

    if es_frontera.sum() != len(fronteras) or es_ruido.sum() != len(aislados):
        raise RuntimeError(
            f"La geometria no produjo los roles esperados: "
            f"nucleos={es_nucleo.sum()}, fronteras={es_frontera.sum()} "
            f"(se esperaban {len(fronteras)}), ruido={es_ruido.sum()} "
            f"(se esperaban {len(aislados)})"
        )

    fig, ax = plt.subplots(figsize=(8.6, 5.0))

    # Vecindario de radio eps sobre un nucleo real y sobre un punto de ruido:
    # la comparacion visual es el argumento entero del algoritmo.
    centro_nucleo = P[np.flatnonzero(es_nucleo)[len(denso) // 2]]
    centro_ruido = P[np.flatnonzero(es_ruido)[0]]
    for centro, color in [(centro_nucleo, SERIE["1_azul"]), (centro_ruido, TINTA_MUTED)]:
        ax.add_patch(plt.Circle(centro, eps, facecolor=color, alpha=0.10,
                                edgecolor=color, linestyle="--", linewidth=1.3))

    for g in sorted(set(etiquetas.tolist()) - {-1}):
        m = etiquetas == g
        ax.annotate(f"Grupo {g}", (P[m, 0].mean(), P[m, 1].max()),
                    textcoords="offset points", xytext=(0, 26), ha="center",
                    fontsize=10, fontweight="600", color=TINTA_2)

    ax.scatter(P[es_nucleo, 0], P[es_nucleo, 1], s=62, color=SERIE["1_azul"],
               linewidth=0, zorder=3, label=f"Núcleo — ≥ {min_samples} vecinos dentro del radio")
    ax.scatter(P[es_frontera, 0], P[es_frontera, 1], s=62, color=SERIE["3_aqua"],
               linewidth=0, zorder=3, label="Frontera — pocos vecinos, pero pegado a un núcleo")
    ax.scatter(P[es_ruido, 0], P[es_ruido, 1], s=62, facecolor="none",
               edgecolor=TINTA_MUTED, linewidth=1.6, zorder=3,
               label="Ruido — no alcanza densidad ni toca ningún grupo")

    ax.annotate("radio = eps", centro_nucleo + [0, eps], textcoords="offset points",
                xytext=(0, 7), ha="center", fontsize=9, color=TINTA_2)

    ax.set_title("Cómo decide DBSCAN: densidad, no distancia a un centro")
    ax.set_xlabel("Un grupo es una cadena de núcleos que se alcanzan entre sí. "
                  "Nadie fija cuántos grupos hay.", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # Aire arriba para que ni la leyenda ni los rotulos de grupo choquen con
    # el titulo, que es lo primero que se rompe al cambiar la geometria.
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + (y1 - y0) * 0.30)
    ax.legend(loc="lower left", fontsize=9)
    return _guardar(fig, nombre)


def mapa_clusters(X_2d, etiquetas, eps: float, min_samples: int,
                  nombre: str = "09_dbscan") -> str:
    """Embedding coloreado por grupo de DBSCAN, con el ruido aparte.

    El ruido no recibe color de serie: va en gris, hueco y detrás de todo. Es
    la lectura correcta —no es un grupo más, es la ausencia de grupo— y además
    deja que los grupos reales dominen visualmente.
    """
    etiquetas = np.asarray(etiquetas)
    ids = sorted(i for i in set(etiquetas.tolist()) if i != -1)

    # La paleta valida hasta 4 series en todos los pares; a partir de ahí los
    # grupos extra se pliegan a un color neutro en vez de inventar tonos.
    slots = [SERIE["1_azul"], SERIE["2_naranja"], SERIE["3_aqua"], "#4a3aa7"]
    color_de = {g: slots[k] for k, g in enumerate(ids[:len(slots)])}

    fig, ax = plt.subplots(figsize=(6.8, 6.2))

    ruido = etiquetas == -1
    if ruido.any():
        ax.scatter(X_2d[ruido, 0], X_2d[ruido, 1], s=26, facecolor="none",
                   edgecolor=TINTA_MUTED, linewidth=0.9, zorder=1,
                   label=f"ruido ({int(ruido.sum())})")

    for g in ids:
        m = etiquetas == g
        color = color_de.get(g, EJE)
        ax.scatter(X_2d[m, 0], X_2d[m, 1], s=13, color=color, linewidth=0,
                   zorder=2, label=f"grupo {g} ({int(m.sum())})")
        # Etiqueta directa sobre el centroide: cumple la regla de relieve del
        # aqua y evita depender solo del color para identificar el grupo.
        # Se corre hacia arriba del centroide: encima del grumo taparia justo
        # la densidad que el grafico tiene que mostrar.
        ax.annotate(str(g), (X_2d[m, 0].mean(), X_2d[m, 1].max()),
                    textcoords="offset points", xytext=(0, 13),
                    fontsize=10, fontweight="700", color=TINTA, zorder=3,
                    ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.22", facecolor=SUPERFICIE,
                              edgecolor=color, linewidth=1.5))

    n_grupos = len(ids)
    ax.set_title(f"DBSCAN sobre el embedding  ·  {n_grupos} grupos  ·  "
                 f"eps={eps:.3f}, min_samples={min_samples}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.legend(loc="best", ncols=2, fontsize=8)
    return _guardar(fig, nombre)


def grafico_k_distancias(distancias, i_codo: int, eps_codo: float,
                         min_samples: int, candidatos=None,
                         nombre: str = "08_k_distancias") -> str:
    """Curva de k-distancias con el codo marcado.

    La parte plana son los puntos en zonas densas; la subida final, los
    aislados. El codo separa ambos regímenes y sugiere el eps.
    """
    n = len(distancias)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    ax.plot(np.arange(n), distancias, color=SERIE["1_azul"], linewidth=2)

    # Los candidatos van primero para que queden por detrás del codo.
    if candidatos is not None:
        for etiqueta, fila in candidatos.iterrows():
            if etiqueta == "codo":
                continue
            ax.axhline(fila["eps"], color=TINTA_MUTED, linewidth=1, linestyle=":")
            ax.annotate(f"{etiqueta}  eps {fila['eps']:.3f}  ·  "
                        f"{fila['pct_no_nucleo']:.0f}% fuera",
                        (0, fila["eps"]), textcoords="offset points",
                        xytext=(4, 4), fontsize=8, color=TINTA_MUTED)

    ax.axhline(eps_codo, color="#e34948", linewidth=1.4, linestyle="--")
    ax.scatter([i_codo], [eps_codo], s=95, color="#e34948", zorder=3,
               edgecolor=SUPERFICIE, linewidth=2)
    ax.annotate(f"codo · eps ≈ {eps_codo:.3f}", (i_codo, eps_codo),
                textcoords="offset points", xytext=(-12, 16), fontsize=10,
                color="#e34948", fontweight="600", ha="right")

    ax.set_title(f"Curva de k-distancias  ·  k = min_samples = {min_samples}")
    ax.set_xlabel(f"Entidades ordenadas por distancia a su {min_samples}º vecino")
    ax.set_ylabel(f"Distancia al {min_samples}º vecino")
    ax.set_xlim(0, n - 1)
    ax.set_ylim(0, distancias.max() * 1.08)
    ax.yaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    return _guardar(fig, nombre)


# --- Exploración: figuras del paso cero ------------------------------------
def distribucion_montos(df_txn: pd.DataFrame) -> str:
    """Histograma del monto en escala log.

    En escala lineal la cola larga aplasta todo contra el eje y no se ve nada.
    El log revela si la distribución es aproximadamente lognormal, que es lo
    que justifica modelar log(monto) más adelante.
    """
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.hist(np.log10(df_txn["amt"]), bins=70, color=SERIE["1_azul"], alpha=0.85)

    mediana = df_txn["amt"].median()
    ax.axvline(np.log10(mediana), color=TINTA_2, linewidth=1.4, linestyle="--")
    ax.annotate(f"mediana  {mediana:,.0f}", (np.log10(mediana), ax.get_ylim()[1] * 0.92),
                textcoords="offset points", xytext=(8, 0), fontsize=9, color=TINTA_2)

    ticks = [1, 10, 100, 1000, 10000]
    ax.set_xticks(np.log10(ticks), [f"{t:,}" for t in ticks])
    ax.set_title("Distribución del monto por transacción (escala log)")
    ax.set_xlabel("Monto")
    ax.set_ylabel("Transacciones")
    ax.yaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    return _guardar(fig, "00a_distribucion_montos")


def cobertura_temporal(df_txn: pd.DataFrame) -> str:
    """Volumen mensual: sirve para detectar huecos antes de modelar."""
    m = df_txn.set_index("trans_date_trans_time").resample("MS").size()

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.plot(m.index, m.values, color=SERIE["1_azul"], marker="o", markersize=4,
            markerfacecolor=SUPERFICIE, markeredgewidth=1.5)
    ax.set_ylim(0, m.max() * 1.15)
    ax.set_title("Cobertura temporal: transacciones por mes")
    ax.set_ylabel("Transacciones")
    ax.yaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    fig.autofmt_xdate(rotation=0, ha="center")
    return _guardar(fig, "00b_cobertura_temporal")


def generar_todas(df_txn: pd.DataFrame, panel: pd.DataFrame, perfil: dict,
                  diag: pd.DataFrame | None, k: int, varianza: np.ndarray,
                  nombres: dict, ape_largo: pd.DataFrame) -> list[str]:
    aplicar_estilo()
    rutas = [
        composicion_categorias(df_txn),
        mapa_segmentos(panel, varianza, nombres),
        heatmap_lift(perfil["lift_categorias"]),
        perfil_horario(df_txn),
    ]
    if diag is not None:
        rutas.insert(1, diagnostico_k(diag, k))
    r = efectos_edad(ape_largo)
    if r:
        rutas.append(r)
    return rutas
