"""Figuras del proyecto.

Paleta compartida con el resto del portafolio y validada con el validador de la
guia de visualizacion. El verde de acento no alcanza 3:1 contra la superficie
clara, de modo que las figuras que lo usan llevan siempre etiqueta visible.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from . import config as cfg

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TINTA_MUTED = "#898781"
GRILLA = "#e1e0d9"
EJE = "#c3c2b7"

AZUL = cfg.PALETA["favorita"]
NARANJA = cfg.PALETA["andina"]
VERDE = cfg.PALETA["acento"]
ROJO = cfg.PALETA["brecha"]

CMAP_BRECHA = LinearSegmentedColormap.from_list("brecha", [ROJO, "#f0efec", AZUL])


def aplicar_estilo() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "savefig.bbox": "tight",
        "figure.dpi": 140,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "text.color": TINTA,
        "axes.labelcolor": TINTA_2,
        "axes.edgecolor": EJE,
        "axes.linewidth": 0.8,
        "axes.titlesize": 12.5,
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
    })


def _titular(ax, titulo: str, subtitulo: str | None = None) -> None:
    """Titular y bajada, con espacio suficiente para que no se pisen."""
    if not subtitulo:
        ax.set_title(titulo)
        return
    ax.set_title(titulo, pad=34)
    ax.text(0, 1.022, subtitulo, transform=ax.transAxes, fontsize=9.5,
            color=TINTA_2, va="bottom")


def _nombre(familia: str) -> str:
    """Nombres de familia en espanol, para que la figura se lea sola."""
    tr = {
        "GROCERY I": "Abarrotes", "GROCERY II": "Abarrotes II",
        "BEVERAGES": "Bebidas", "PRODUCE": "Frutas y verduras",
        "CLEANING": "Limpieza", "DAIRY": "Lácteos", "BREAD/BAKERY": "Panadería",
        "POULTRY": "Aves", "MEATS": "Carnes", "PERSONAL CARE": "Cuidado personal",
        "DELI": "Charcutería", "HOME CARE": "Cuidado del hogar", "EGGS": "Huevos",
        "FROZEN FOODS": "Congelados", "PREPARED FOODS": "Comida preparada",
        "LIQUOR,WINE,BEER": "Licores y cerveza", "SEAFOOD": "Mariscos",
        "HOME AND KITCHEN I": "Hogar I", "HOME AND KITCHEN II": "Hogar II",
        "CELEBRATION": "Fiesta", "LINGERIE": "Lencería", "LADIESWEAR": "Ropa de mujer",
        "PLAYERS AND ELECTRONICS": "Electrónica", "AUTOMOTIVE": "Automotriz",
        "LAWN AND GARDEN": "Jardín", "PET SUPPLIES": "Mascotas", "BEAUTY": "Belleza",
        "SCHOOL AND OFFICE SUPPLIES": "Escolar y oficina", "MAGAZINES": "Revistas",
        "HARDWARE": "Ferretería", "HOME APPLIANCES": "Electrodomésticos",
        "BABY CARE": "Bebé", "BOOKS": "Libros",
    }
    return tr.get(familia, familia.title())


def _celda(ciudad: str, familia: str) -> str:
    return f"{ciudad} · {_nombre(familia)}"


# --------------------------------------------------------------- portafolio
def fig_concentracion(panel: pd.DataFrame, top: int = 12):
    """Peso de cada familia en volumen frente a su peso en contribucion."""
    p = panel[panel.abierta]
    g = p.groupby("family", observed=True)[["unidades", "ingreso", "contribucion"]].sum()
    g["pct_volumen"] = 100 * g.unidades / g.unidades.sum()
    g["pct_contribucion"] = 100 * g.contribucion / g.contribucion.sum()
    g = g.sort_values("pct_volumen", ascending=False).head(top).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    y = np.arange(len(g))
    h = 0.36
    ax.barh(y + h / 2 + 0.02, g.pct_volumen, height=h, color=AZUL,
            label="Volumen (unidades)")
    ax.barh(y - h / 2 - 0.02, g.pct_contribucion, height=h, color=NARANJA,
            label="Contribución bruta (USD)")
    ax.set_yticks(y, [_nombre(f) for f in g.index])
    ax.set_xlabel("Porcentaje del total")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    for yy, v in zip(y + h / 2 + 0.02, g.pct_volumen):
        ax.text(v + 0.4, yy, f"{v:.0f}%", va="center", fontsize=8.5, color=TINTA_2)
    for yy, v in zip(y - h / 2 - 0.02, g.pct_contribucion):
        ax.text(v + 0.4, yy, f"{v:.0f}%", va="center", fontsize=8.5, color=TINTA_2)
    ax.set_xlim(0, max(g.pct_volumen.max(), g.pct_contribucion.max()) * 1.12)
    ax.legend(loc="lower right")
    _titular(ax, "El volumen y el dinero no viven en la misma familia",
             "Participación en las unidades vendidas y en la contribución bruta")
    fig.tight_layout()
    return fig


def fig_share_familia(bench: pd.DataFrame, meses: int = 12):
    """Participacion de Favorita por familia frente al competidor."""
    corte = bench.mes.max() - pd.DateOffset(months=meses - 1)
    v = bench[bench.mes >= corte]
    g = (v.groupby("family", observed=True)
          .agg(fav=("unidades", "sum"), mercado=("mercado_unidades", "sum"),
               ingreso=("mercado_ingreso", "sum")))
    g = g[g.ingreso >= 0.004 * g.ingreso.sum()]
    g["share"] = 100 * g.fav / g.mercado
    g = g.sort_values("share")
    global_share = 100 * v.unidades.sum() / v.mercado_unidades.sum()

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    y = np.arange(len(g))
    ax.hlines(y, 0, g.share, color=GRILLA, linewidth=1.4)
    colores = np.where(g.share < global_share, NARANJA, AZUL)
    ax.scatter(g.share, y, s=46, color=colores, zorder=3)
    ax.axvline(global_share, color=TINTA_MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(global_share + 1.0, 0.1, f"promedio {global_share:.0f}%",
            fontsize=9, color=TINTA_2)
    ax.set_yticks(y, [_nombre(f) for f in g.index])
    ax.set_xlabel("Participación de Favorita en el mercado de la familia (%)")
    ax.set_xlim(0, 100)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    for yy, v_ in zip(y, g.share):
        ax.text(v_ + 1.6, yy, f"{v_:.0f}%", va="center", fontsize=8.5, color=TINTA_2)
    _titular(ax, "Dónde muerde el competidor",
             "Últimos 12 meses · naranja: por debajo del promedio de la cadena")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ brechas
def fig_mapa_brechas(d: pd.DataFrame, min_celdas: int = 3):
    """Mapa de calor del residuo: que celdas rinden peor de lo esperado."""
    piv = d.pivot_table(index="city", columns="family", values="residuo")
    piv = piv.loc[piv.notna().sum(1) >= min_celdas, piv.notna().sum(0) >= min_celdas]
    orden_c = (d.groupby("city").mercado_ingreso.sum()
               .reindex(piv.index).sort_values(ascending=False))
    orden_f = (d.groupby("family").mercado_ingreso.sum()
               .reindex(piv.columns).sort_values(ascending=False))
    piv = piv.loc[orden_c.index, orden_f.index]

    lim = float(np.nanmax(np.abs(piv.to_numpy())))
    datos = np.ma.masked_invalid(piv.to_numpy())
    cmap = CMAP_BRECHA.copy()
    cmap.set_bad(GRILLA)  # sin dato: gris, para no confundirlo con un residuo nulo
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    im = ax.imshow(datos, cmap=cmap,
                   norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim), aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [_nombre(f) for f in piv.columns],
                  rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(piv.index)), piv.index, fontsize=9)
    ax.set_xticks(np.arange(-.5, len(piv.columns)), minor=True)
    ax.set_yticks(np.arange(-.5, len(piv.index)), minor=True)
    ax.grid(which="minor", color=SUPERFICIE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            val = piv.iat[i, j]
            if not np.isnan(val) and abs(val) >= 0.20:
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7.5,
                        color=TINTA if abs(val) < 0.6 * lim else SUPERFICIE)
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.035)
    cb.set_label("Residuo en log-odds", fontsize=9, color=TINTA_2)
    cb.outline.set_visible(False)
    _titular(ax, "Lo que la plaza y la familia no explican",
             "Rojo: rinde bajo lo esperado · gris: sin presencia · se anotan residuos sobre 0,20")
    fig.tight_layout()
    return fig


def fig_embudo(emb: pd.DataFrame):
    """Cuanto sobrevive a cada criterio, en celdas y en dinero."""
    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    y = np.arange(len(emb))[::-1]
    ax.barh(y, emb.contribucion_usd / 1e3, height=0.52, color=AZUL)
    ax.set_yticks(y, [c.replace("+ ", "   + ") for c in emb.criterio], fontsize=9.5)
    ax.set_xlabel("Contribución bruta anual en juego (miles de USD)")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    for yy, usd, n in zip(y, emb.contribucion_usd, emb.celdas):
        ax.text(usd / 1e3 + 28, yy, f"USD {usd/1e3:,.0f} k · {n} celdas",
                va="center", fontsize=9, color=TINTA_2)
    ax.set_xlim(0, emb.contribucion_usd.max() / 1e3 * 1.45)
    _titular(ax, "De la brecha contable a la oportunidad accionable",
             "Cada criterio descarta plata que no resiste el escrutinio")
    fig.tight_layout()
    return fig


def fig_persistencia(res_mens: pd.DataFrame, prioritarias: list[tuple[str, str]],
                     descartada: tuple[str, str] | None = None):
    """Residuo mes a mes: la brecha estructural no se despinta."""
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.axhline(0, color=EJE, linewidth=1)

    if descartada is not None:
        g = res_mens[(res_mens.city == descartada[0]) &
                     (res_mens.family == descartada[1])].sort_values("mes")
        if len(g):
            ax.plot(g.mes, g.residuo, color=TINTA_MUTED, linewidth=1.6,
                    linestyle=(0, (4, 3)),
                    label=f"{_celda(*descartada)} (descartada)")

    colores = [ROJO, NARANJA, AZUL, VERDE]
    for (ciudad, familia), color in zip(prioritarias, colores):
        g = res_mens[(res_mens.city == ciudad) &
                     (res_mens.family == familia)].sort_values("mes")
        if not len(g):
            continue
        ax.plot(g.mes, g.residuo, color=color, label=_celda(ciudad, familia))
        ax.scatter([g.mes.iloc[-1]], [g.residuo.iloc[-1]], s=30, color=color, zorder=3)

    ax.set_ylabel("Residuo mensual (log-odds)")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", ncols=2)
    _titular(ax, "Una brecha estructural no se despinta",
             "Bajo cero, la celda rinde menos de lo que su plaza y su familia predicen")
    fig.tight_layout()
    return fig


def fig_elasticidad(el: pd.DataFrame, familias=None, top: int = 14):
    """Elasticidad del volumen a los items en promocion, con intervalo.

    Se limita a las familias con peso economico: la elasticidad de Revistas es
    un dato, pero no una palanca de negocio.
    """
    e = el if familias is None else el[el.family.isin(set(familias))]
    e = e.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    y = np.arange(len(e))
    colores = np.where(e.t > 2, AZUL, TINTA_MUTED)
    ax.hlines(y, 0, e.elasticidad, color=GRILLA, linewidth=1.4)
    ax.errorbar(e.elasticidad, y, xerr=1.96 * e.ee, fmt="none",
                ecolor=EJE, elinewidth=1.4, capsize=2.5)
    ax.scatter(e.elasticidad, y, s=44, color=colores, zorder=3)
    ax.axvline(0, color=EJE, linewidth=1)
    ax.set_yticks(y, [_nombre(f) for f in e.family])
    ax.set_xlabel("Elasticidad: % de unidades por 1 % más de ítems en promoción")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    for yy, v in zip(y, e.elasticidad):
        ax.text(v + 0.04, yy, f"{v:.2f}", va="center", fontsize=8.5, color=TINTA_2)
    ax.set_xlim(right=e.elasticidad.max() * 1.18)
    _titular(ax, "No todas las familias responden a la promoción",
             "Efectos fijos de tienda y de mes · gris: no distinguible de cero")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- robustez
def fig_valor(prior: pd.DataFrame):
    """Contribucion recuperable por celda prioritaria."""
    p = prior.sort_values("oport_contribucion").copy()
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    y = np.arange(len(p))
    ax.barh(y, p.oport_contribucion / 1e3, height=0.5, color=AZUL)
    ax.set_yticks(y, [_celda(c, f) for c, f in zip(p.city, p.family)], fontsize=9.5)
    ax.set_xlabel("Contribución bruta anual recuperable (miles de USD)")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    for yy, usd, pp in zip(y, p.oport_contribucion, p.brecha_pp):
        ax.text(usd / 1e3 + 4, yy, f"USD {usd/1e3:,.0f} k · brecha {pp:.0f} pp",
                va="center", fontsize=9, color=TINTA_2)
    ax.set_xlim(0, p.oport_contribucion.max() / 1e3 * 1.48)
    _titular(ax, "Qué vale cerrar cada brecha",
             "Oportunidad anualizada, a los precios y márgenes supuestos")
    fig.tight_layout()
    return fig


def fig_umbral(curva: pd.DataFrame, corte_elegido: float = 0.20):
    """Precision y cobertura segun donde se ponga el corte."""
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(curva.corte, 100 * curva.precision, color=AZUL, marker="o",
            markersize=5, label="Precisión: de lo marcado, cuánto era real")
    ax.plot(curva.corte, 100 * curva.cobertura, color=NARANJA, marker="s",
            markersize=5, label="Cobertura: de lo real, cuánto se encontró")
    ax.axvline(corte_elegido, color=TINTA_MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(corte_elegido + 0.008, 6, "corte elegido", fontsize=9, color=TINTA_2)
    ax.set_xlabel("Corte mínimo del residuo (log-odds)")
    ax.set_ylabel("Porcentaje")
    ax.set_ylim(0, 108)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center")
    _titular(ax, "Dónde poner el corte",
             "Un filtro laxo llena la lista de ruido; uno estricto deja plata sobre la mesa")
    fig.tight_layout()
    return fig


def fig_semillas(frec: pd.DataFrame, n_semillas: int):
    """Cuantas veces sobrevive cada celda al re-sortear el mercado."""
    f = frec.sort_values("veces").tail(12)
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    y = np.arange(len(f))
    colores = np.where(f.sembrada, AZUL, TINTA_MUTED)
    ax.barh(y, f.veces, height=0.5, color=colores)
    ax.set_yticks(y, [_celda(c, fa) for c, fa in zip(f.city, f.family)], fontsize=9.5)
    ax.set_xlabel(f"Veces que la celda entra en la lista (de {n_semillas} simulaciones)")
    ax.set_xlim(0, n_semillas * 1.3)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    for yy, v, s in zip(y, f.veces, f.sembrada):
        ax.text(v + 0.18, yy, f"{v}/{n_semillas} · {'ventaja real' if s else 'ruido'}",
                va="center", fontsize=9, color=TINTA_2)
    _titular(ax, "Qué sobrevive cuando se vuelve a sortear el mercado",
             "Azul: celdas donde el generador sembró una ventaja competitiva real")
    fig.tight_layout()
    return fig
