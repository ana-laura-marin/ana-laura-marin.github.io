"""Construye los cuadernos del proyecto y los deja listos para ejecutar.

    python scripts/armar_notebooks.py

Los cuadernos son la fuente de las figuras que el reporte embebe; se generan
desde aqui para que el codigo viva en src/ y no se duplique a mano.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "notebooks"

PREAMBULO = '''import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd

from src import benchmark, brechas, data_io, robustez, viz
from src import config as cfg

pd.set_option("display.width", 200)
viz.aplicar_estilo()

panel = data_io.construir_panel()
plaza = data_io.panel_plaza(panel)
bench = benchmark.generar(plaza)
'''

ANALISIS = '''ventana = brechas.ventana(bench, 12)
material = brechas.filtrar_material(ventana)
d = brechas.descomponer(material)

res_mens = brechas.residuos_mensuales(bench, 24)
persist = brechas.persistencia(res_mens)
elast = brechas.elasticidad_promo(panel)
r = brechas.ranking(d, persist, elast)
prioritarias = r[r.prioritaria].sort_values("oport_contribucion", ascending=False)
'''

CUADERNOS = {
    "00_mercado": [
        ("md", "# El mercado y el portafolio\n\n"
               "Composición de las ventas de Favorita y posición frente al "
               "competidor de referencia."),
        ("code", PREAMBULO + '\nprint(f"{len(panel):,} filas · '
                 '{panel.mes.min():%Y-%m} a {panel.mes.max():%Y-%m}")'),
        ("md", "## Volumen contra dinero\n\n"
               "La familia que más unidades mueve no es la que más contribución "
               "deja. Medir el portafolio en unidades y decidir en dólares son "
               "dos cosas distintas."),
        ("code", '#| label: fig-concentracion\n'
                 '#| fig-cap: "Participación de cada familia en el volumen y en la '
                 'contribución bruta."\nfig = viz.fig_concentracion(panel)'),
        ("md", "## Dónde muerde el competidor\n\n"
               "Participación de Favorita en cada familia durante los últimos doce "
               "meses del panel, contra la cadena de referencia."),
        ("code", '#| label: fig-share-familia\n'
                 '#| fig-cap: "Participación de mercado por familia frente al '
                 'competidor de referencia."\nfig = viz.fig_share_familia(bench)'),
    ],
    "01_brechas": [
        ("md", "# De la brecha a la oportunidad\n\n"
               "Descomposición del share en efecto de plaza, efecto de familia y "
               "residuo, y valoración de lo que vale cerrarlo."),
        ("code", PREAMBULO + "\n" + ANALISIS +
                 '\nprint(f"R2 ponderado: {d.attrs[\'r2\']:.3f} · '
                 '{len(material)} celdas materiales de {len(ventana)}")'),
        ("md", "## Lo que la plaza y la familia no explican\n\n"
               "El residuo del modelo aditivo: cuánto rinde cada celda por encima "
               "o por debajo de lo que predicen su plaza y su familia."),
        ("code", '#| label: fig-mapa-brechas\n'
                 '#| fig-cap: "Residuo en log-odds por plaza y familia."\n'
                 'fig = viz.fig_mapa_brechas(d)'),
        ("md", "## Persistencia\n\n"
               "Una brecha que aparece un trimestre y se va es ruido. La que "
               "sobrevive dos años es una posición competitiva."),
        ("code", '#| label: fig-persistencia\n'
                 '#| fig-cap: "Residuo mensual de las celdas prioritarias y de una '
                 'celda descartada por inestable."\n'
                 'celdas = list(zip(prioritarias.city, prioritarias.family))[:4]\n'
                 'sobrantes = r[(r.material) & (~r.prioritaria) & (r.oport_contribucion > 0)]\n'
                 'peor = sobrantes.sort_values("oport_contribucion", ascending=False).iloc[0]\n'
                 'fig = viz.fig_persistencia(res_mens, celdas, (peor.city, peor.family))'),
        ("md", "## El embudo\n\n"
               "De toda la brecha contable, cuánta sobrevive a cada filtro."),
        ("code", '#| label: fig-embudo\n'
                 '#| fig-cap: "Contribución en juego según el criterio aplicado."\n'
                 'fig = viz.fig_embudo(brechas.embudo(r))'),
        ("md", "## ¿Sirve la promoción?\n\n"
               "Respuesta del volumen a los ítems en promoción, con efectos fijos "
               "de tienda y de mes."),
        ("code", '#| label: fig-elasticidad\n'
                 '#| fig-cap: "Elasticidad promocional por familia."\n'
                 'fig = viz.fig_elasticidad(elast, familias=d.family.unique())'),
        ("md", "## Qué vale cada oportunidad"),
        ("code", '#| label: fig-valor\n'
                 '#| fig-cap: "Contribución bruta anual recuperable por celda."\n'
                 'fig = viz.fig_valor(prioritarias)'),
        ("code", '#| label: tbl-prioritarias\n'
                 '#| tbl-cap: "Oportunidades prioritarias."\n'
                 'cols = ["city", "family", "share", "share_esperado", "brecha_pp",\n'
                 '        "oport_ingreso", "oport_contribucion", "t_ajustado", "elasticidad"]\n'
                 'prioritarias[cols].round(3)'),
    ],
    "02_robustez": [
        ("md", "# Qué tan firme es esto\n\n"
               "Los precios son un supuesto y el competidor es una simulación. "
               "Esta sección mide cuánto de la conclusión depende de cada cosa."),
        ("code", PREAMBULO + "\n" + ANALISIS +
                 '\nprioritarias = r[r.prioritaria].sort_values('
                 '"oport_contribucion", ascending=False)'),
        ("md", "## Si los precios estuvieran mal\n\n"
               "Cada familia recibe un error de precio independiente de hasta 30 %, "
               "cuatrocientas veces, y se observa cuánto se mueve el total y el orden."),
        ("code", '#| label: tbl-sensibilidad\n'
                 '#| tbl-cap: "Posición en el ranking bajo precios inciertos."\n'
                 'sens = robustez.sensibilidad_precios(prioritarias)\n'
                 'print(f"Total USD {sens.attrs[\'total_base\']/1e3:,.0f} k  "\n'
                 '      f"(p10 {sens.attrs[\'total_p10\']/1e3:,.0f} k · "\n'
                 '      f"p90 {sens.attrs[\'total_p90\']/1e3:,.0f} k)")\n'
                 'sens.round(2)'),
        ("md", "## Dónde poner el corte\n\n"
               "Como el competidor es simulado, se conoce cuáles ventajas son "
               "reales: se puede medir qué encuentra y qué inventa cada umbral."),
        ("code", '#| label: fig-umbral\n'
                 '#| fig-cap: "Precisión y cobertura del filtro según el corte."\n'
                 'fig = viz.fig_umbral(robustez.curva_umbral(d, persist, elast))'),
        ("md", "## Si se vuelve a sortear el mercado\n\n"
               "Quince realizaciones del competidor con las mismas ventajas "
               "sembradas y distinto ruido."),
        ("code", '#| label: fig-semillas\n'
                 '#| fig-cap: "Frecuencia con que cada celda entra en la lista."\n'
                 'est = robustez.estabilidad_semillas(plaza, panel, semillas=15)\n'
                 'print(f"Precisión media {est.precision.mean():.2f} · "\n'
                 '      f"cobertura media {est.cobertura.mean():.2f}")\n'
                 'fig = viz.fig_semillas(est.attrs["frecuencia"], 15)'),
    ],
}


def construir() -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    for nombre, celdas in CUADERNOS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = [nbf.v4.new_markdown_cell(txt) if tipo == "md"
                    else nbf.v4.new_code_cell(txt) for tipo, txt in celdas]
        nb.metadata = {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        }
        ruta = DESTINO / f"{nombre}.ipynb"
        nbf.write(nb, ruta)
        print(f"escrito {ruta.relative_to(RAIZ)}  ({len(celdas)} celdas)")


def ejecutar() -> None:
    """Ejecuta cada cuaderno en su propia carpeta y guarda las salidas."""
    from nbclient import NotebookClient

    for ruta in sorted(DESTINO.glob("*.ipynb")):
        nb = nbf.read(ruta, as_version=4)
        cliente = NotebookClient(nb, timeout=1800, kernel_name="python3",
                                 resources={"metadata": {"path": str(DESTINO)}})
        cliente.execute()
        nbf.write(nb, ruta)
        print(f"ejecutado {ruta.name}")


if __name__ == "__main__":
    import sys

    construir()
    if "--ejecutar" in sys.argv:
        ejecutar()
