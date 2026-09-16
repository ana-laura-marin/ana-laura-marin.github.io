"""Reproduce el analisis completo de punta a punta.

    python run_pipeline.py                # analisis y figuras
    python run_pipeline.py --semillas 25  # mas realizaciones del competidor
    python run_pipeline.py --sin-robustez # salta la parte lenta

Deja las figuras en data/figuras/ y las tablas en data/derivados/.
"""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt

from src import benchmark, brechas, data_io, robustez, viz
from src import config as cfg


def _titulo(txt: str) -> None:
    print(f"\n{'=' * 68}\n{txt}\n{'=' * 68}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=15)
    ap.add_argument("--sin-robustez", action="store_true")
    ap.add_argument("--forzar", action="store_true",
                    help="Reconstruye el panel aunque exista el cache")
    args = ap.parse_args()

    t0 = time.time()
    cfg.FIGURAS.mkdir(parents=True, exist_ok=True)
    viz.aplicar_estilo()

    def guardar(fig, nombre: str) -> None:
        fig.savefig(cfg.FIGURAS / f"{nombre}.png")
        plt.close(fig)

    # ------------------------------------------------------------ 1. datos
    _titulo("1. Panel de ventas")
    panel = data_io.construir_panel(forzar=args.forzar)
    print(f"{len(panel):,} filas mes x tienda x familia  |  "
          f"{panel.mes.min():%Y-%m} a {panel.mes.max():%Y-%m}")
    print(f"Ingreso simulado del ultimo ano completo: "
          f"USD {panel[panel.mes.dt.year == 2016].ingreso.sum()/1e6:,.0f} M")
    guardar(viz.fig_concentracion(panel), "concentracion")

    # ------------------------------------------------------- 2. competidor
    _titulo("2. Competidor de referencia")
    plaza = data_io.panel_plaza(panel)
    bench = benchmark.generar(plaza)
    share = 100 * bench.unidades.sum() / bench.mercado_unidades.sum()
    print(f"Share medio de Favorita en el mercado simulado: {share:.1f} %")
    print(f"Ventajas sembradas: {len(benchmark.INTERACCIONES)}")
    guardar(viz.fig_share_familia(bench), "share_familia")

    # ---------------------------------------------------------- 3. brechas
    _titulo("3. Descomposicion de la brecha")
    ventana = brechas.ventana(bench, 12)
    material = brechas.filtrar_material(ventana)
    d = brechas.descomponer(material)
    print(f"Celdas plaza x familia: {len(ventana)} -> {len(material)} materiales")
    print(f"R2 ponderado del modelo aditivo: {d.attrs['r2']:.3f}")
    guardar(viz.fig_mapa_brechas(d), "mapa_brechas")

    res_mens = brechas.residuos_mensuales(bench, 24)
    persist = brechas.persistencia(res_mens)
    elast = brechas.elasticidad_promo(panel)
    r = brechas.ranking(d, persist, elast)

    emb = brechas.embudo(r)
    print()
    print(emb.to_string(index=False, float_format=lambda x: f"{x:,.0f}"))
    guardar(viz.fig_embudo(emb), "embudo")
    guardar(viz.fig_elasticidad(elast, familias=d.family.unique()), "elasticidad")

    prior = r[r.prioritaria].sort_values("oport_contribucion", ascending=False)
    celdas = list(zip(prior.city, prior.family))[:4]
    descartada = None
    sobrantes = r[(r.material) & (~r.prioritaria) & (r.oport_contribucion > 0)]
    if len(sobrantes):
        peor = sobrantes.sort_values("oport_contribucion", ascending=False).iloc[0]
        descartada = (peor.city, peor.family)
    guardar(viz.fig_persistencia(res_mens, celdas, descartada), "persistencia")
    guardar(viz.fig_valor(prior), "valor")

    _titulo("4. Oportunidades prioritarias")
    cols = ["city", "family", "share", "share_esperado", "brecha_pp",
            "oport_ingreso", "oport_contribucion", "t_ajustado", "elasticidad"]
    print(prior[cols].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # --------------------------------------------------------- 5. robustez
    if not args.sin_robustez:
        _titulo("5. Robustez")
        sens = robustez.sensibilidad_precios(prior)
        print(f"Contribucion recuperable: USD {sens.attrs['total_base']/1e3:,.0f} k  "
              f"(p10 {sens.attrs['total_p10']/1e3:,.0f} k - "
              f"p90 {sens.attrs['total_p90']/1e3:,.0f} k con precios inciertos)")

        curva = robustez.curva_umbral(d, persist, elast)
        guardar(viz.fig_umbral(curva), "umbral")
        print()
        print(curva.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

        est = robustez.estabilidad_semillas(plaza, panel, semillas=args.semillas)
        frec = est.attrs["frecuencia"]
        guardar(viz.fig_semillas(frec, args.semillas), "semillas")
        print()
        print(f"Precision media {est.precision.mean():.2f} · "
              f"cobertura media {est.cobertura.mean():.2f} "
              f"sobre {args.semillas} realizaciones")
        print(frec.head(10).to_string(index=False))
        frec.to_parquet(cfg.DERIVADOS / "frecuencia_celdas.parquet", index=False)

    # ---------------------------------------------------------- 6. guardar
    d.to_parquet(cfg.DERIVADOS / "brechas.parquet", index=False)
    r.drop(columns=[c for c in ["eta"] if c in r.columns]).to_parquet(
        cfg.DERIVADOS / "ranking.parquet", index=False)
    elast.to_parquet(cfg.DERIVADOS / "elasticidad.parquet", index=False)
    print(f"\nListo en {time.time() - t0:,.0f} s · figuras en {cfg.FIGURAS}")


if __name__ == "__main__":
    main()
