"""Dashboard del reto IA aplicada a finanzas (día 5).

Muestra los indicadores de Walmart, Target y Costco (día 1, datos de la SEC) y la
morosidad bancaria de EE. UU. (día 3, datos de la Reserva Federal vía FRED).
Ejecutar desde la raíz del repositorio:
    streamlit run blog/ia-finanzas-5-dashboard/app/app.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Las rutas se resuelven desde este archivo: funciona igual en local y en Community Cloud
DATA = Path(__file__).parent / "data"

COLORES = {"Walmart": "#2a78d6", "Target": "#eb6834", "Costco": "#1baf7a"}
INDICADORES = {
    "Margen neto": ("margen_neto", True),
    "ROA": ("roa", True),
    "ROE": ("roe", True),
    "Razón corriente": ("razon_corriente", False),
    "Pasivo / patrimonio": ("pasivo_patrimonio", False),
}
CATEGORIAS = {
    "DRALACBS": "Total de préstamos",
    "DRCCLACBS": "Tarjetas de crédito",
    "DRCLACBS": "Consumo (total)",
    "DRSFRMACBS": "Hipotecas residenciales",
    "DRCRELEXFACBS": "Inmobiliario comercial",
    "DRBLACBS": "Comerciales e industriales",
}


@st.cache_data
def cargar_empresas():
    df = pd.read_csv(DATA / "indicadores.csv", parse_dates=["cierre_fiscal"])
    return df.sort_values(["empresa", "cierre_fiscal"])


@st.cache_data
def cargar_morosidad():
    df = pd.read_csv(DATA / "morosidad_fred.csv", parse_dates=["trimestre_inicio"])
    return df


def formato(valor, pct):
    if pd.isna(valor):
        return "n.d."
    texto = f"{valor * 100:.1f} %" if pct else f"{valor:.2f}"
    return texto.replace(".", ",")


st.set_page_config(page_title="IA aplicada a finanzas · Dashboard", layout="wide")
st.title("IA aplicada a finanzas · Dashboard")
st.caption("Reto de 5 días · Datos públicos de la SEC y de la Reserva Federal. "
           "No es una recomendación de inversión.")

tab_empresas, tab_credito, tab_ia = st.tabs(["Retailers", "Morosidad bancaria", "Interpretación asistida"])

# --- Pestaña 1: indicadores de empresas ---------------------------------------------
with tab_empresas:
    empresas = cargar_empresas()
    col_filtros, col_ind = st.columns([2, 1])
    elegidas = col_filtros.multiselect("Empresas", list(COLORES), default=list(COLORES))
    nombre_ind = col_ind.selectbox("Indicador", list(INDICADORES))
    columna, pct = INDICADORES[nombre_ind]

    datos = empresas[empresas["empresa"].isin(elegidas)]
    if datos.empty:
        st.info("Elige al menos una empresa.")
    else:
        tarjetas = st.columns(len(elegidas))
        for tarjeta, emp in zip(tarjetas, elegidas):
            serie = datos[datos["empresa"] == emp]
            ultimo, previo = serie[columna].iloc[-1], serie[columna].iloc[-2]
            delta = (ultimo - previo) * (100 if pct else 1)
            tarjeta.metric(
                f"{emp} · cierre {serie['cierre_fiscal'].iloc[-1]:%m/%Y}",
                formato(ultimo, pct),
                (f"{delta:+.1f} pp" if pct else f"{delta:+.2f}").replace(".", ","),
            )

        fig = px.line(datos, x="cierre_fiscal", y=columna, color="empresa", markers=True,
                      color_discrete_map=COLORES,
                      labels={"cierre_fiscal": "Cierre fiscal", columna: nombre_ind, "empresa": "Empresa"})
        if pct:
            fig.update_yaxes(tickformat=".0%")
        fig.update_xaxes(dtick="M12", tickformat="%Y")
        fig.update_layout(legend_title_text="", hovermode="x unified", margin=dict(t=30))
        st.plotly_chart(fig, width="stretch")
        st.caption("Cada punto es un cierre fiscal real: Walmart y Target cierran en enero, "
                   "Costco en agosto. Fuente: 10-K presentados ante la SEC (API de EDGAR).")

        with st.expander("Ver y descargar los datos"):
            st.dataframe(datos, width="stretch", hide_index=True)
            st.download_button("Descargar CSV", datos.to_csv(index=False).encode("utf-8"),
                               "indicadores_retailers.csv", "text/csv")

# --- Pestaña 2: morosidad bancaria ----------------------------------------------------
with tab_credito:
    moro = cargar_morosidad()
    elegidas_cat = st.multiselect("Categorías de préstamo", list(CATEGORIAS.values()),
                                  default=["Total de préstamos", "Tarjetas de crédito"])
    codigos = [c for c, n in CATEGORIAS.items() if n in elegidas_cat]
    if not codigos:
        st.info("Elige al menos una categoría.")
    else:
        largo = moro.melt(id_vars="trimestre_inicio", value_vars=codigos,
                          var_name="serie", value_name="tasa")
        largo["categoria"] = largo["serie"].map(CATEGORIAS)
        fig = px.line(largo, x="trimestre_inicio", y="tasa", color="categoria",
                      labels={"trimestre_inicio": "Trimestre", "tasa": "Tasa de morosidad (%)",
                              "categoria": "Categoría"})
        fig.update_xaxes(dtick="M60", tickformat="%Y")
        fig.update_layout(legend_title_text="", hovermode="x unified", margin=dict(t=30))
        st.plotly_chart(fig, width="stretch")

        resumen = pd.DataFrame([{
            "Categoría": CATEGORIAS[c],
            "Último (%)": moro[c].iloc[-1],
            "Promedio desde 1991 (%)": round(moro[c].mean(), 2),
            "Percentil histórico": round((moro[c] < moro[c].iloc[-1]).mean() * 100),
        } for c in codigos])
        st.dataframe(resumen, width="stretch", hide_index=True)
        st.caption("Préstamos con 30 días o más de atraso que siguen devengando intereses, más "
                   "los que están en no devengo, sobre el saldo de cada categoría. Todos los bancos "
                   f"comerciales, desestacionalizado. Fuente: Reserva Federal vía FRED, descargado "
                   f"el {moro['fecha_descarga'].iloc[0]}.")

# --- Pestaña 3: prompt para interpretación asistida ----------------------------------
with tab_ia:
    st.markdown(
        "Esta pestaña **no llama a ningún modelo de IA**: arma un prompt con los datos que "
        "estás viendo para que lo pegues en el asistente que uses. La interpretación que "
        "obtengas es asistida: verifica cada afirmación contra los datos y las fuentes."
    )
    empresas = cargar_empresas()
    ultimos = (empresas.groupby("empresa").tail(1)
                       .set_index("empresa")[[c for c, _ in INDICADORES.values()]]
                       .round(4))
    prompt = (
        "Eres analista financiero. Estos son los indicadores del último año fiscal de tres "
        "retailers de EE. UU., calculados a partir de sus 10-K (proporciones, no porcentajes):\n\n"
        f"{ultimos.to_csv()}\n"
        "1. Compara sus modelos de negocio usando ROA = margen neto x rotación de activos.\n"
        "2. Señala qué indicador merece seguimiento en cada empresa y por qué.\n"
        "Reglas: usa solo estos datos; marca como 'por verificar' cualquier afirmación que "
        "requiera información externa; no hagas recomendaciones de inversión."
    )
    st.code(prompt, language="text")
