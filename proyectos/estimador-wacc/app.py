"""
Estimador de WACC para mercados seleccionados
=============================================

Aplicación web interactiva (Streamlit) para estimar el Costo Promedio Ponderado
de Capital (WACC) de una empresa hipotética, según el país, la industria y la
estructura de capital seleccionada por el usuario.

Herramienta educativa y demostrativa. Todos los datos incorporados son
ILUSTRATIVOS y deben reemplazarse por los valores vigentes de Damodaran antes
de usar la aplicación para cualquier análisis real.

Ejecución:
    streamlit run app.py

Autora: Ana Laura Marín Sánchez
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Estimador de WACC",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# ▼▼▼  SECCIÓN EDITABLE — DATOS ILUSTRATIVOS  ▼▼▼
# =============================================================================
# IMPORTANTE
# ----------
# Los valores siguientes son ILUSTRATIVOS y NO representan datos oficiales.
# Antes de utilizar esta herramienta para un análisis real, sustitúyalos por
# los datos más recientes publicados por el profesor Aswath Damodaran en:
#   https://pages.stern.nyu.edu/~adamodar/  (Data > Current Data)
#
#   - Betas por industria (levered/unlevered):  "Betas by Sector"
#   - Primas de riesgo país (Country Risk Premium): "Country Default Spreads
#     and Risk Premiums"
#   - Prima de riesgo del mercado maduro (Implied ERP) y tasa libre de riesgo.
#
# Solo debe editar los diccionarios y valores de esta sección.
# =============================================================================

# --- Supuestos globales (moneda de valoración: dólares estadounidenses, USD) --
RISK_FREE_RATE = 0.0425          # Tasa libre de riesgo (ilustrativa)  ->  4.25%
MATURE_MARKET_RISK_PREMIUM = 0.050  # Prima de riesgo mercado maduro (ilustr.) -> 5.00%

# --- Prima de riesgo país (Country Risk Premium, CRP) -------------------------
# Estados Unidos se toma como mercado maduro de referencia: CRP = 0%.
COUNTRY_RISK_PREMIUM = {
    "Costa Rica": 0.0450,        # 4.50% (ilustrativo)
    "Guatemala": 0.0620,         # 6.20% (ilustrativo)
    "Estados Unidos": 0.0000,    # 0.00% (mercado maduro de referencia)
}

# --- Betas por industria (ilustrativos) ---------------------------------------
# levered_beta   = beta apalancado promedio observado en la industria.
# unlevered_beta = beta desapalancado de la industria (corregido por deuda/caja).
INDUSTRY_BETAS = {
    "Software":            {"levered_beta": 1.20, "unlevered_beta": 1.05},
    "Alimentos procesados": {"levered_beta": 0.75, "unlevered_beta": 0.62},
    "Bancos":              {"levered_beta": 1.10, "unlevered_beta": 0.55},
}
# =============================================================================
# ▲▲▲  FIN DE LA SECCIÓN EDITABLE  ▲▲▲
# =============================================================================


# =============================================================================
# CARGA DE DATOS
# =============================================================================
def load_country_data() -> pd.DataFrame:
    """Devuelve la tabla interna de primas de riesgo país (editable arriba)."""
    df = pd.DataFrame(
        [{"pais": k, "prima_riesgo_pais": v} for k, v in COUNTRY_RISK_PREMIUM.items()]
    )
    return df


def load_industry_data() -> pd.DataFrame:
    """Devuelve la tabla interna de betas por industria (editable arriba)."""
    df = pd.DataFrame(
        [
            {
                "industria": k,
                "beta_apalancado": v["levered_beta"],
                "beta_desapalancado": v["unlevered_beta"],
            }
            for k, v in INDUSTRY_BETAS.items()
        ]
    )
    return df


# =============================================================================
# FUNCIONES DE CÁLCULO
# =============================================================================
def calculate_debt_to_equity(debt_weight: float, equity_weight: float) -> float:
    """
    Razón deuda-patrimonio (D/E).

        D/E = porcentaje de deuda / porcentaje de patrimonio

    Los pesos deben expresarse en decimales (p. ej. 0.40 y 0.60).
    """
    if equity_weight <= 0:
        raise ValueError("El patrimonio no puede ser cero: D/E quedaría indefinido.")
    return debt_weight / equity_weight


def calculate_levered_beta(
    unlevered_beta: float, tax_rate: float, debt_to_equity: float
) -> float:
    """
    Beta apalancado ajustado a la estructura de capital (fórmula de Hamada).

        β_L = β_U * [1 + (1 - t) * (D/E)]
    """
    return unlevered_beta * (1 + (1 - tax_rate) * debt_to_equity)


def calculate_cost_of_equity(
    risk_free_rate: float,
    levered_beta: float,
    market_risk_premium: float,
    country_risk_premium: float,
) -> float:
    """
    Costo del patrimonio (CAPM con prima de riesgo país aditiva).

        Ke = Rf + β_L * PRM + CRP

    La prima de riesgo país (CRP) se suma de forma aditiva y NO se multiplica
    por el beta.
    """
    return risk_free_rate + levered_beta * market_risk_premium + country_risk_premium


def calculate_after_tax_cost_of_debt(
    pretax_cost_of_debt: float, tax_rate: float
) -> float:
    """
    Costo de la deuda después de impuestos (escudo fiscal).

        Kd_dt = Kd * (1 - t)
    """
    return pretax_cost_of_debt * (1 - tax_rate)


def calculate_wacc(
    cost_of_equity: float,
    equity_weight: float,
    pretax_cost_of_debt: float,
    tax_rate: float,
    debt_weight: float,
) -> float:
    """
    Costo Promedio Ponderado de Capital (WACC).

        WACC = Ke * We + Kd * (1 - t) * Wd

    Todos los pesos y tasas deben venir en decimales.
    """
    return (
        cost_of_equity * equity_weight
        + pretax_cost_of_debt * (1 - tax_rate) * debt_weight
    )


def build_debt_scenario_table(
    unlevered_beta: float,
    tax_rate: float,
    pretax_cost_of_debt: float,
    risk_free_rate: float,
    market_risk_premium: float,
    country_risk_premium: float,
    debt_levels: list,
) -> pd.DataFrame:
    """
    Construye la tabla de escenarios de WACC para distintos niveles de deuda.

    Mantiene constantes país, industria, tasa impositiva, costo de deuda, tasa
    libre de riesgo, prima de mercado y prima de riesgo país. Para cada nivel de
    deuda recalcula patrimonio, D/E, beta apalancado, costo del patrimonio y WACC.

    Nota metodológica: el costo de la deuda se mantiene CONSTANTE en la
    simulación, aunque en la práctica podría aumentar con el apalancamiento.
    """
    filas = []
    for debt_weight in debt_levels:
        equity_weight = 1 - debt_weight
        de = calculate_debt_to_equity(debt_weight, equity_weight)
        beta_l = calculate_levered_beta(unlevered_beta, tax_rate, de)
        ke = calculate_cost_of_equity(
            risk_free_rate, beta_l, market_risk_premium, country_risk_premium
        )
        wacc = calculate_wacc(
            ke, equity_weight, pretax_cost_of_debt, tax_rate, debt_weight
        )
        filas.append(
            {
                "% Deuda": debt_weight,
                "% Patrimonio": equity_weight,
                "D/E": de,
                "Beta apalancado": beta_l,
                "Costo del patrimonio": ke,
                "WACC": wacc,
            }
        )
    return pd.DataFrame(filas)


def generate_interpretation(
    pais: str,
    industria: str,
    debt_weight: float,
    equity_weight: float,
    unlevered_beta: float,
    levered_beta: float,
    cost_of_equity: float,
    after_tax_cost_of_debt: float,
    pretax_cost_of_debt: float,
    country_risk_premium: float,
    tax_rate: float,
    wacc: float,
) -> str:
    """Genera un texto dinámico que interpreta el resultado (sin juzgar óptimo)."""
    # ¿Ke > Kd?
    if cost_of_equity > pretax_cost_of_debt:
        comparacion = (
            f"El **costo del patrimonio** ({cost_of_equity * 100:.2f}%) es **mayor** "
            f"que el costo de la deuda antes de impuestos ({pretax_cost_of_debt * 100:.2f}%), "
            "lo cual es habitual porque los accionistas asumen más riesgo que los acreedores."
        )
    else:
        comparacion = (
            f"En este escenario el costo del patrimonio ({cost_of_equity * 100:.2f}%) "
            f"resulta **igual o menor** que el costo de la deuda antes de impuestos "
            f"({pretax_cost_of_debt * 100:.2f}%), una situación poco común que conviene revisar."
        )

    incremento_beta = levered_beta - unlevered_beta
    aporte_pais = country_risk_premium  # aporte directo al Ke (aditivo)
    escudo = pretax_cost_of_debt - after_tax_cost_of_debt

    texto = f"""
El **WACC estimado** para una empresa de la industria de **{industria}** en
**{pais}**, con una estructura de **{debt_weight * 100:.0f}% de deuda** y
**{equity_weight * 100:.0f}% de patrimonio**, es de **{wacc * 100:.2f}%**
(en dólares estadounidenses).

- {comparacion}
- El apalancamiento seleccionado eleva el beta desde **{unlevered_beta:.2f}**
  (desapalancado) hasta **{levered_beta:.2f}** (apalancado ajustado), un aumento
  de **{incremento_beta:.2f}**. A mayor deuda, mayor beta apalancado y, por tanto,
  mayor costo del patrimonio.
- La **prima de riesgo país** aporta **{aporte_pais * 100:.2f} puntos porcentuales**
  de forma directa al costo del patrimonio (se suma, no se multiplica por el beta).
- El **beneficio fiscal de la deuda** reduce su costo desde
  **{pretax_cost_of_debt * 100:.2f}%** (antes de impuestos) hasta
  **{after_tax_cost_of_debt * 100:.2f}%** (después de impuestos), es decir,
  un ahorro de **{escudo * 100:.2f} p.p.** gracias al escudo fiscal con una tasa
  impositiva del **{tax_rate * 100:.0f}%**.
"""
    return texto.strip()


# =============================================================================
# COMPONENTES DE INTERFAZ
# =============================================================================
def render_header() -> None:
    """Encabezado con título, subtítulo y explicación de la herramienta."""
    st.title("Estimador de WACC para mercados seleccionados")
    st.subheader(
        "Simulación del costo promedio ponderado de capital según país, "
        "industria y estructura de financiamiento."
    )
    st.markdown(
        """
Esta herramienta combina **beta sectorial**, **apalancamiento financiero**,
**prima de riesgo país**, **costo de deuda** y **escudo fiscal** para estimar
el WACC de una empresa hipotética. La moneda de valoración se asume en
**dólares estadounidenses (USD)**.
        """
    )


def render_sidebar():
    """
    Dibuja la barra lateral con todos los inputs del usuario.

    Devuelve una tupla con los valores capturados (en decimales las tasas/pesos).
    """
    st.sidebar.header("Parámetros de entrada")

    # --- País ---------------------------------------------------------------
    pais = st.sidebar.selectbox(
        "País",
        options=list(COUNTRY_RISK_PREMIUM.keys()),
        index=0,
        help="Determina la prima de riesgo país aplicada al costo del patrimonio.",
    )

    # --- Industria ----------------------------------------------------------
    industria = st.sidebar.selectbox(
        "Industria",
        options=list(INDUSTRY_BETAS.keys()),
        index=0,
        help="Determina el beta sectorial (apalancado y desapalancado).",
    )

    # --- Porcentaje de deuda (sin 100%) ------------------------------------
    debt_pct = st.sidebar.select_slider(
        "Porcentaje de deuda",
        options=[10, 20, 30, 40, 50, 60, 70, 80, 90],
        value=40,
        format_func=lambda x: f"{x}%",
        help="No se incluye 100% porque el patrimonio sería cero y D/E quedaría indefinido.",
    )

    # --- Porcentaje de patrimonio (informativo, no editable) ---------------
    equity_pct = 100 - debt_pct
    st.sidebar.metric("Porcentaje de patrimonio (calculado)", f"{equity_pct}%")
    st.sidebar.caption("Patrimonio = 100% − Deuda")

    st.sidebar.divider()

    # --- Tasa impositiva ----------------------------------------------------
    tax_pct = st.sidebar.number_input(
        "Tasa impositiva (%)",
        min_value=0.0,
        max_value=50.0,
        value=30.0,
        step=1.0,
        help="Entre 0% y 50%.",
    )

    # --- Costo de la deuda antes de impuestos ------------------------------
    kd_pct = st.sidebar.number_input(
        "Costo de la deuda antes de impuestos (%)",
        min_value=0.0,
        max_value=30.0,
        value=7.0,
        step=0.5,
        help="Entre 0% y 30%.",
    )

    return pais, industria, debt_pct, equity_pct, tax_pct, kd_pct


def render_assumptions_box(country_risk_premium: float) -> None:
    """Muestra los supuestos globales vigentes (editables en el código)."""
    with st.sidebar:
        st.divider()
        st.caption("**Supuestos globales (editables en el código)**")
        st.caption(f"Tasa libre de riesgo: {RISK_FREE_RATE * 100:.2f}%")
        st.caption(f"Prima de mercado maduro: {MATURE_MARKET_RISK_PREMIUM * 100:.2f}%")
        st.caption(f"Prima de riesgo país: {country_risk_premium * 100:.2f}%")
        st.caption("Moneda: USD · Datos ilustrativos (actualizar con Damodaran).")


def validate_inputs(debt_pct, equity_pct, tax_pct, kd_pct) -> list:
    """
    Valida los inputs y devuelve una lista de mensajes de error (vacía si todo OK).

    Reglas:
      - Deuda + patrimonio = 100%.
      - Patrimonio no puede ser cero.
      - Tasa impositiva entre 0% y 50%.
      - Costo de deuda entre 0% y 30%.
    """
    errores = []
    if debt_pct + equity_pct != 100:
        errores.append("La deuda y el patrimonio deben sumar 100%.")
    if equity_pct <= 0:
        errores.append("El patrimonio no puede ser cero.")
    if not (0 <= tax_pct <= 50):
        errores.append("La tasa impositiva debe estar entre 0% y 50%.")
    if not (0 <= kd_pct <= 30):
        errores.append("El costo de la deuda debe estar entre 0% y 30%.")
    return errores


def render_result_cards(
    beta_apalancado_industria: float,
    beta_desapalancado: float,
    beta_apalancado_ajustado: float,
    country_risk_premium: float,
    cost_of_equity: float,
    after_tax_cost_of_debt: float,
    wacc: float,
) -> None:
    """Tarjetas (métricas) con los siete indicadores solicitados."""
    st.subheader("Resultados")

    fila1 = st.columns(4)
    fila1[0].metric("Beta apalancado (industria)", f"{beta_apalancado_industria:.2f}")
    fila1[1].metric("Beta desapalancado (industria)", f"{beta_desapalancado:.2f}")
    fila1[2].metric("Beta apalancado ajustado", f"{beta_apalancado_ajustado:.2f}")
    fila1[3].metric("Prima de riesgo país", f"{country_risk_premium * 100:.2f}%")

    fila2 = st.columns(3)
    fila2[0].metric("Costo del patrimonio", f"{cost_of_equity * 100:.2f}%")
    fila2[1].metric(
        "Costo de deuda después de impuestos", f"{after_tax_cost_of_debt * 100:.2f}%"
    )
    fila2[2].metric("WACC", f"{wacc * 100:.2f}%")


def render_wacc_by_debt_chart(scenario_df: pd.DataFrame, debt_pct: int) -> None:
    """Visualización 1: WACC según % de deuda, resaltando el nivel seleccionado."""
    st.subheader("WACC según porcentaje de deuda")

    x = scenario_df["% Deuda"] * 100
    y = scenario_df["WACC"] * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name="WACC",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=8),
            hovertemplate="Deuda %{x:.0f}%<br>WACC %{y:.2f}%<extra></extra>",
        )
    )

    # Resaltar el nivel de deuda seleccionado por el usuario.
    sel = scenario_df.loc[scenario_df["% Deuda"] == debt_pct / 100]
    if not sel.empty:
        fig.add_trace(
            go.Scatter(
                x=[debt_pct],
                y=[sel["WACC"].iloc[0] * 100],
                mode="markers",
                name="Nivel seleccionado",
                marker=dict(size=16, color="#dc2626", symbol="circle-open", line=dict(width=3)),
                hovertemplate="Seleccionado<br>Deuda %{x:.0f}%<br>WACC %{y:.2f}%<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Porcentaje de deuda",
        yaxis_title="WACC (%)",
        xaxis=dict(tickmode="array", tickvals=list(range(10, 100, 10)), ticksuffix="%"),
        yaxis=dict(ticksuffix="%"),
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Nota metodológica: el costo de la deuda se mantiene **constante** en la "
        "simulación, aunque en la práctica podría aumentar con el apalancamiento."
    )


def render_wacc_composition_chart(
    equity_contribution: float, debt_contribution: float, wacc: float
) -> None:
    """Visualización 2: composición del WACC (patrimonio vs. deuda)."""
    st.subheader("Composición del WACC")

    fig = go.Figure(
        data=[
            go.Bar(
                x=["Patrimonio", "Deuda", "WACC total"],
                y=[
                    equity_contribution * 100,
                    debt_contribution * 100,
                    wacc * 100,
                ],
                marker_color=["#2563eb", "#f59e0b", "#16a34a"],
                text=[
                    f"{equity_contribution * 100:.2f}%",
                    f"{debt_contribution * 100:.2f}%",
                    f"{wacc * 100:.2f}%",
                ],
                textposition="outside",
                hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        yaxis_title="Contribución al WACC (%)",
        yaxis=dict(ticksuffix="%"),
        template="plotly_white",
        height=420,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Patrimonio = costo del patrimonio × % patrimonio · "
        "Deuda = costo de deuda después de impuestos × % deuda. "
        "La suma de ambas contribuciones equivale al WACC."
    )


def render_summary_table(datos: dict) -> None:
    """Tabla de resumen con todos los parámetros y resultados del escenario."""
    st.subheader("Tabla de resumen")
    df = pd.DataFrame(
        {
            "Concepto": list(datos.keys()),
            "Valor": list(datos.values()),
        }
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_methodology() -> None:
    """Sección expandible con fórmulas y explicación de variables."""
    with st.expander("Información metodológica (fórmulas y variables)"):
        st.markdown(
            r"""
**Razón deuda-patrimonio**

$$ D/E = \frac{\%\,\text{Deuda}}{\%\,\text{Patrimonio}} $$

**Beta apalancado ajustado (fórmula de Hamada)**

$$ \beta_L = \beta_U \left[ 1 + (1 - t)\,\frac{D}{E} \right] $$

**Costo del patrimonio (CAPM + prima de riesgo país aditiva)**

$$ K_e = R_f + \beta_L \cdot PRM + CRP $$

**Costo de la deuda después de impuestos**

$$ K_d^{dt} = K_d \,(1 - t) $$

**WACC**

$$ WACC = K_e \cdot W_e + K_d \,(1 - t)\, W_d $$

**Variables**

- $\beta_U$: beta desapalancado de la industria.
- $\beta_L$: beta apalancado ajustado a la estructura de capital.
- $t$: tasa impositiva.
- $D/E$: razón deuda-patrimonio.
- $R_f$: tasa libre de riesgo.
- $PRM$: prima de riesgo del mercado maduro.
- $CRP$: prima de riesgo país (se suma, no se multiplica por el beta).
- $K_e$: costo del patrimonio.
- $K_d$: costo de la deuda antes de impuestos.
- $W_e$, $W_d$: pesos de patrimonio y deuda (suman 100%).
            """
        )


def render_disclaimer() -> None:
    """Disclaimer y aclaraciones sobre las limitaciones del modelo."""
    st.divider()
    st.markdown("#### Disclaimer")
    st.info(
        "Esta herramienta tiene fines educativos y demostrativos. Los resultados "
        "se basan en datos sectoriales y supuestos simplificados, por lo que no "
        "constituyen asesoría financiera ni una estimación definitiva para una "
        "empresa específica. El costo de capital puede variar según la moneda de "
        "valoración, estructura de financiamiento, exposición geográfica, riesgo "
        "crediticio y características particulares de cada compañía."
    )
    st.markdown(
        """
Además tenga en cuenta que:

- Los datos sectoriales y las primas de riesgo deben actualizarse periódicamente.
- El costo de la deuda se mantiene fijo en el análisis de escenarios.
- El modelo **no** incorpora una prima por tamaño.
- El modelo **no** incorpora una lambda específica de exposición al riesgo país.
- El modelo **no** estima un costo de deuda sintético.
- El modelo **no** sustituye un análisis financiero completo.
        """
    )


# =============================================================================
# APLICACIÓN PRINCIPAL
# =============================================================================
def main() -> None:
    render_header()

    # --- Inputs -------------------------------------------------------------
    pais, industria, debt_pct, equity_pct, tax_pct, kd_pct = render_sidebar()

    # --- Validaciones -------------------------------------------------------
    errores = validate_inputs(debt_pct, equity_pct, tax_pct, kd_pct)
    if errores:
        st.error("No es posible calcular el WACC. Corrija lo siguiente:")
        for e in errores:
            st.markdown(f"- {e}")
        return  # El WACC solo se muestra cuando los inputs son válidos.

    # --- Conversión de porcentajes a decimales ------------------------------
    debt_weight = debt_pct / 100
    equity_weight = equity_pct / 100
    tax_rate = tax_pct / 100
    pretax_cost_of_debt = kd_pct / 100

    # --- Datos sectoriales y de país ---------------------------------------
    betas = INDUSTRY_BETAS[industria]
    beta_apalancado_industria = betas["levered_beta"]
    beta_desapalancado = betas["unlevered_beta"]
    country_risk_premium = COUNTRY_RISK_PREMIUM[pais]

    render_assumptions_box(country_risk_premium)

    # --- Cálculos principales ----------------------------------------------
    de_ratio = calculate_debt_to_equity(debt_weight, equity_weight)
    beta_apalancado_ajustado = calculate_levered_beta(
        beta_desapalancado, tax_rate, de_ratio
    )
    cost_of_equity = calculate_cost_of_equity(
        RISK_FREE_RATE,
        beta_apalancado_ajustado,
        MATURE_MARKET_RISK_PREMIUM,
        country_risk_premium,
    )
    after_tax_cost_of_debt = calculate_after_tax_cost_of_debt(
        pretax_cost_of_debt, tax_rate
    )
    wacc = calculate_wacc(
        cost_of_equity, equity_weight, pretax_cost_of_debt, tax_rate, debt_weight
    )

    # Contribuciones al WACC (suman el WACC).
    equity_contribution = cost_of_equity * equity_weight
    debt_contribution = after_tax_cost_of_debt * debt_weight

    # --- Tarjetas de resultados --------------------------------------------
    render_result_cards(
        beta_apalancado_industria,
        beta_desapalancado,
        beta_apalancado_ajustado,
        country_risk_premium,
        cost_of_equity,
        after_tax_cost_of_debt,
        wacc,
    )

    st.divider()

    # --- Visualizaciones ----------------------------------------------------
    col_izq, col_der = st.columns(2)

    scenario_df = build_debt_scenario_table(
        unlevered_beta=beta_desapalancado,
        tax_rate=tax_rate,
        pretax_cost_of_debt=pretax_cost_of_debt,
        risk_free_rate=RISK_FREE_RATE,
        market_risk_premium=MATURE_MARKET_RISK_PREMIUM,
        country_risk_premium=country_risk_premium,
        debt_levels=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    )

    with col_izq:
        render_wacc_by_debt_chart(scenario_df, debt_pct)
    with col_der:
        render_wacc_composition_chart(equity_contribution, debt_contribution, wacc)

    st.divider()

    # --- Tabla de resumen ---------------------------------------------------
    resumen = {
        "País": pais,
        "Industria": industria,
        "Porcentaje de deuda": f"{debt_pct:.0f}%",
        "Porcentaje de patrimonio": f"{equity_pct:.0f}%",
        "D/E": f"{de_ratio:.2f}",
        "Tasa impositiva": f"{tax_pct:.0f}%",
        "Costo de deuda antes de impuestos": f"{pretax_cost_of_debt * 100:.2f}%",
        "Costo de deuda después de impuestos": f"{after_tax_cost_of_debt * 100:.2f}%",
        "Beta desapalancado": f"{beta_desapalancado:.2f}",
        "Beta apalancado ajustado": f"{beta_apalancado_ajustado:.2f}",
        "Prima de riesgo país": f"{country_risk_premium * 100:.2f}%",
        "Costo del patrimonio": f"{cost_of_equity * 100:.2f}%",
        "WACC": f"{wacc * 100:.2f}%",
    }
    render_summary_table(resumen)

    st.divider()

    # --- Interpretación automática -----------------------------------------
    st.subheader("Interpretación del resultado")
    st.markdown(
        generate_interpretation(
            pais=pais,
            industria=industria,
            debt_weight=debt_weight,
            equity_weight=equity_weight,
            unlevered_beta=beta_desapalancado,
            levered_beta=beta_apalancado_ajustado,
            cost_of_equity=cost_of_equity,
            after_tax_cost_of_debt=after_tax_cost_of_debt,
            pretax_cost_of_debt=pretax_cost_of_debt,
            country_risk_premium=country_risk_premium,
            tax_rate=tax_rate,
            wacc=wacc,
        )
    )

    # --- Metodología y disclaimer ------------------------------------------
    st.divider()
    render_methodology()
    render_disclaimer()


if __name__ == "__main__":
    main()
