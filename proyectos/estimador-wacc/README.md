# Estimador de WACC para mercados seleccionados

Aplicación web interactiva construida con **Streamlit** para estimar el **Costo
Promedio Ponderado de Capital (WACC)** de una empresa hipotética, en función del
país, la industria y la estructura de capital seleccionada por el usuario.

## Descripción del proyecto

La herramienta permite explorar, de forma visual e interactiva, cómo cambia el
costo de capital al modificar el país, la industria y el nivel de apalancamiento.
Combina el beta sectorial, el apalancamiento financiero (fórmula de Hamada), la
prima de riesgo país, el costo de la deuda y el escudo fiscal.

La moneda de valoración se asume en **dólares estadounidenses (USD)**.

## Objetivo

Ofrecer una demo pública y educativa que muestre, paso a paso, la construcción
del WACC:

- Beta apalancado promedio de la industria.
- Beta desapalancado de la industria.
- Beta apalancado ajustado a la estructura de capital seleccionada.
- Prima de riesgo país.
- Costo del patrimonio.
- Costo de la deuda después de impuestos.
- WACC.

## Metodología

1. A partir de la estructura de capital elegida (% de deuda), se calcula el
   patrimonio como `100% − deuda` y la razón deuda-patrimonio `D/E`.
2. El beta desapalancado de la industria se re-apalanca con la fórmula de Hamada
   usando el `D/E` y la tasa impositiva.
3. El costo del patrimonio se obtiene con el CAPM, sumando de forma **aditiva**
   la prima de riesgo país.
4. El costo de la deuda después de impuestos aplica el escudo fiscal.
5. El WACC pondera ambos costos por sus respectivos pesos.

Se genera además un análisis de escenarios que recalcula el WACC para niveles de
deuda de 10% a 90%, manteniendo constantes el resto de supuestos.

## Fórmulas utilizadas

```text
D/E            = % Deuda / % Patrimonio
Beta_L         = Beta_U * [ 1 + (1 - t) * (D/E) ]                (Hamada)
Costo_equity   = Rf + Beta_L * PRM + CRP
Costo_deuda_dt = Kd * (1 - t)
WACC           = Costo_equity * We + Kd * (1 - t) * Wd
```

Donde:

- `Beta_U`  = beta desapalancado de la industria.
- `Beta_L`  = beta apalancado ajustado.
- `t`       = tasa impositiva.
- `Rf`      = tasa libre de riesgo.
- `PRM`     = prima de riesgo del mercado maduro.
- `CRP`     = prima de riesgo país (se suma, no se multiplica por el beta).
- `Kd`      = costo de la deuda antes de impuestos.
- `We`, `Wd`= pesos de patrimonio y deuda (suman 100%).

## Estructura de archivos

```text
estimador-wacc/
├── app.py            # Aplicación Streamlit (lógica, cálculos y dashboard)
├── requirements.txt  # Dependencias
└── README.md         # Este archivo
```

## Instalación

Requiere **Python 3.9+**. Se recomienda un entorno virtual:

```bash
# 1. Crear y activar un entorno virtual
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt
```

## Ejecución

```bash
streamlit run app.py
```

La aplicación se abrirá en el navegador (por defecto en `http://localhost:8501`).

## Despliegue en Streamlit Community Cloud

La app puede publicarse gratis en [Streamlit Community Cloud](https://share.streamlit.io)
para obtener una URL pública (p. ej. `https://estimador-wacc-analaura.streamlit.app`,
la que enlaza el botón *"Abrir demo"* de la página del portafolio).

Pasos:

1. Asegúrate de que este proyecto esté **subido a GitHub** (ya vive en el repo del
   sitio: `proyectos/estimador-wacc/`).
2. Entra a <https://share.streamlit.io> e inicia sesión con GitHub (autoriza el
   acceso al repositorio la primera vez).
3. Haz clic en **"Create app" → "Deploy a public app from GitHub"** y completa:
   - **Repository:** `ana-laura-marin/ana-laura-marin.github.io`
   - **Branch:** `main`
   - **Main file path:** `proyectos/estimador-wacc/app.py`
   - **App URL (subdominio):** `estimador-wacc-analaura`
     (debe coincidir con el botón de la página; si está ocupado, elige otro y
     actualiza el enlace en `index.qmd`).
4. **Advanced settings → Python version:** selecciona **3.11** (o 3.12).
5. Haz clic en **Deploy**. El primer arranque tarda 1–3 minutos.

> **Importante — dependencias.** Streamlit Cloud instala las dependencias del
> `requirements.txt` que está **junto al `app.py`** (en esta misma carpeta). La
> raíz del repositorio contiene un `pyproject.toml`/`uv.lock` de **otro** proyecto
> del sitio; al fijar el *Main file path* dentro de esta carpeta, Streamlit usa el
> `requirements.txt` local y no esas dependencias. Si el despliegue fallara por
> resolución de dependencias, la causa más probable es esa: confirma que el *Main
> file path* apunta exactamente a `proyectos/estimador-wacc/app.py`.

Para actualizar la demo, basta con hacer `git push`: Streamlit Cloud redepliega
automáticamente.

## Dependencias

- [Streamlit](https://streamlit.io/) — interfaz web interactiva.
- [Pandas](https://pandas.pydata.org/) — manejo de tablas y escenarios.
- [Plotly](https://plotly.com/python/) — visualizaciones interactivas.

Versiones mínimas en [`requirements.txt`](requirements.txt).

## Dónde actualizar los datos de Damodaran

Todos los datos incorporados son **ILUSTRATIVOS**. Para usar la herramienta con
datos reales, edite únicamente la sección claramente marcada en `app.py`:

```text
# ▼▼▼  SECCIÓN EDITABLE — DATOS ILUSTRATIVOS  ▼▼▼
```

Allí encontrará:

- `RISK_FREE_RATE` — tasa libre de riesgo.
- `MATURE_MARKET_RISK_PREMIUM` — prima de riesgo del mercado maduro (ERP).
- `COUNTRY_RISK_PREMIUM` — prima de riesgo país (Estados Unidos = 0%).
- `INDUSTRY_BETAS` — beta apalancado y desapalancado por industria.

Fuente recomendada: Aswath Damodaran (NYU Stern),
<https://pages.stern.nyu.edu/~adamodar/> → *Data → Current Data*:

- *Betas by Sector* (betas apalancados y desapalancados por industria).
- *Country Default Spreads and Risk Premiums* (prima de riesgo país).
- *Implied ERP* (prima de riesgo del mercado maduro).

## Limitaciones del modelo

- Los datos sectoriales y las primas de riesgo deben actualizarse periódicamente.
- El costo de la deuda se mantiene fijo en el análisis de escenarios (en la
  práctica podría aumentar con el apalancamiento).
- No incorpora una prima por tamaño.
- No incorpora una lambda específica de exposición al riesgo país.
- No estima un costo de deuda sintético.
- No sustituye un análisis financiero completo.

## Disclaimer

Esta herramienta tiene fines educativos y demostrativos. Los resultados se basan
en datos sectoriales y supuestos simplificados, por lo que no constituyen
asesoría financiera ni una estimación definitiva para una empresa específica. El
costo de capital puede variar según la moneda de valoración, estructura de
financiamiento, exposición geográfica, riesgo crediticio y características
particulares de cada compañía.
