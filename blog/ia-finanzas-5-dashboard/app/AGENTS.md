# Instrucciones para asistentes de código

Este proyecto es un dashboard de Streamlit con indicadores financieros públicos.

## Datos

- `data/indicadores.csv`: indicadores anuales de Walmart, Target y Costco, calculados a
  partir de sus 10-K (API de EDGAR de la SEC). Las proporciones están en tanto por uno
  (0,031 = 3,1 %).
- `data/morosidad_fred.csv`: tasas trimestrales de morosidad de la Reserva Federal
  (FRED), en porcentaje. La columna `fecha_descarga` indica la versión de los datos.
- Los datos son de solo lectura. Nunca los modifiques ni inventes valores.

## Reglas

- Python 3.10 o superior, pandas, plotly y la versión actual de Streamlit. Revisa la
  documentación vigente de Streamlit antes de usar un parámetro: algunos cambiaron
  (por ejemplo, `use_container_width` fue reemplazado por `width`).
- Las rutas a los datos se construyen con `Path(__file__).parent`, porque en Streamlit
  Community Cloud el directorio de trabajo es la raíz del repositorio.
- Los textos de la interfaz van en español. Los números con coma decimal.
- Cada gráfico lleva su fuente en un `st.caption`.
- La app no llama a modelos de IA ni usa claves de API.
- Antes de dar por terminado un cambio, ejecuta `streamlit run` y confirma que no hay
  errores ni advertencias en la consola.
