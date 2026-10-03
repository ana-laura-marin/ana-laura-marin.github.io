# Dashboard · IA aplicada a finanzas

Aplicación de Streamlit del día 5 del reto *IA aplicada a finanzas*. Muestra los
indicadores de rentabilidad de Walmart, Target y Costco (datos de la SEC) y la
morosidad de los bancos de EE. UU. (datos de la Reserva Federal vía FRED).

## Ejecutar en local

Desde la raíz del repositorio:

```bash
python -m venv .venv
.venv/Scripts/activate        # en macOS o Linux: source .venv/bin/activate
pip install -r blog/ia-finanzas-5-dashboard/app/requirements.txt
streamlit run blog/ia-finanzas-5-dashboard/app/app.py
```

La aplicación se abre en `http://localhost:8501`.

## Archivos

- `app.py`: la aplicación.
- `data/`: los CSV de los días 1 y 3 del reto.
- `requirements.txt`: dependencias.
- `AGENTS.md`: instrucciones para asistentes de código (Cursor y otros las leen).
