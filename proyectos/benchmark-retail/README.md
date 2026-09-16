# Dónde está la oportunidad · Benchmark competitivo por familia de producto

Análisis financiero del portafolio de producto de **Corporación Favorita** (54 tiendas en
Ecuador, 33 familias, 2013–2017) contra un **competidor de referencia simulado**, para
cuantificar en dinero dónde hay ingreso no capturado y cuál oportunidad vale la pena atacar
primero.

## La pregunta

Una cadena sabe cuánto vende. Lo que no sabe mirando solo sus propios números es **cuánto
podría estar vendiendo**: una familia que crece 4 % al año parece sana hasta que se descubre
que la plaza creció 11 %. El agregado propio no tiene vara de medición; el benchmark sí.

El proyecto construye esa vara y responde tres preguntas:

1. **¿Dónde está la brecha?** Participación por familia y por plaza frente al competidor.
2. **¿Cuánto vale?** Traducción de la brecha a ingreso anualizado, con supuestos declarados.
3. **¿Cuál atacar?** Ranking que pondera tamaño, persistencia y respuesta a promoción.

## El competidor es ficticio (y por qué eso no invalida el ejercicio)

No existe un dataset público con las ventas de los competidores de Favorita. El benchmark se
**simula desde la estructura real** de los datos con un generador reproducible (semilla fija):
conserva el calendario, la estacionalidad y la exposición al ciclo del petróleo —Ecuador está
dolarizado y su consumo sigue al crudo—, y se aparta en supuestos explícitos y documentados:
otro mix de familias, otra política promocional y una presencia desigual por ciudad.

Ningún número sobre el competidor describe a una empresa real. Lo que el proyecto demuestra
es el **método**: cómo se construye una vara de medición defendible, cómo se aísla la brecha
estructural del ruido, y cuánto de la conclusión depende de los supuestos. La última sección
es explícita sobre qué resultado sobrevive si se cambia el generador y cuál no.

Los datos de Favorita **sí son reales**: `sales` mide volumen de ventas por familia, tienda y
día. Como el dataset no trae precios ni márgenes, la conversión a dinero usa un precio
promedio por familia declarado como supuesto, y se acompaña de un análisis de sensibilidad.

## Estructura

- `index.qmd` — reporte narrativo del portafolio.
- `notebooks/` — cuadernos de análisis cuyas figuras se embeben en el reporte.
- `src/` — módulos: carga, generador del benchmark, métricas de brecha y visualización.
- `run_pipeline.py` — reproduce el análisis completo de punta a punta.
- `preparar_datos.py` — extrae y valida el ZIP de Kaggle en `data/raw/`.

## Datos

Store Sales — Time Series Forecasting, Corporación Favorita ·
[Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).
Los CSV no se versionan: se descargan a `data/raw/` y el resto se regenera con el pipeline.
