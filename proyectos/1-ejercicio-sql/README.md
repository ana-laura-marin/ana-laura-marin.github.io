# Análisis de performance comercial · Olist Marketplace (2016–2018)

Análisis en **SQL** (SQLite) del dataset público de e-commerce brasileño **Olist**: cerca de
99.000 pedidos, 3.100 vendedores y 98.000 reseñas entre septiembre de 2016 y octubre de 2018.

El reporte cubre cinco ejes y cierra con una síntesis de hallazgos y recomendaciones:

1. **Caracterización del período** — volumen, revenue y estacionalidad.
2. **Red de vendedores** — concentración por vendedor y distribución geográfica.
3. **Satisfacción del cliente** — distribución de reseñas e impacto del retraso en la entrega.
4. **Eficiencia logística** — *On-Time Delivery Rate* y tiempos de entrega.
5. **Categorías y métodos de pago** — revenue, ratio flete/precio y uso de cuotas.

## Hallazgos clave

- Revenue bruto de **R$ 20,3 M** con un pico en el **Black Friday de noviembre de 2017**.
- **93,2 %** de entregas a tiempo; los retrasos hunden la satisfacción de **4,30 a 2,27 estrellas**,
  y el 54 % de los pedidos retrasados termina en una reseña de 1 estrella.
- La red está muy concentrada: el **10 % de los vendedores genera el 66 % del revenue**, y
  **São Paulo aporta el 64,6 %** (fuerte dependencia geográfica).
- El volumen de ventas **no predice** la satisfacción (ρ de Spearman = −0,13).
- El flete pesa hasta un **68 % del precio** en las categorías de bajo ticket (electrónica).

## Nota sobre el cálculo del OTD

`fecha_estimada_entrega` está registrada a las 00:00:00 del día prometido, mientras que
`fecha_entrega_cliente` lleva la hora real. Comparar ambos campos como instantes clasifica
como retraso los pedidos entregados *durante* el día comprometido. El reporte compara por día
calendario con `date()`, lo que recupera 1.292 entregas cumplidas y sube el OTD de 91,9 % a
93,2 %. La sección 3.3 del notebook lo documenta.

## Estructura

- `index.ipynb` — reporte completo (consultas SQL, tablas y visualizaciones).
- `data/portafolio_olist.db` — base de datos SQLite utilizada en el análisis.

El notebook se publica con sus salidas embebidas: Quarto no re-ejecuta los `.ipynb`, así que
tras editarlo hay que ejecutarlo antes de renderizar.

## Datos

Olist Brazilian E-Commerce Public Dataset — [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
