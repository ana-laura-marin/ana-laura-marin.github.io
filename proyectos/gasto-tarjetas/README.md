# Composición del gasto con tarjeta de crédito

Análisis de **en qué se gasta**, no de cuánto: segmentación de tarjetahabientes
por la composición de su canasta de consumo y estimación econométrica de los
determinantes de esa composición.

Datos: [Credit Card Transactions Dataset](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset)
(Kaggle, licencia Apache 2.0), ~1.3M transacciones de 2019–2020, 14 categorías
de comercio, demografía del tarjetahabiente y marca de fraude.

## La pregunta

Dos clientes que gastan lo mismo al mes pueden ser negocios completamente
distintos: uno concentra en supermercado y combustible, el otro en viajes y
restaurantes. La pregunta del proyecto es cómo se **recompone** la canasta (qué
mueve la participación de cada categoría) y si esa recomposición define grupos
estables de clientes.

El eje de lectura es la distinción entre **gasto hedónico** (entretenimiento,
restaurantes, viajes, compras, cuidado personal) y **utilitario** (supermercado,
combustible, hogar, salud, niños y mascotas), definida en `src/config.py`.

## Cómo correrlo

El pipeline resuelve la fuente de datos solo: usa el CSV de Kaggle si está
descargado, lo baja por API si hay credenciales, y si no genera un dataset
sintético con el mismo esquema. **Corre de entrada, sin configurar nada.**

```bash
python run_pipeline.py
```

> **En Windows con Anaconda**, si `python` no resuelve al intérprete correcto
> (suele quedar apuntando al stub de la Microsoft Store, y el launcher `py`
> puede apuntar a otra instalación sin las dependencias), invocá el de Anaconda
> por ruta completa o inicializá conda una vez con `conda init powershell` y
> reabrí la terminal.

Dependencias:

```bash
pip install -r requirements.txt
```

### Datos reales de Kaggle

Hay que generar el token a mano (Kaggle → Settings → Create New API Token) y
dejar `kaggle.json` en `%USERPROFILE%\.kaggle\`. Con eso el pipeline descarga y
descomprime el CSV solo en la siguiente corrida. Alternativa sin token: bajar el
ZIP desde la página del dataset y dejar `credit_card_transactions.csv` en
`data/raw/`.

Opciones:

| Flag | Efecto |
|---|---|
| `--sintetico` | Fuerza datos sintéticos aunque exista el CSV real |
| `--nrows 200000` | Submuestra el CSV para iterar rápido |
| `--k 4` | Fija el número de segmentos (por defecto lo elige la silueta) |
| `--min-txn 30` | Mínimo de transacciones para incluir a un cliente |

## Decisiones metodológicas

Son las tres cosas que separan esto de un `groupby` con gráficos.

**1. Las participaciones son datos composicionales.** Los shares por categoría
suman 1: viven en el símplex, no en R^14. La distancia euclidiana entre ellos no
tiene sentido geométrico y k-means aplicado directo produce grupos artefactuales.
El pipeline aplica una transformación **log-ratio centrada (CLR)** con
sustitución multiplicativa de ceros antes de agrupar (`features.clr`).

**2. La variable dependiente es una proporción.** Un MCO sobre un share predice
fuera de [0,1] y asume efectos constantes en todo el rango. Se usa **logit
fraccional** (Papke y Wooldridge, 1996): cuasi-máxima verosimilitud con link
logit, consistente aunque la distribución no sea binomial, con errores estándar
robustos HC1 siempre. Se reportan **efectos parciales promedio**, no coeficientes
crudos, porque son lo interpretable.

**3. El fraude se excluye del comportamiento.** Una transacción fraudulenta no es
una decisión de consumo del tarjetahabiente. Mezclarla contamina los perfiles.
`prep.limpiar` la separa en lugar de descartarla en silencio.

## Validación

El generador sintético construye 4 segmentos latentes conocidos con demografía
correlacionada. Corriendo `python run_pipeline.py --sintetico --k 4`, la
segmentación recupera esa estructura con **99.5% de pureza**: es la prueba de
que el CLR + k-means funciona, no solo de que corre.

También expone una limitación honesta: con `--k` libre, el criterio de silueta
elige **k=2**, no 4. La silueta premia particiones gruesas y colapsa el eje
secundario. El pipeline lo reporta explícitamente en vez de esconderlo.

El camino de densidad se audita igual y llega más lejos: `python -m src.densidad`
recupera los 4 grupos latentes con **ARI 0.991**, o sea 3 asignaciones erradas de
600. Ahí la trampa fue otra: la regla de dedo `min_samples = 2 × dimensiones` da
4 sobre un mapa 2D y parte los grupos reales en 20 pedazos. La regla está pensada
para el espacio original, donde la dimensión es alta; después de reducir a 2D el
criterio útil es de negocio, `min_samples` como el grupo más chico que uno acepta
llamar grupo. El barrido que lo muestra está en el notebook del paso 2.

## Salidas

`reports/figures/`

| Figura | Qué muestra |
|---|---|
| `01_composicion_categorias` | Peso de cada categoría en el gasto total, coloreado por tipo |
| `02_diagnostico_k` | Silueta e inercia por número de segmentos |
| `03_mapa_segmentos` | Segmentos en el espacio log-ratio (múltiplos pequeños) |
| `04_lift_segmentos` | Índice de especialización por segmento (100 = promedio) |
| `05_perfil_horario` | Distribución horaria del gasto hedónico vs utilitario |
| `06_efecto_edad` | Efecto parcial de la edad sobre cada categoría, con IC 95% |
| `07_umap_2d` | Embedding UMAP del panel, antes de agrupar |
| `08_k_distancias` | Curva de k-distancias con el codo y los `eps` candidatos |
| `09_dbscan` | Grupos hallados por DBSCAN, con el ruido aparte |
| `10_como_agrupa_dbscan` | Diagrama didáctico de núcleo, frontera y ruido |

`reports/tables/`: perfil de segmentos, índices de especialización, efectos
parciales (crudos y formateados) y contrastes conductuales, todo en CSV.

`data/processed/panel_clientes.parquet`: el panel a nivel de tarjetahabiente
con segmento asignado, listo para análisis posterior.

## Estructura

```
src/config.py          rutas, contrato de esquema, taxonomía hedónico/utilitario
src/data_io.py         descarga (API de Kaggle) y carga con validación de esquema
src/synthetic.py       generador con estructura latente conocida
src/prep.py            limpieza y features a nivel de transacción
src/features.py        panel de clientes, shares por categoría, transformación CLR
src/explorar.py        perfilado de la base: esquema, integridad, cobertura
src/segment.py         selección de k, k-means, perfilamiento
src/densidad.py        UMAP, curva de k-distancias, DBSCAN, análisis de atípicos
src/econometrics.py    logit fraccional, efectos parciales, contrastes conductuales
src/viz.py             figuras (paleta validada para daltonismo)
run_pipeline.py        orquestador del camino k-means
```

Hay **dos caminos de segmentación**, no uno. `run_pipeline.py` corre el de
k-means sobre las participaciones CLR. El de densidad vive en `densidad.py` y se
corre aparte:

```bash
python -m src.densidad
```

Reduce las 26 variables del panel a 2D con UMAP, calibra `eps` con la curva de
k-distancias y aplica DBSCAN. La diferencia práctica es que no hay que fijar el
número de grupos y los clientes que no encajan quedan marcados como atípicos en
vez de forzados dentro del grupo menos malo.

## Limitaciones

- El dataset de Kaggle es **sintético** (generado con Sparkov), no transacciones
  reales. Las magnitudes no son extrapolables a una cartera real; la
  contribución del proyecto es el método, no los parámetros estimados.
- La asignación hedónico/utilitario es una decisión del analista sobre 14
  categorías gruesas, no una clasificación validada. Está aislada en
  `config.py` justo para que sea fácil cuestionarla y cambiarla.
- Sin ingreso observado. `log_city_pop` y la ocupación son proxies pobres, así
  que los efectos estimados cargan sesgo de variable omitida: hay que leerlos
  como asociaciones condicionales, no como efectos causales.
- Corte transversal por cliente. No se explota la dimensión temporal del panel
  para efectos fijos individuales, que es la extensión natural.
