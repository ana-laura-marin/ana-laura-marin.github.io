# 🗂️ Portafolio de Ciencia de Datos

Portafolio personal construido con [Quarto](https://quarto.org) y
[Jupyter Notebooks](https://jupyter.org/), desplegado automáticamente
en GitHub Pages.

## Demo

👉 [tu-usuario.github.io](https://tu-usuario.github.io)

## Cómo usar esta plantilla

### Prerrequisitos

- [Quarto CLI](https://quarto.org/docs/get-started/) (≥ 1.4)
- Python 3.9+ con `jupyter`, `nbformat`, `Pillow`
- Git + cuenta de GitHub

### Inicio rápido

1. Haz fork de este repositorio.
2. Renómbralo a `<tu-usuario>.github.io`.
3. Clona localmente:
   ```bash
   git clone https://github.com/<tu-usuario>/<tu-usuario>.github.io.git
   cd <tu-usuario>.github.io
   ```
4. Edita `_quarto.yml` con tu nombre, links y descripción.
5. Reemplaza `assets/img/perfil.jpg` con tu foto.
6. Agrega tus notebooks:
   ```bash
   python scripts/preparar_notebook.py ~/ruta/a/tu-notebook.ipynb
   ```
7. Renderiza y verifica:
   ```bash
   quarto render
   quarto preview
   ```
8. Push:
   ```bash
   git add -A && git commit -m "mi portafolio" && git push
   ```
9. Configura GitHub Pages (Settings → Pages → Source: GitHub Actions).

### Agregar un proyecto nuevo

```bash
python scripts/preparar_notebook.py mi-notebook.ipynb
quarto render
git add -A && git commit -m "agregar proyecto X" && git push
```

### Estructura del repositorio

```
.
├── _quarto.yml                  ← Configuración principal del sitio
├── index.qmd                    ← Página principal (bio + proyectos destacados)
├── sobre-mi.qmd                 ← Página "Sobre mí"
├── proyectos/
│   ├── index.qmd                ← Listing de todos los proyectos
│   ├── _metadata.yml            ← Config compartida para todos los notebooks
│   ├── churn-telecom/
│   │   ├── index.ipynb          ← Notebook del proyecto (con celda Raw YAML)
│   │   └── thumbnail.png        ← Imagen para el listing (600×340px)
│   ├── eda-inmobiliario/
│   │   ├── index.ipynb
│   │   └── thumbnail.png
│   └── sentimiento-nlp/
│       ├── index.ipynb
│       └── thumbnail.png
├── blog/
│   ├── index.qmd                ← Listing del blog
│   ├── _metadata.yml
│   └── mi-primer-post/
│       └── index.qmd
├── assets/
│   ├── img/
│   │   └── perfil.jpg           ← Reemplazar con tu foto
│   └── favicon.svg
├── styles/
│   └── custom.scss              ← Estilos personalizados
├── scripts/
│   └── preparar_notebook.py     ← Script para preparar notebooks
├── .github/
│   └── workflows/
│       └── publish.yml          ← GitHub Actions CI/CD
├── .gitignore
├── .nojekyll
└── README.md
```

### Personalización

- **Colores y fuentes:** editar `styles/custom.scss`
- **Navegación:** editar `website.navbar` en `_quarto.yml`
- **Foto y bio:** editar `index.qmd` y `sobre-mi.qmd`
- **Tema oscuro:** ya incluido (toggle automático en la barra de navegación)
- **URL del sitio:** editar `website.site-url` en `_quarto.yml`

### Cómo funciona el script `preparar_notebook.py`

El script copia un notebook existente al portafolio y:

1. **Inserta una celda Raw** con YAML front matter como primera celda (título, descripción, categorías, imagen).
2. **Agrega `#| code-fold: true`** a las celdas de código para que el lector vea primero los resultados y pueda expandir el código.
3. **Crea un thumbnail** placeholder si no existe.
4. **No modifica el notebook original**.

```bash
# Modo interactivo
python scripts/preparar_notebook.py ~/notebooks/mi-analisis.ipynb

# Modo con argumentos
python scripts/preparar_notebook.py ~/notebooks/mi-analisis.ipynb \
  --titulo "Mi análisis" \
  --descripcion "Descripción del proyecto." \
  --categorias "Machine Learning" "EDA" \
  --destacado \
  --github "https://github.com/usuario/repo"
```

### Troubleshooting

| Problema | Solución |
|----------|----------|
| `quarto render` falla en un `.ipynb` | Ejecutar todas las celdas en Jupyter antes de agregar al portafolio |
| Los gráficos no aparecen | El notebook no fue ejecutado. Abrir en Jupyter → Run All → guardar |
| GitHub Action falla con "Permission denied" | Settings → Actions → General → Workflow permissions → **Read and write** |
| Sitio muestra 404 | Settings → Pages → Source → **GitHub Actions** |
| Listing no muestra un proyecto | Verificar que la primera celda del `.ipynb` sea Raw con `---` |
| Thumbnail no aparece | `image: thumbnail.png` debe ser relativo a la carpeta del notebook |
| El notebook se re-ejecuta en CI | Verificar `execute: freeze: auto` en `_quarto.yml` y que `_freeze/` esté en Git |

## Licencia

MIT