#!/usr/bin/env python3
"""
preparar_notebook.py — Prepara un Jupyter Notebook existente para incluirlo
en el portafolio Quarto.

Pasos que realiza:
1. Copia el notebook al directorio proyectos/<nombre-slug>/index.ipynb
2. Inserta (o reemplaza) la celda Raw con YAML front matter como primera celda
3. Agrega opciones #| code-fold: true a las celdas de código que no las tengan
4. Crea un thumbnail.png placeholder si no existe
5. No modifica el notebook original

Uso:
    # Modo interactivo (recomendado)
    python scripts/preparar_notebook.py ~/notebooks/mi-analisis.ipynb

    # Modo con argumentos
    python scripts/preparar_notebook.py ~/notebooks/mi-analisis.ipynb \\
        --titulo "Mi análisis de datos" \\
        --descripcion "Descripción breve del proyecto." \\
        --categorias "Machine Learning" "Visualización" \\
        --destacado \\
        --github "https://github.com/usuario/repo"
"""

import argparse
import json
import os
import re
import shutil
import sys
from datetime import date
from pathlib import Path

try:
    import nbformat
    HAS_NBFORMAT = True
except ImportError:
    HAS_NBFORMAT = False


def slugify(text: str) -> str:
    """Convierte un texto en un slug apto para nombre de carpeta."""
    text = text.lower().strip()
    text = re.sub(r'[áàäâ]', 'a', text)
    text = re.sub(r'[éèëê]', 'e', text)
    text = re.sub(r'[íìïî]', 'i', text)
    text = re.sub(r'[óòöô]', 'o', text)
    text = re.sub(r'[úùüû]', 'u', text)
    text = re.sub(r'[ñ]', 'n', text)
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s-]+', '-', text)
    return text.strip('-')[:60]


def make_yaml_front_matter(titulo, descripcion, autor, fecha, categorias,
                            destacado, jupyter, github_url, demo_url):
    """Genera el contenido YAML front matter."""
    lines = [
        f'title: "{titulo}"',
        f'description: "{descripcion}"',
        f'author: "{autor}"',
        f'date: "{fecha}"',
    ]
    if categorias:
        cats = ', '.join(f'"{c}"' for c in categorias)
        lines.append(f'categories: [{cats}]')
    lines.append('image: thumbnail.png')
    lines.append(f'destacado: "{"true" if destacado else "false"}"')
    lines.append(f'jupyter: {jupyter}')
    if github_url:
        lines.append(f'github-url: "{github_url}"')
    if demo_url:
        lines.append(f'demo-url: "{demo_url}"')
    return '---\n' + '\n'.join(lines) + '\n---'


def load_notebook(path: Path) -> dict:
    """Carga el notebook desde disco."""
    if HAS_NBFORMAT:
        with open(path, 'r', encoding='utf-8') as f:
            return nbformat.read(f, as_version=4)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


def save_notebook(nb: dict, path: Path):
    """Guarda el notebook en disco."""
    if HAS_NBFORMAT:
        with open(path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
    else:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)


def get_cells(nb: dict) -> list:
    """Retorna la lista de celdas del notebook."""
    return nb['cells'] if isinstance(nb, dict) else nb.cells


def get_cell_type(cell) -> str:
    """Retorna el tipo de celda."""
    if isinstance(cell, dict):
        return cell.get('cell_type', '')
    return cell.cell_type


def get_cell_source(cell) -> str:
    """Retorna el source de la celda como string."""
    if isinstance(cell, dict):
        src = cell.get('source', '')
        if isinstance(src, list):
            return ''.join(src)
        return src
    return ''.join(cell.source) if isinstance(cell.source, list) else cell.source


def set_cell_source(cell, source: str):
    """Establece el source de la celda."""
    if isinstance(cell, dict):
        cell['source'] = source
    else:
        cell.source = source


def make_raw_cell(content: str) -> dict:
    """Crea una nueva celda Raw."""
    return {
        'cell_type': 'raw',
        'id': 'yaml-front-matter',
        'metadata': {},
        'source': content,
    }


def has_yaml_front_matter(cell) -> bool:
    """Verifica si una celda contiene YAML front matter de Quarto."""
    if get_cell_type(cell) != 'raw':
        return False
    source = get_cell_source(cell)
    return source.strip().startswith('---') and '---' in source.strip()[3:]


def add_code_fold_option(source: str) -> str:
    """Agrega la opción #| code-fold: true si no está presente."""
    lines = source.split('\n')
    # Buscar si ya tiene opciones #|
    has_code_fold = any('#| code-fold' in line for line in lines)
    if has_code_fold:
        return source

    # Buscar la última línea de opciones #| para insertar después
    last_option_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('#|'):
            last_option_idx = i

    if last_option_idx >= 0:
        lines.insert(last_option_idx + 1, '#| code-fold: true')
    else:
        lines.insert(0, '#| code-fold: true')

    return '\n'.join(lines)


def create_thumbnail_placeholder(dest_dir: Path):
    """Crea un thumbnail.png placeholder si no existe."""
    thumb_path = dest_dir / 'thumbnail.png'
    if thumb_path.exists():
        return

    # Intentar crear con Pillow
    try:
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (600, 340), color=(241, 245, 249))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 300), (600, 340)], fill=(250, 250, 250))
        draw.text((300, 170), 'Thumbnail del proyecto', fill=(148, 163, 184), anchor='mm')
        draw.text((300, 320), dest_dir.name, fill=(26, 26, 26), anchor='mm')
        img.save(str(thumb_path), 'PNG')
        print(f'  ✓ thumbnail.png creado con Pillow')
        return
    except ImportError:
        pass

    # Fallback: SVG como PNG (Quarto lo acepta)
    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 340" width="600" height="340">
  <rect width="600" height="340" fill="#f1f5f9"/>
  <text x="300" y="170" text-anchor="middle" dominant-baseline="middle" fill="#94a3b8"
        font-family="system-ui" font-size="18">Thumbnail: {dest_dir.name}</text>
</svg>'''
    svg_path = dest_dir / 'thumbnail.svg'
    svg_path.write_text(svg_content, encoding='utf-8')
    # Copy SVG content to PNG path as fallback
    shutil.copy(str(svg_path), str(thumb_path))
    print(f'  ✓ thumbnail.png (placeholder SVG) creado')


def ask(prompt: str, default: str = '') -> str:
    """Solicita input al usuario con un valor por defecto."""
    if default:
        result = input(f'{prompt} [{default}]: ').strip()
        return result if result else default
    return input(f'{prompt}: ').strip()


def interactive_mode(notebook_path: Path) -> dict:
    """Solicita los metadatos al usuario de forma interactiva."""
    print('\n' + '='*60)
    print('Preparar notebook para portafolio Quarto')
    print('='*60)
    print(f'Notebook: {notebook_path.name}\n')

    titulo = ask('Título del proyecto',
                 notebook_path.stem.replace('-', ' ').replace('_', ' ').title())
    descripcion = ask('Descripción breve (1-2 oraciones)')
    autor = ask('Autor', 'Tu Nombre')
    fecha = ask('Fecha (YYYY-MM-DD)', str(date.today()))
    categorias_str = ask('Categorías (separadas por coma)', 'Machine Learning')
    categorias = [c.strip() for c in categorias_str.split(',') if c.strip()]
    destacado_str = ask('¿Proyecto destacado? (s/n)', 'n')
    destacado = destacado_str.lower() in ('s', 'si', 'sí', 'y', 'yes')
    github_url = ask('URL de GitHub (opcional)', '')
    demo_url = ask('URL de demo (opcional)', '')
    slug = ask('Nombre de carpeta', slugify(titulo))

    return {
        'titulo': titulo,
        'descripcion': descripcion,
        'autor': autor,
        'fecha': fecha,
        'categorias': categorias,
        'destacado': destacado,
        'github_url': github_url,
        'demo_url': demo_url,
        'slug': slug,
    }


def prepare_notebook(
    notebook_path: Path,
    titulo: str = None,
    descripcion: str = None,
    autor: str = 'Tu Nombre',
    fecha: str = None,
    categorias: list = None,
    destacado: bool = False,
    github_url: str = '',
    demo_url: str = '',
    slug: str = None,
    jupyter: str = 'python3',
    base_dir: Path = None,
):
    """
    Prepara un notebook para el portafolio Quarto.

    Args:
        notebook_path: Ruta al notebook original.
        titulo: Título del proyecto.
        descripcion: Descripción corta.
        autor: Nombre del autor.
        fecha: Fecha en formato YYYY-MM-DD.
        categorias: Lista de categorías.
        destacado: Si debe aparecer en la página principal.
        github_url: URL del repositorio en GitHub.
        demo_url: URL de demo interactiva.
        slug: Nombre de la carpeta destino.
        jupyter: Nombre del kernel de Jupyter.
        base_dir: Directorio base del portafolio (donde está _quarto.yml).
    """
    if base_dir is None:
        # Detectar el directorio del portafolio
        base_dir = Path(__file__).parent.parent
        if not (base_dir / '_quarto.yml').exists():
            base_dir = Path.cwd()

    if not fecha:
        fecha = str(date.today())
    if not slug:
        slug = slugify(titulo or notebook_path.stem)
    if not categorias:
        categorias = []

    # Directorio destino
    dest_dir = base_dir / 'proyectos' / slug
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / 'index.ipynb'

    print(f'\n→ Copiando notebook a: {dest_path}')

    # Cargar notebook original
    nb = load_notebook(notebook_path)
    cells = get_cells(nb)

    # Generar YAML front matter
    yaml_content = make_yaml_front_matter(
        titulo=titulo or notebook_path.stem.replace('-', ' ').replace('_', ' ').title(),
        descripcion=descripcion or '',
        autor=autor,
        fecha=fecha,
        categorias=categorias,
        destacado=destacado,
        jupyter=jupyter,
        github_url=github_url,
        demo_url=demo_url,
    )

    # Insertar o reemplazar la primera celda con YAML
    if cells and has_yaml_front_matter(cells[0]):
        print('  ✓ Reemplazando celda Raw YAML existente')
        set_cell_source(cells[0], yaml_content)
    else:
        print('  ✓ Insertando celda Raw YAML como primera celda')
        new_cell = make_raw_cell(yaml_content)
        if isinstance(nb, dict):
            nb['cells'].insert(0, new_cell)
        else:
            nb.cells.insert(0, new_cell)
        cells = get_cells(nb)

    # Agregar code-fold a celdas de código
    code_cells_modified = 0
    for cell in cells:
        if get_cell_type(cell) == 'code':
            source = get_cell_source(cell)
            new_source = add_code_fold_option(source)
            if new_source != source:
                set_cell_source(cell, new_source)
                code_cells_modified += 1

    if code_cells_modified > 0:
        print(f'  ✓ Agregado #| code-fold: true a {code_cells_modified} celda(s) de código')

    # Guardar notebook en destino
    save_notebook(nb, dest_path)
    print(f'  ✓ Notebook guardado: {dest_path}')

    # Crear thumbnail placeholder
    create_thumbnail_placeholder(dest_dir)

    print(f'\n✅ Proyecto preparado en: proyectos/{slug}/')
    print(f'\nPróximos pasos:')
    print(f'  1. Reemplaza proyectos/{slug}/thumbnail.png con una imagen real (600×340px)')
    print(f'  2. Renderiza: quarto render')
    print(f'  3. Verifica: quarto preview')
    print(f'  4. Push: git add -A && git commit -m "feat: agregar proyecto {slug}" && git push')

    return dest_path


def main():
    parser = argparse.ArgumentParser(
        description='Prepara un Jupyter Notebook para el portafolio Quarto.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('notebook', type=Path, help='Ruta al notebook .ipynb')
    parser.add_argument('--titulo', '-t', help='Título del proyecto')
    parser.add_argument('--descripcion', '-d', help='Descripción breve')
    parser.add_argument('--autor', '-a', default='Tu Nombre', help='Nombre del autor')
    parser.add_argument('--fecha', '-f', help='Fecha (YYYY-MM-DD), por defecto hoy')
    parser.add_argument('--categorias', '-c', nargs='+', help='Categorías del proyecto')
    parser.add_argument('--destacado', action='store_true',
                        help='Marcar como proyecto destacado en la página principal')
    parser.add_argument('--github', '-g', default='', help='URL del repositorio en GitHub')
    parser.add_argument('--demo', default='', help='URL de demo interactiva')
    parser.add_argument('--slug', '-s', help='Nombre de la carpeta destino')
    parser.add_argument('--jupyter', default='python3', help='Kernel de Jupyter (default: python3)')
    parser.add_argument('--directorio', type=Path, help='Directorio base del portafolio')

    args = parser.parse_args()

    # Validar que el notebook existe
    if not args.notebook.exists():
        print(f'Error: no se encontró el archivo: {args.notebook}', file=sys.stderr)
        sys.exit(1)

    if not args.notebook.suffix == '.ipynb':
        print(f'Error: el archivo debe tener extensión .ipynb', file=sys.stderr)
        sys.exit(1)

    # Modo interactivo si no se proveen los metadatos mínimos
    if args.titulo is None:
        meta = interactive_mode(args.notebook)
        titulo = meta['titulo']
        descripcion = meta['descripcion']
        autor = meta['autor']
        fecha = meta['fecha']
        categorias = meta['categorias']
        destacado = meta['destacado']
        github_url = meta['github_url']
        demo_url = meta['demo_url']
        slug = meta['slug']
    else:
        titulo = args.titulo
        descripcion = args.descripcion or ''
        autor = args.autor
        fecha = args.fecha
        categorias = args.categorias or []
        destacado = args.destacado
        github_url = args.github
        demo_url = args.demo
        slug = args.slug

    prepare_notebook(
        notebook_path=args.notebook,
        titulo=titulo,
        descripcion=descripcion,
        autor=autor,
        fecha=fecha,
        categorias=categorias,
        destacado=destacado,
        github_url=github_url,
        demo_url=demo_url,
        slug=slug,
        jupyter=args.jupyter,
        base_dir=args.directorio,
    )


if __name__ == '__main__':
    main()
