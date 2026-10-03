"""Crea el dataset sucio del día 2 a partir de los datos limpios del día 1.

Cada error se siembra de forma deliberada y queda anotado en `errores_sembrados.csv`,
que funciona como clave de respuestas: permite medir qué tan bien funcionó la limpieza.
Uso: python crear_dataset_sucio.py (desde la carpeta del post).
"""
from pathlib import Path

import pandas as pd

DATA = Path("data")
ref = pd.read_csv(DATA / "referencia_dia1.csv", parse_dates=["cierre_fiscal"])
ref = ref.sort_values(["empresa", "cierre_fiscal"]).reset_index(drop=True)

CUENTAS = ["ingresos", "utilidad_neta", "activos", "patrimonio",
           "activo_corriente", "pasivo_corriente", "pasivo_total"]
ENCABEZADOS = {
    "empresa": "Empresa", "cierre_fiscal": "Fecha de cierre", "ingresos": "Ingresos (MM USD)",
    "utilidad_neta": "Utilidad Neta", "activos": " Activos totales", "patrimonio": "Patrimonio",
    "activo_corriente": "Activo Corriente", "pasivo_corriente": "pasivo corriente",
    "pasivo_total": "Pasivo Total", "moneda": "Moneda", "margen_neto": "Margen neto",
}

# Variantes de nombre, como las escribiría cada persona que alimentó el archivo
NOMBRES = {
    "Costco": ["Costco Wholesale", "COSTCO WHOLESALE CORP /NEW", "Costco", "costco", "Costco Wholesale Corp."],
    "Target": ["Target Corporation", "TARGET CORP", "Target", "target corp.", "Target Corp"],
    "Walmart": ["Walmart Inc.", "WALMART", "Wal-Mart", "walmart ", "Walmart"],
}
FECHAS = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%b %d, %Y", "%Y-%m-%d 00:00:00"]
MONEDAS = ["USD", "usd", "US$", "Dólares", "USD"]

errores = []


def anotar(fila, columna, tipo, original, sucio):
    errores.append({"fila_origen": fila, "columna": columna, "tipo": tipo,
                    "valor_correcto": original, "valor_sucio": sucio})


def formatear(valor_mm, estilo):
    """Escribe un monto en millones con uno de cuatro estilos de captura."""
    v = round(valor_mm)
    if estilo == 0:
        return str(v)                                   # 713163
    if estilo == 1:
        return f"${v:,}"                                # $713,163
    if estilo == 2:
        return f"{v:,}".replace(",", ".")               # 713.163 (separador de miles español)
    return f"{v}.0"                                     # 713163.0


filas = []
for i, r in ref.iterrows():
    k = i % 5
    fila = {"empresa": NOMBRES[r["empresa"]][k],
            "cierre_fiscal": r["cierre_fiscal"].strftime(FECHAS[k]),
            "moneda": MONEDAS[k]}
    if fila["empresa"] != r["empresa"]:
        anotar(i, "empresa", "nombre inconsistente", r["empresa"], fila["empresa"])
    if k:
        anotar(i, "cierre_fiscal", "formato de fecha", r["cierre_fiscal"].strftime("%Y-%m-%d"), fila["cierre_fiscal"])
    if fila["moneda"] != "USD":
        anotar(i, "moneda", "moneda inconsistente", "USD", fila["moneda"])
    estilo = i % 4
    for c in CUENTAS:
        fila[c] = formatear(r[c] / 1e6, estilo)
        if estilo:
            anotar(i, c, "número como texto", round(r[c] / 1e6), fila[c])
    margen = r["margen_neto"]
    fila["margen_neto"] = [f"{margen * 100:.1f}%", f"{margen:.3f}", f"{margen * 100:.1f} %".replace(".", ","), ""][i % 4]
    filas.append(fila)

sucio = pd.DataFrame(filas)


def sembrar(fila, columna, nuevo, tipo):
    anotar(fila, columna, tipo, sucio.at[fila, columna], nuevo)
    sucio.at[fila, columna] = nuevo


# Errores puntuales, uno por tipo
sembrar(12, "ingresos", f"{ref.at[12, 'ingresos'] / 1e9:.1f}", "unidad equivocada (miles de millones)")
sembrar(3, "activos", str(round(ref.at[3, "activos"] / 1e6) * 10), "cero de más")
sembrar(11, "pasivo_total", "N/A", "valor faltante")
sembrar(7, "activo_corriente", "", "valor faltante")
sembrar(1, "pasivo_corriente", "-", "valor faltante")

# Duplicados: uno exacto y uno con otro nombre y otro formato de fecha
dup_exacto = sucio.iloc[[4]].copy()
dup_parcial = sucio.iloc[[9]].copy()
dup_parcial["empresa"] = "Target"
dup_parcial["cierre_fiscal"] = ref.at[9, "cierre_fiscal"].strftime("%d/%m/%Y")
anotar(4, "fila completa", "duplicado exacto", "", "")
anotar(9, "fila completa", "duplicado con otro formato", "", "")

sucio = pd.concat([sucio, dup_exacto, dup_parcial], ignore_index=True)
sucio = sucio.sample(frac=1, random_state=7).reset_index(drop=True)   # el orden de un archivo armado a mano
sucio = sucio[list(ENCABEZADOS)].rename(columns=ENCABEZADOS)

sucio.to_csv(DATA / "estados_sucio.csv", index=False)
pd.DataFrame(errores).to_csv(DATA / "errores_sembrados.csv", index=False)
print(f"{len(sucio)} filas sucias, {len(errores)} errores sembrados")
