"""
OE1 - Extracción de OWASP Top 10:2021 desde su fuente oficial en Markdown
--------------------------------------------------------------------------
Qué hace:
  1. Si no están en corpus/fuentes/owasp/, descarga los 10 archivos .md
     oficiales del repositorio OWASP/Top10 (carpeta 2021/docs/en).
  2. Genera UN registro por sección de cada categoría:
     Factors, Overview, Description, How to Prevent, Example Attack Scenarios,
     List of Mapped CWEs. (Se omite "References": son solo enlaces.)
  3. Limpia el Markdown (imágenes, enlaces, énfasis) y convierte la tabla
     "Factors" en líneas "campo: valor".
  4. Conserva los metadatos del esquema común (estándar, categoría, sección...).

Uso (desde C:\\tesis-rag con el .venv activo):
  python corpus\\extraer_owasp.py

Salidas:  corpus/procesado/owasp_top10_2021.jsonl
          corpus/procesado/owasp_manifest.json
"""

import json
import re
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
FUENTES = RAIZ / "corpus" / "fuentes" / "owasp"
SALIDA_DIR = RAIZ / "corpus" / "procesado"
SALIDA = SALIDA_DIR / "owasp_top10_2021.jsonl"
MANIFIESTO = SALIDA_DIR / "owasp_manifest.json"

URL_BASE = "https://raw.githubusercontent.com/OWASP/Top10/master/2021/docs/en/"
ARCHIVOS = [
    "A01_2021-Broken_Access_Control.md",
    "A02_2021-Cryptographic_Failures.md",
    "A03_2021-Injection.md",
    "A04_2021-Insecure_Design.md",
    "A05_2021-Security_Misconfiguration.md",
    "A06_2021-Vulnerable_and_Outdated_Components.md",
    "A07_2021-Identification_and_Authentication_Failures.md",
    "A08_2021-Software_and_Data_Integrity_Failures.md",
    "A09_2021-Security_Logging_and_Monitoring_Failures.md",
    "A10_2021-Server-Side_Request_Forgery_(SSRF).md",
]
SECCIONES_OMITIDAS = {"References"}


def descargar() -> None:
    FUENTES.mkdir(parents=True, exist_ok=True)
    for nombre in ARCHIVOS:
        destino = FUENTES / nombre
        if not destino.exists():
            url = URL_BASE + urllib.parse.quote(nombre)
            print(f"Descargando {nombre}")
            urllib.request.urlretrieve(url, destino)


def limpiar(md: str) -> str:
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)(\{[^}]*\})?", "", md)      # imágenes
    md = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", md)               # enlaces -> texto
    md = re.sub(r"(\*\*|\*|__|_)(\S[^*_]*?\S|\S)\1", r"\2", md)   # énfasis
    md = re.sub(r"^\s*[-*]\s+", "- ", md, flags=re.M)              # viñetas
    md = re.sub(r"[ \t]+", " ", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    lineas = md.split("\n")
    # Unir líneas cortadas dentro de un mismo párrafo
    salida = []
    for l in lineas:
        l = l.strip()
        if salida and l and salida[-1] and not l.startswith(("- ", "|")) and not salida[-1].startswith("|"):
            if not re.match(r"^(Scenario|\d+\.)", l):
                salida[-1] += " " + l
                continue
        salida.append(l)
    return "\n".join(salida).strip()


def tabla_a_texto(md: str) -> str:
    filas = [f for f in md.strip().split("\n") if f.strip().startswith("|")]
    if len(filas) < 3:
        return limpiar(md)
    cab = [c.strip() for c in filas[0].strip("|").split("|")]
    val = [c.strip() for c in filas[2].strip("|").split("|")]
    return "\n".join(f"{c}: {v}" for c, v in zip(cab, val))


def main() -> None:
    descargar()
    registros = []
    for nombre in ARCHIVOS:
        md = (FUENTES / nombre).read_text(encoding="utf-8")
        titulo_linea = md.split("\n", 1)[0]
        m = re.match(r"#\s*(A\d{2}:2021)\s*[–-]\s*(.+?)(\s{2,}!|\s*!\[|$)", titulo_linea)
        codigo, titulo = m.group(1), m.group(2).strip()
        partes = re.split(r"^##\s+(.+)$", md, flags=re.M)
        # partes = [encabezado, sec1, cuerpo1, sec2, cuerpo2, ...]
        for i in range(1, len(partes), 2):
            seccion, cuerpo = partes[i].strip(), partes[i + 1]
            if seccion in SECCIONES_OMITIDAS:
                continue
            texto = tabla_a_texto(cuerpo) if seccion == "Factors" else limpiar(cuerpo)
            if not texto:
                continue
            slug = re.sub(r"[^a-z0-9]+", "-", seccion.lower()).strip("-")
            registros.append({
                "id": f"owasp-{codigo[:3]}-{slug}",
                "estandar": "OWASP Top 10",
                "version": "2021",
                "id_control": codigo,
                "titulo": titulo,
                "familia_id": codigo[:3],
                "familia": titulo,
                "tipo": "seccion",
                "seccion": seccion,
                "parent_id": codigo,
                "nivel": 2,
                "relacionados": sorted(set(re.findall(r"CWE-\d+", cuerpo))) if seccion == "List of Mapped CWEs" else [],
                "idioma": "en",
                "texto": f"{codigo} – {titulo}\n{seccion}\n\n{texto}",
            })

    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    with SALIDA.open("w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    MANIFIESTO.write_text(json.dumps({
        "fuente": "github.com/OWASP/Top10, carpeta 2021/docs/en (rama master)",
        "fecha_extraccion": datetime.now().isoformat(timespec="seconds"),
        "categorias": len(ARCHIVOS),
        "registros": len(registros),
        "palabras_totales": sum(len(r["texto"].split()) for r in registros),
        "excluido": "secciones References (solo enlaces)",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Registros: {len(registros)} de {len(ARCHIVOS)} categorías -> {SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
