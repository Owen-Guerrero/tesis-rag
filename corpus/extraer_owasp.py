"""
OE1 - Extracción de OWASP Top 10:2025 desde su fuente oficial en Markdown
--------------------------------------------------------------------------
Qué hace:
  1. Si no están en corpus/fuentes/owasp/2025/, descarga los 10 archivos .md
     oficiales del repositorio de la OWASP Foundation (OWASP/Top10, carpeta
     2025/docs/en), que es la fuente con la que se publica owasp.org/Top10/2025.
  2. Genera UN registro por sección de cada categoría:
     Background, Score table, Description, How to prevent,
     Example attack scenarios, List of Mapped CWEs.
     (Se omite "References": son solo enlaces.)
  3. Limpia el Markdown (imágenes, enlaces, énfasis) y convierte la tabla de
     puntajes (HTML) en líneas "campo: valor".
  4. Conserva los metadatos del esquema común (estándar, categoría, sección...).

Uso (desde C:\\tesis-rag con el .venv activo):
  python corpus\\extraer_owasp.py

Salidas:  corpus/procesado/owasp_top10_2025.jsonl
          corpus/procesado/owasp_manifest.json
"""

import json
import re
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

VERSION = "2025"
RAIZ = Path(__file__).resolve().parent.parent
FUENTES = RAIZ / "corpus" / "fuentes" / "owasp" / VERSION
SALIDA_DIR = RAIZ / "corpus" / "procesado"
SALIDA = SALIDA_DIR / f"owasp_top10_{VERSION}.jsonl"
MANIFIESTO = SALIDA_DIR / "owasp_manifest.json"

URL_BASE = f"https://raw.githubusercontent.com/OWASP/Top10/master/{VERSION}/docs/en/"
ARCHIVOS = [
    "A01_2025-Broken_Access_Control.md",
    "A02_2025-Security_Misconfiguration.md",
    "A03_2025-Software_Supply_Chain_Failures.md",
    "A04_2025-Cryptographic_Failures.md",
    "A05_2025-Injection.md",
    "A06_2025-Insecure_Design.md",
    "A07_2025-Authentication_Failures.md",
    "A08_2025-Software_or_Data_Integrity_Failures.md",
    "A09_2025-Security_Logging_and_Alerting_Failures.md",
    "A10_2025-Mishandling_of_Exceptional_Conditions.md",
]
SECCIONES_OMITIDAS = {"References"}


def descargar() -> None:
    FUENTES.mkdir(parents=True, exist_ok=True)
    for nombre in ARCHIVOS:
        destino = FUENTES / nombre
        if not destino.exists():
            print(f"Descargando {nombre}")
            urllib.request.urlretrieve(URL_BASE + urllib.parse.quote(nombre), destino)


def limpiar(md: str) -> str:
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)(\{[^}]*\})?", "", md)      # imágenes
    md = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", md)               # enlaces -> texto
    md = re.sub(r"<[^>]+>", " ", md)                               # restos de HTML
    md = re.sub(r"(\*\*|\*|__|_)(\S[^*_]*?\S|\S)\1", r"\2", md)   # énfasis
    md = re.sub(r"^\s*[-*]\s+", "- ", md, flags=re.M)              # viñetas
    md = re.sub(r"^#{3,}\s*(.+)$", r"\1:", md, flags=re.M)         # subtítulos ###
    md = re.sub(r"[ \t]+", " ", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    # Unir líneas cortadas dentro de un mismo párrafo
    salida = []
    for l in md.split("\n"):
        l = l.strip()
        if (salida and l and salida[-1] and not l.startswith(("- ", "|"))
                and not salida[-1].endswith(":") and not re.match(r"^(Scenario|\d+\.)", l)):
            salida[-1] += " " + l
            continue
        salida.append(l)
    return "\n".join(salida).strip()


def tabla_a_texto(cuerpo: str) -> str:
    """Tabla de puntajes: 1.ª fila = nombres, 2.ª fila = valores (HTML o Markdown)."""
    filas = re.findall(r"<tr>(.*?)</tr>", cuerpo, flags=re.S)
    if len(filas) >= 2:
        celdas = [[re.sub(r"\s+", " ", c).strip() for c in re.findall(r"<td>(.*?)</td>", f, flags=re.S)]
                  for f in filas[:2]]
    else:
        md = [f for f in cuerpo.strip().split("\n") if f.strip().startswith("|")]
        if len(md) < 3:
            return limpiar(cuerpo)
        celdas = [[c.strip() for c in md[i].strip("|").split("|")] for i in (0, 2)]
    return "\n".join(f"{c}: {v}" for c, v in zip(*celdas))


def main() -> None:
    descargar()
    registros = []
    for nombre in ARCHIVOS:
        md = (FUENTES / nombre).read_text(encoding="utf-8")
        m = re.match(r"#\s*(A\d{2}:\d{4})\s*[–-]?\s*(.+?)\s*(!\[|$)", md.split("\n", 1)[0])
        codigo, titulo = m.group(1), m.group(2).strip()
        partes = re.split(r"^##\s+(.+)$", md, flags=re.M)   # [encabezado, sec1, cuerpo1, ...]
        for i in range(1, len(partes), 2):
            seccion = partes[i].strip().rstrip(".").strip()
            seccion = seccion[0].upper() + seccion[1:]
            cuerpo = partes[i + 1]
            if seccion in SECCIONES_OMITIDAS:
                continue
            texto = tabla_a_texto(cuerpo) if seccion.lower().startswith(("score table", "factors")) else limpiar(cuerpo)
            if not texto:
                continue
            slug = re.sub(r"[^a-z0-9]+", "-", seccion.lower()).strip("-")
            registros.append({
                "id": f"owasp-{codigo[:3]}-{slug}",
                "estandar": "OWASP Top 10",
                "version": VERSION,
                "id_control": codigo,
                "titulo": titulo,
                "familia_id": codigo[:3],
                "familia": titulo,
                "tipo": "seccion",
                "seccion": seccion,
                "parent_id": codigo,
                "nivel": 2,
                "relacionados": sorted(set(re.findall(r"CWE-\d+", cuerpo)), key=lambda c: int(c[4:]))
                                if seccion == "List of Mapped CWEs" else [],
                "idioma": "en",
                "texto": f"{codigo} – {titulo}\n{seccion}\n\n{texto}",
            })

    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    with SALIDA.open("w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    MANIFIESTO.write_text(json.dumps({
        "fuente": f"github.com/OWASP/Top10, carpeta {VERSION}/docs/en (rama master); "
                  f"publicado en owasp.org/Top10/{VERSION}/",
        "version": f"OWASP Top 10:{VERSION}",
        "fecha_extraccion": datetime.now().isoformat(timespec="seconds"),
        "categorias": len(ARCHIVOS),
        "registros": len(registros),
        "palabras_totales": sum(len(r["texto"].split()) for r in registros),
        "excluido": "secciones References (solo enlaces)",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Registros: {len(registros)} de {len(ARCHIVOS)} categorías -> {SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
