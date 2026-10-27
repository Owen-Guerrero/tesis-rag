"""
OE1 - Extracción de ISO/IEC 27001:2022 e ISO/IEC 27002:2022 desde PDF
---------------------------------------------------------------------
Qué hace:
  1. Lee los PDF de corpus/fuentes/iso/ (carpeta EXCLUIDA de GitHub: texto con licencia).
  2. Limpia cada página: quita encabezado, pie de página, número de página y marca de
     agua lateral ("licensed to ..."), caracteres invisibles y guiones de corte.
     Lee el texto por bloques ordenados por posición (así las listas a), b), c)...
     quedan en orden, cosa que falla con la extracción simple).
  3. ISO/IEC 27001:2022 genera:
       - un registro por subcláusula de las cláusulas 4 a 10 (requisitos del SGSI),
         con la ruta completa (p. ej. "6 Planning > 6.1 ... > 6.1.2 ...");
       - un registro por control del Anexo A (p. ej. A.5.15).
  4. ISO/IEC 27002:2022 genera:
       - un registro por término definido (3.1.x);
       - un registro por control (5.1 a 8.34) con sus atributos
         (tipo de control, propiedades, conceptos, capacidad operativa, dominio),
         enunciado (Control), propósito, guía (Guidance) e información adicional.
       - Cada control queda vinculado con su equivalente del Anexo A de 27001.
  5. Guarda un manifiesto con la huella SHA-256 de cada PDF (qué copia se usó).

Uso (desde C:\\tesis-rag con el .venv activo):
  pip install pymupdf
  python corpus\\extraer_iso.py

Entradas: corpus/fuentes/iso/ISO_IEC_27001_2022.pdf
          corpus/fuentes/iso/ISO_IEC_27002_2022.pdf
Salidas:  corpus/procesado/iso27001_2022.jsonl   (EXCLUIDO de GitHub)
          corpus/procesado/iso27002_2022.jsonl   (EXCLUIDO de GitHub)
          corpus/procesado/iso_manifest.json     (sí se sube: solo metadatos)
"""

import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pymupdf

RAIZ = Path(__file__).resolve().parent.parent
FUENTES = RAIZ / "corpus" / "fuentes" / "iso"
SALIDA_DIR = RAIZ / "corpus" / "procesado"
PDF_27001 = FUENTES / "ISO_IEC_27001_2022.pdf"
PDF_27002 = FUENTES / "ISO_IEC_27002_2022.pdf"

# Marca de agua de la copia usada. Si se cambia a otra copia con otra marca,
# basta con ajustar este patrón.
PATRON_MARCA_AGUA = re.compile(r"licensed to|SNV\s*/", re.I)

TEMAS_27002 = {"5": "Organizational controls", "6": "People controls",
               "7": "Physical controls", "8": "Technological controls"}

# Vocabulario oficial de atributos de ISO/IEC 27002:2022 (cláusula 4.2)
ATRIBUTOS = {
    "tipo_control": {"Preventive", "Detective", "Corrective"},
    "propiedades_seguridad": {"Confidentiality", "Integrity", "Availability"},
    "conceptos_ciberseguridad": {"Identify", "Protect", "Detect", "Respond", "Recover"},
    "capacidades_operativas": {
        "Governance", "Asset_management", "Information_protection", "Human_resource_security",
        "Physical_security", "System_and_network_security", "Application_security",
        "Secure_configuration", "Identity_and_access_management",
        "Threat_and_vulnerability_management", "Continuity", "Supplier_relationships_security",
        "Legal_and_compliance", "Information_security_event_management",
        "Information_security_assurance"},
    "dominios_seguridad": {"Governance_and_Ecosystem", "Protection", "Defence", "Resilience"},
}


# ---------------------------------------------------------------- utilidades
def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def normalizar(t: str) -> str:
    t = t.replace("\xad\n", "").replace("\xad", "")                 # guion suave de corte
    t = re.sub(r"[\u200b\ufeff\u00a0]", lambda m: " " if m.group() == "\u00a0" else "", t)
    t = t.replace("\t", " ")
    return t


def bloques(doc, desde: int, hasta: int):
    """Bloques de texto limpios, en orden de lectura: (página, x0, y0, texto).

    El orden se calcula con la LÍNEA BASE del primer renglón de cada bloque (no con el
    recuadro del bloque, que en estos PDF viene inflado y desordena títulos y párrafos).
    """
    for n in range(desde, hasta):
        pagina = doc[n]
        alto = pagina.rect.height
        lista = []
        for b in pagina.get_text("dict")["blocks"]:
            if b.get("type") != 0 or not b.get("lines"):
                continue
            x0, y0, x1, y1 = b["bbox"]
            lineas = ["".join(s["text"] for s in l["spans"]) for l in b["lines"]]
            texto = "\n".join(lineas)
            if y1 < 60 or y0 > alto - 60:                           # encabezado / pie
                continue
            if x1 < 32 or PATRON_MARCA_AGUA.search(texto):           # marca de agua lateral
                continue
            if texto.strip().startswith("Table A.1 (continued)"):
                continue
            texto = normalizar(texto).strip()
            if texto:
                primera = b["lines"][0]
                lx0 = primera["bbox"][0]
                base = primera["spans"][0]["origin"][1]
                lista.append((n, round(lx0), round(base), texto))
        lista.sort(key=lambda b: (b[2], b[1]))
        yield from lista


def agregar(parrafos: list, texto: str) -> None:
    """Agrega un párrafo; si empieza en minúscula es continuación del anterior."""
    texto = un_parrafo(texto)
    es_item = re.match(r"^(([a-z]|\d{1,2})\)|—|NOTE|Note \d)", texto)
    if parrafos and texto[:1].islower() and not es_item:
        parrafos[-1] = parrafos[-1] + " " + texto
    elif texto:
        parrafos.append(texto)


def un_parrafo(texto: str) -> str:
    """Une las líneas de un bloque; conserva el marcador de lista (a), 1), —)."""
    lineas = [l.strip() for l in texto.split("\n") if l.strip()]
    if len(lineas) > 1 and re.fullmatch(r"([a-z]|\d{1,2})\)|—|NOTE \d*|Note \d+ to entry:", lineas[0]):
        lineas = [lineas[0] + " " + lineas[1]] + lineas[2:]
    unido = " ".join(lineas)
    unido = re.sub(r"(\w)- (\w)", lambda m: m.group(1) + m.group(2)
                   if m.group(2).islower() else m.group(0), unido)  # palabra cortada con guion
    return re.sub(r"\s{2,}", " ", unido).strip()


def primera_pagina(doc, patron: str, desde: int = 0) -> int:
    for n in range(desde, len(doc)):
        for *_, texto, _b, _t in doc[n].get_text("blocks"):
            if re.match(patron, normalizar(texto).strip()) and "...." not in texto:
                return n
    raise ValueError(f"No se encontró: {patron}")


def registro(estandar, version, id_control, titulo, texto, **extra):
    base = {
        "id": f"ISO{re.sub(r'\D', '', estandar)}-{id_control}",   # p. ej. ISO27002-5.15
        "estandar": estandar,
        "version": version,
        "id_control": id_control,
        "titulo": titulo,
        "idioma": "en",
        "texto": texto,
    }
    base.update(extra)
    return base


# ---------------------------------------------------------------- ISO/IEC 27001
def extraer_27001() -> list:
    doc = pymupdf.open(PDF_27001)
    ini = primera_pagina(doc, r"^4\s+Context of the organization")
    anexo = primera_pagina(doc, r"^Annex\s+A", ini)
    biblio = primera_pagina(doc, r"^Bibliography", anexo)
    estandar, version = "ISO/IEC 27001", "2022 (3.ª edición, sin Amd 1:2024)"
    registros = []

    # --- Cláusulas 4 a 10 ---
    ruta = {}            # nivel -> (numero, titulo)
    actual = None
    patron_titulo = re.compile(r"^(\d{1,2}(?:\.\d){0,2})\s+([A-Z][^\n]{2,90})$")
    for n, x0, y0, texto in bloques(doc, ini, anexo + 1):
        if n == anexo and texto.startswith("Annex"):
            break
        plano = " ".join(texto.split())
        m = patron_titulo.match(plano)
        if m and "shall" not in plano and int(m.group(1).split(".")[0]) in range(4, 11):
            numero, titulo = m.group(1), m.group(2).strip()
            nivel = numero.count(".")
            ruta = {k: v for k, v in ruta.items() if k < nivel}
            ruta[nivel] = (numero, titulo)
            actual = {"numero": numero, "titulo": titulo, "nivel": nivel + 1,
                      "ruta": " > ".join(f"{a} {b}" for a, b in (ruta[k] for k in sorted(ruta))),
                      "parrafos": []}
            registros.append(actual)
            continue
        if actual is not None:
            agregar(actual["parrafos"], texto)

    salida = []
    for r in registros:
        if not r["parrafos"]:
            continue                      # títulos sin texto propio (p. ej. "6 Planning")
        padre = r["numero"].rsplit(".", 1)[0] if "." in r["numero"] else None
        salida.append(registro(
            estandar, version, r["numero"], r["titulo"],
            f"ISO/IEC 27001:2022 — {r['ruta']}\n\n" + "\n".join(r["parrafos"]),
            tipo="requisito", seccion=r["ruta"], parent_id=padre, nivel=r["nivel"],
            familia_id=f"Cl.{r['numero'].split('.')[0]}", familia=r["ruta"].split(" > ")[0],
            relacionados=[]))

    # --- Anexo A (tabla: col. izquierda = número y título; col. derecha = enunciado) ---
    controles, actual = [], None
    for n, x0, y0, texto in bloques(doc, anexo, biblio):
        lineas = [l.strip() for l in texto.split("\n") if l.strip()]
        if x0 < 215 and re.fullmatch(r"[5-8]\.\d{1,2}", lineas[0]):
            titulo = " ".join(l for l in lineas[1:] if l != "Control")
            actual = {"numero": lineas[0], "titulo": titulo, "texto": []}
            controles.append(actual)
        elif actual is not None and x0 >= 215:
            cuerpo = [l for l in lineas if l != "Control"]
            if cuerpo:
                actual["texto"].append(" ".join(cuerpo))
        elif actual is not None and x0 < 215 and not re.match(r"^[5-8]\s", lineas[0]):
            actual["titulo"] = (actual["titulo"] + " " + " ".join(
                l for l in lineas if l != "Control")).strip()   # título partido en 2 bloques
    for c in controles:
        tema = c["numero"].split(".")[0]
        c["titulo"] = re.sub(r"\s*[5-8]\s+(Organizational|People|Physical|Technological) controls\s*$",
                             "", c["titulo"]).strip()
        enunciado = re.sub(r"\s{2,}", " ", " ".join(c["texto"])).strip()
        salida.append(registro(
            estandar, version, f"A.{c['numero']}", c["titulo"],
            f"ISO/IEC 27001:2022 — Annex A — A.{c['numero']} {c['titulo']}\nControl:\n{enunciado}",
            tipo="control_anexo_a", seccion=f"Annex A — {TEMAS_27002[tema]}",
            parent_id=f"A.{tema}", nivel=2, familia_id=f"A.{tema}", familia=TEMAS_27002[tema],
            relacionados=[f"ISO27002-{c['numero']}"]))
    return salida


# ---------------------------------------------------------------- ISO/IEC 27002
def extraer_27002() -> list:
    doc = pymupdf.open(PDF_27002)
    ini_terminos = primera_pagina(doc, r"^3\.1\s+Terms and definitions")
    ini_controles = primera_pagina(doc, r"^5\s+Organizational controls", ini_terminos)
    anexo = primera_pagina(doc, r"^Annex\s+A", ini_controles)
    estandar, version = "ISO/IEC 27002", "2022 (3.ª edición)"
    salida = []

    # --- 3.1 Términos y definiciones ---
    terminos, actual = [], None
    for n, x0, y0, texto in bloques(doc, ini_terminos, ini_controles):
        if re.match(r"^3\.2\s", texto):
            break
        lineas = [l.strip() for l in texto.split("\n") if l.strip()]
        if re.fullmatch(r"3\.1\.\d{1,2}", lineas[0]) and len(lineas) >= 2:
            actual = {"numero": lineas[0], "termino": lineas[1],
                      "texto": [un_parrafo("\n".join(lineas[2:]))] if len(lineas) > 2 else []}
            terminos.append(actual)
        elif actual is not None:
            agregar(actual["texto"], texto)
    for t in terminos:
        salida.append(registro(
            estandar, version, t["numero"], t["termino"],
            f"ISO/IEC 27002:2022 — 3.1 Terms and definitions — {t['numero']} {t['termino']}\n"
            + "\n".join(p for p in t["texto"] if p),
            tipo="definicion", seccion="3.1 Terms and definitions", parent_id="3.1", nivel=3,
            familia_id="Cl.3", familia="Terms and definitions", relacionados=[]))

    # --- Controles 5.1 a 8.34 ---
    controles, actual, seccion = [], None, None
    patron_control = re.compile(r"^([5-8]\.\d{1,2})\s+(.+)$")
    for n, x0, y0, texto in bloques(doc, ini_controles, anexo + 1):
        if texto.startswith("Annex") and n == anexo:
            break
        plano = " ".join(texto.split())
        m = patron_control.match(plano)
        if m and len(plano) < 110 and not plano.endswith((".", ";")):
            actual = {"numero": m.group(1), "titulo": m.group(2), "hashtags": "",
                      "secciones": {}}
            controles.append(actual)
            seccion = "atributos"
            continue
        if actual is None or re.fullmatch(r"[5-8] [A-Z][a-z]+ controls", plano):
            continue
        if plano in ("Control", "Purpose", "Guidance", "Other information"):
            seccion = plano
            actual["secciones"].setdefault(seccion, [])
            continue
        if seccion == "atributos":
            if "#" in texto:
                actual["hashtags"] += " " + re.sub(r"(?<=[-_])\s*\n\s*", "", texto).replace(
                    "-", "").replace("\n", " ")
            continue
        agregar(actual["secciones"][seccion], texto)

    for c in controles:
        etiquetas = re.findall(r"#([A-Za-z_]+)", c["hashtags"])
        atributos = {k: [e for e in etiquetas if e in vocab] for k, vocab in ATRIBUTOS.items()}
        tema = c["numero"].split(".")[0]
        partes = [f"ISO/IEC 27002:2022 — {tema} {TEMAS_27002[tema]} — {c['numero']} {c['titulo']}",
                  "Attributes: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in atributos.items() if v)]
        for nombre in ("Control", "Purpose", "Guidance", "Other information"):
            if c["secciones"].get(nombre):
                partes.append(f"{nombre}:\n" + "\n".join(c["secciones"][nombre]))
        salida.append(registro(
            estandar, version, c["numero"], c["titulo"], "\n\n".join(partes),
            tipo="control", seccion=f"{tema} {TEMAS_27002[tema]}", parent_id=tema, nivel=2,
            familia_id=tema, familia=TEMAS_27002[tema],
            relacionados=[f"ISO27001-A.{c['numero']}"], atributos=atributos))
    return salida


# ---------------------------------------------------------------- principal
def guardar(nombre: str, registros: list) -> None:
    with (SALIDA_DIR / nombre).open("w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    r1, r2 = extraer_27001(), extraer_27002()
    guardar("iso27001_2022.jsonl", r1)
    guardar("iso27002_2022.jsonl", r2)

    c1, c2 = Counter(r["tipo"] for r in r1), Counter(r["tipo"] for r in r2)
    iam = sorted((r["id_control"] for r in r2 if "Identity_and_access_management"
                 in r.get("atributos", {}).get("capacidades_operativas", [])),
                 key=lambda c: [int(p) for p in c.split(".")])
    manifiesto = {
        "fecha_extraccion": datetime.now().isoformat(timespec="seconds"),
        "ISO/IEC 27001:2022": {"archivo": PDF_27001.name, "sha256": sha256(PDF_27001),
                               "registros": len(r1), "por_tipo": dict(c1),
                               "palabras": sum(len(r["texto"].split()) for r in r1)},
        "ISO/IEC 27002:2022": {"archivo": PDF_27002.name, "sha256": sha256(PDF_27002),
                               "registros": len(r2), "por_tipo": dict(c2),
                               "palabras": sum(len(r["texto"].split()) for r in r2),
                               "controles_identity_and_access_management": iam},
        "nota": "El texto de ISO no se versiona en GitHub (licencia). Solo este manifiesto.",
    }
    (SALIDA_DIR / "iso_manifest.json").write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"ISO/IEC 27001: {len(r1)} registros {dict(c1)}")
    print(f"ISO/IEC 27002: {len(r2)} registros {dict(c2)}")
    print(f"Controles 27002 con #Identity_and_access_management: {', '.join(iam)}")


if __name__ == "__main__":
    main()
