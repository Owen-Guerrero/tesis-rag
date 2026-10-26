"""
OE1 - Extracción de NIST SP 800-53 Rev. 5 desde el catálogo oficial OSCAL (JSON)
-------------------------------------------------------------------------------
Qué hace:
  1. Lee el catálogo OSCAL publicado por NIST (usnistgov/oscal-content).
  2. Genera UN registro por control y por mejora (enhancement) vigente.
     - Se omiten los controles retirados ("withdrawn").
     - Se omiten las partes de evaluación (SP 800-53A: assessment-objective,
       assessment-method), porque el corpus es SP 800-53, no SP 800-53A.
  3. Cada registro conserva los metadatos de trazabilidad del esquema común
     (estándar, versión, id del control, familia, padre, nivel, relacionados).
  4. Reemplaza los parámetros {{ insert: param, ... }} por el texto que usa
     la publicación en PDF: [Assignment: organization-defined ...] o
     [Selection (one or more): ...].
  5. Guarda un manifiesto con versión, fecha y conteos (para el artículo).

Uso (desde C:\\tesis-rag con el .venv activo):
  python corpus\\extraer_nist.py

Entrada:  corpus/fuentes/nist/NIST_SP-800-53_rev5_catalog.json
Salidas:  corpus/procesado/nist_sp800-53.jsonl
          corpus/procesado/nist_manifest.json
"""

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "corpus" / "fuentes" / "nist" / "NIST_SP-800-53_rev5_catalog.json"
SALIDA_DIR = RAIZ / "corpus" / "procesado"
SALIDA = SALIDA_DIR / "nist_sp800-53.jsonl"
MANIFIESTO = SALIDA_DIR / "nist_manifest.json"

PATRON_PARAM = re.compile(r"\{\{\s*insert:\s*param,\s*([\w.\-]+)\s*\}\}")


def prop(objeto: dict, nombre: str, clase=None):
    """Devuelve el valor de una propiedad OSCAL (p. ej. label sin clase = 'AC-2')."""
    for p in objeto.get("props", []):
        if p.get("name") == nombre and p.get("class") == clase:
            return p.get("value")
    return None


def texto_parametro(pid: str, params: dict, profundidad: int = 0) -> str:
    """Convierte un parámetro OSCAL a la redacción de la publicación oficial."""
    p = params.get(pid)
    if p is None or profundidad > 5:
        return "[Assignment: organization-defined value]"
    if "select" in p:
        sel = p["select"]
        opciones = [resolver(c, params, profundidad + 1) for c in sel.get("choice", [])]
        cuantas = "one or more" if sel.get("how-many") == "one-or-more" else "one"
        return f"[Selection ({cuantas}): " + "; ".join(opciones) + "]"
    etiqueta = p.get("label", "value")
    etiqueta = re.sub(r"^organization-defined\s+", "", etiqueta)
    return f"[Assignment: organization-defined {etiqueta}]"


def resolver(prosa: str, params: dict, profundidad: int = 0) -> str:
    return PATRON_PARAM.sub(lambda m: texto_parametro(m.group(1), params, profundidad), prosa)


def aplanar_partes(partes: list, params: dict, sangria: int = 0) -> list:
    """Recorre las partes 'item' manteniendo sus etiquetas (a., 1., ...) y la jerarquía."""
    lineas = []
    for parte in partes:
        etiqueta = prop(parte, "label") or ""
        prosa = resolver(parte.get("prose", ""), params)
        if etiqueta or prosa:
            lineas.append("  " * sangria + f"{etiqueta} {prosa}".strip())
        lineas += aplanar_partes(parte.get("parts", []), params, sangria + 1)
    return lineas


def contar_descendientes(ctl: dict) -> int:
    return sum(1 + contar_descendientes(h) for h in ctl.get("controls", []))


def construir_texto(ctl: dict, id_control: str, params: dict) -> str:
    bloques = [f"{id_control} {ctl['title']}"]
    for parte in ctl.get("parts", []):
        if parte["name"] == "statement":
            cuerpo = [resolver(parte.get("prose", ""), params)] if parte.get("prose") else []
            cuerpo += aplanar_partes(parte.get("parts", []), params)
            bloques.append("Control:\n" + "\n".join(l for l in cuerpo if l))
        elif parte["name"] == "guidance":
            bloques.append("Discussion:\n" + resolver(parte.get("prose", ""), params))
    return "\n\n".join(bloques)


def main() -> None:
    catalogo = json.loads(ENTRADA.read_text(encoding="utf-8"))["catalog"]
    meta = catalogo["metadata"]

    registros, retirados = [], 0
    por_familia = Counter()

    def recorrer(controles, familia_id, familia_titulo, padre, nivel):
        nonlocal retirados
        for ctl in controles:
            id_control = prop(ctl, "label")  # 'AC-2' o 'AC-2(1)'
            if prop(ctl, "status") == "withdrawn":
                retirados += 1 + contar_descendientes(ctl)
                continue
            params = {p["id"]: p for p in ctl.get("params", [])}
            relacionados = [
                l["href"].lstrip("#").upper()
                for l in ctl.get("links", [])
                if l.get("rel") == "related"
            ]
            registros.append({
                "id": f"nist-{id_control}",
                "estandar": "NIST SP 800-53",
                "version": f"Rev. 5 (OSCAL {meta.get('version')})",
                "id_control": id_control,
                "titulo": ctl["title"],
                "familia_id": familia_id.upper(),
                "familia": familia_titulo,
                "tipo": "control" if nivel == 1 else "mejora",
                "parent_id": padre,
                "nivel": nivel,
                "relacionados": relacionados,
                "idioma": "en",
                "texto": construir_texto(ctl, id_control, params),
            })
            por_familia[familia_id.upper()] += 1
            recorrer(ctl.get("controls", []), familia_id, familia_titulo, id_control, nivel + 1)

    for grupo in catalogo["groups"]:
        recorrer(grupo["controls"], grupo["id"], grupo["title"], None, 1)

    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    with SALIDA.open("w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_control = sum(r["tipo"] == "control" for r in registros)
    manifiesto = {
        "fuente": "usnistgov/oscal-content: nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json",
        "titulo": meta.get("title"),
        "version_oscal": meta.get("version"),
        "ultima_modificacion_nist": meta.get("last-modified"),
        "fecha_extraccion": datetime.now().isoformat(timespec="seconds"),
        "registros": len(registros),
        "controles": n_control,
        "mejoras": len(registros) - n_control,
        "retirados_omitidos": retirados,
        "palabras_totales": sum(len(r["texto"].split()) for r in registros),
        "registros_por_familia": dict(sorted(por_familia.items())),
        "excluido": "partes de evaluación SP 800-53A (assessment-objective, assessment-method)",
    }
    MANIFIESTO.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Registros: {len(registros)} ({n_control} controles, {len(registros) - n_control} mejoras)")
    print(f"Retirados omitidos: {retirados}")
    print(f"AC: {por_familia['AC']} | IA: {por_familia['IA']}")
    print(f"Guardado en {SALIDA.relative_to(RAIZ)} y {MANIFIESTO.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
