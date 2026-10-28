"""
OE1 -> OE5/OE6 - Fragmentación del corpus en "hojas"
------------------------------------------------------
Las hojas son la unidad mínima de recuperación y son LAS MISMAS para las dos
arquitecturas:
  - naive RAG indexa y recupera estas hojas;
  - RAPTOR las usa como nodos hoja de su árbol (y encima construye los resúmenes).
Así, naive RAG es una ablación exacta de RAPTOR (Laitenberger et al., 2025):
lo único que cambia entre ambas es la estrategia de recuperación.

Regla de fragmentación (Sarthi et al., 2024, y el código oficial de RAPTOR):
  - máximo 100 tokens por hoja, contados con el tokenizador cl100k_base (tiktoken);
  - no se cortan oraciones: si una oración no cabe, pasa entera a la siguiente hoja;
  - solo si una oración sola supera el límite, se divide en sus comas/punto y coma
    y, en último caso, por palabras;
  - una hoja nunca mezcla texto de dos registros (controles) distintos.
Decisión propia (documentar en el artículo): cada hoja empieza con un encabezado
corto "norma + id + título" para que no pierda su contexto. El encabezado cuenta
dentro de los 100 tokens.

Uso (desde C:\\tesis-rag con el .venv activo):
  pip install tiktoken
  python corpus\\fragmentar.py

Entradas: corpus/procesado/*.jsonl (NIST, OWASP, ISO 27001, ISO 27002)
Salidas:  corpus/procesado/hojas.jsonl                 (EXCLUIDO de GitHub: contiene texto ISO)
          corpus/procesado/fragmentacion_manifest.json (sí se sube)
"""

import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import tiktoken
import yaml

RAIZ = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))["fragmentacion"]
PROCESADO = RAIZ / "corpus" / "procesado"
FUENTES = [  # orden fijo = orden de las hojas en el corpus
    "nist_sp800-53.jsonl",
    "owasp_top10_2025.jsonl",
    "iso27001_2022.jsonl",
    "iso27002_2022.jsonl",
]
SALIDA = PROCESADO / "hojas.jsonl"
MANIFIESTO = PROCESADO / "fragmentacion_manifest.json"

MAX_TOKENS = CONFIG["max_tokens"]
ENC = tiktoken.get_encoding(CONFIG["tokenizador"])


def n_tokens(texto: str) -> int:
    return len(ENC.encode(texto))


# ---------------------------------------------------------------- unidades de texto
FIN_ORACION = re.compile(r"(?<=[.;:?!])\s+(?=[A-Z(\[\"“])")


MARCADOR = re.compile(r"^\(?([a-z]|\d{1,2}|[ivx]{1,4})[.)]$")   # "f." "1." "(a)" "b)"
ETIQUETA = re.compile(r"^[A-Z][A-Za-z ]{2,25}:$")               # "Control:" "Guidance:"


def unidades(cuerpo: str) -> list:
    """Divide el cuerpo en oraciones respetando las líneas (ítems de lista, párrafos).

    Un marcador de lista suelto ("f.", "1.") o una etiqueta suelta ("Guidance:")
    nunca queda solo: se une a la oración que le sigue.
    """
    crudas = []
    for linea in cuerpo.split("\n"):
        linea = linea.strip()
        if linea:
            crudas += [o.strip() for o in FIN_ORACION.split(linea) if o.strip()]
    salida, pendiente = [], ""
    for u in crudas:
        if MARCADOR.match(u) or ETIQUETA.match(u):
            pendiente = f"{pendiente} {u}".strip()
            continue
        salida.append(f"{pendiente} {u}".strip() if pendiente else u)
        pendiente = ""
    if pendiente:
        salida.append(pendiente)
    return salida


def partir_oracion_larga(oracion: str, limite: int) -> list:
    """Solo para oraciones que por sí solas superan el límite."""
    piezas, actual = [], ""
    for trozo in re.split(r"(?<=[,;:])\s+", oracion):
        candidato = f"{actual} {trozo}".strip()
        if n_tokens(candidato) <= limite:
            actual = candidato
            continue
        if actual:
            piezas.append(actual)
        if n_tokens(trozo) <= limite:
            actual = trozo
        else:                                   # último recurso: por palabras
            actual = ""
            for palabra in trozo.split():
                candidato = f"{actual} {palabra}".strip()
                if n_tokens(candidato) <= limite:
                    actual = candidato
                else:
                    piezas.append(actual)
                    actual = palabra
    if actual:
        piezas.append(actual)
    return piezas


# ---------------------------------------------------------------- encabezado y cuerpo
def encabezado_y_cuerpo(r: dict) -> tuple:
    anio = re.search(r"\d{4}", r["version"])
    norma = {
        "NIST SP 800-53": "NIST SP 800-53 Rev. 5",
        "OWASP Top 10": f"OWASP Top 10:{r['version']}",
    }.get(r["estandar"], f"{r['estandar']}:{anio.group() if anio else ''}")
    enc = f"[{norma} {r['id_control']} {r['titulo']}"
    if r["estandar"] == "OWASP Top 10":
        enc += f" — {r['seccion']}"
    elif r.get("tipo") == "definicion":
        enc += " (term definition)"
    enc += "]"
    # El primer renglón (OWASP: los dos primeros) del texto ya es el título: se reemplaza por el encabezado.
    saltar = 2 if r["estandar"] == "OWASP Top 10" else 1
    cuerpo = "\n".join(r["texto"].split("\n")[saltar:]).strip()
    return enc, cuerpo


def fragmentar_registro(r: dict) -> list:
    enc, cuerpo = encabezado_y_cuerpo(r)
    usar_enc = CONFIG.get("encabezado", True)
    limite = MAX_TOKENS - (n_tokens(enc) + 1 if usar_enc else 0)

    piezas = []
    for u in unidades(cuerpo):
        piezas += [u] if n_tokens(u) <= limite else partir_oracion_larga(u, limite)

    hojas, actual = [], ""
    for p in piezas:
        candidato = f"{actual} {p}".strip() if actual else p
        if n_tokens(candidato) <= limite:
            actual = candidato
        else:
            hojas.append(actual)
            actual = p
    if actual:
        hojas.append(actual)

    # Una etiqueta suelta al final (p. ej. "Guidance:") no es una hoja útil: se descarta.
    hojas = [h for h in hojas if not re.fullmatch(r"[A-Za-z ]{3,25}:", h)]
    return [(f"{enc}\n{h}" if usar_enc else h) for h in hojas]


# ---------------------------------------------------------------- principal
def main() -> None:
    hojas, orden = [], 0
    for archivo in FUENTES:
        ruta = PROCESADO / archivo
        if not ruta.exists():
            raise SystemExit(f"Falta {ruta.relative_to(RAIZ)}: corre antes su extractor.")
        for linea in ruta.read_text(encoding="utf-8").splitlines():
            r = json.loads(linea)
            for i, texto in enumerate(fragmentar_registro(r)):
                hojas.append({
                    "id_hoja": f"{r['id']}#h{i}",
                    "id_registro": r["id"],
                    "orden": orden,                 # posición en el corpus (orden original)
                    "estandar": r["estandar"],
                    "version": r["version"],
                    "id_control": r["id_control"],
                    "titulo": r["titulo"],
                    "familia_id": r.get("familia_id"),
                    "tipo": r.get("tipo"),
                    "parent_id": r.get("parent_id"),
                    "relacionados": r.get("relacionados", []),
                    "idioma": r.get("idioma", "en"),
                    "n_tokens": n_tokens(texto),
                    "texto": texto,
                })
                orden += 1

    with SALIDA.open("w", encoding="utf-8") as f:
        for h in hojas:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")

    por_norma = defaultdict(list)
    for h in hojas:
        por_norma[h["estandar"]].append(h["n_tokens"])
    excedidas = sum(h["n_tokens"] > MAX_TOKENS for h in hojas)
    manifiesto = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "parametros": CONFIG,
        "fuentes": FUENTES,
        "hojas_totales": len(hojas),
        "registros_totales": len({h["id_registro"] for h in hojas}),
        "hojas_que_exceden_el_limite": excedidas,
        "por_norma": {
            n: {"hojas": len(v), "tokens_total": sum(v),
                "tokens_media": round(statistics.mean(v), 1), "tokens_max": max(v)}
            for n, v in por_norma.items()
        },
        "sha256_hojas": hashlib.sha256(SALIDA.read_bytes()).hexdigest(),
    }
    MANIFIESTO.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Hojas: {len(hojas)} (de {manifiesto['registros_totales']} registros) | "
          f"exceden {MAX_TOKENS} tokens: {excedidas}")
    for n, d in manifiesto["por_norma"].items():
        print(f"  {n}: {d['hojas']} hojas | media {d['tokens_media']} tokens | máx {d['tokens_max']}")


if __name__ == "__main__":
    main()
