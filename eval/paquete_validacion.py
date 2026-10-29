"""
Paquete para validar las preguntas del piloto con ayuda de otra IA
-----------------------------------------------------------------
Arma un archivo de texto con, para cada pregunta: la pregunta, la respuesta de
referencia, los controles esperados y el TEXTO COMPLETO de esos controles
(sacado del corpus). Ese texto es lo que la IA revisora usa como única fuente.

Uso (desde la raíz del repo):
  python eval\\paquete_validacion.py P01 P10      ->  eval/paquete_validacion_P01-P10.txt
  python eval\\paquete_validacion.py P11 P20      ->  eval/paquete_validacion_P11-P20.txt

El archivo contiene texto ISO: NO se sube a GitHub (.gitignore).
"""

import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PROC = RAIZ / "corpus" / "procesado"
NIST_ID = re.compile(r"\b([A-Z]{2}-\d+(?:\(\d+\))?)")


def cargar(nombre):
    return [json.loads(l) for l in (PROC / nombre).read_text(encoding="utf-8").splitlines()]


def main() -> None:
    desde, hasta = (sys.argv[1:3] + ["P01", "P20"])[:2]
    preguntas = [json.loads(l) for l in (RAIZ / "eval/preguntas_piloto.jsonl").read_text(encoding="utf-8").splitlines()]
    preguntas = [q for q in preguntas if desde <= q["id"] <= hasta]
    nist = {r["id_control"]: r for r in cargar("nist_sp800-53.jsonl")}
    iso1 = {r["id_control"]: r for r in cargar("iso27001_2022.jsonl")}
    iso2 = {r["id_control"]: r for r in cargar("iso27002_2022.jsonl")}
    owasp = cargar("owasp_top10_2025.jsonl")

    partes = []
    for q in preguntas:
        textos, vistos = [], set()

        def agregar(etiqueta, texto):
            if etiqueta not in vistos:
                vistos.add(etiqueta)
                textos.append(f"<<< {etiqueta} >>>\n{texto.strip()}")

        # NIST: el control base y las mejoras que menciona "dónde está en la norma"
        bases = [c.split("-", 1)[1] for c in q["controles_esperados"] if c.startswith("NIST-")]
        for ident in bases + [m for m in NIST_ID.findall(q["ubicacion_en_norma"]) if m.split("(")[0] in bases]:
            if ident in nist:
                agregar(f"NIST SP 800-53 Rev. 5 {ident}", nist[ident]["texto"])
        for c in q["controles_esperados"]:
            if c.startswith("ISO27002-") and c[9:] in iso2:
                agregar(f"ISO/IEC 27002:2022 {c[9:]}", iso2[c[9:]]["texto"])
            elif c.startswith("ISO27001-") and c[9:] in iso1:
                agregar(f"ISO/IEC 27001:2022 {c[9:]}", iso1[c[9:]]["texto"])
            elif c.startswith("OWASP-"):
                cat = c[6:]
                for r in owasp:
                    if r["id_control"] == cat and r.get("seccion") in ("Description", "How to prevent"):
                        agregar(f"OWASP Top 10:2025 {cat} — {r['seccion']}", r["texto"])
        partes.append(
            f"################ {q['id']} ({q['tipo']}) ################\n"
            f"PREGUNTA: {q['pregunta']}\n"
            f"CONTROLES ESPERADOS: {', '.join(q['controles_esperados'])}\n"
            f"RESPUESTA DE REFERENCIA: {q['respuesta_referencia']}\n\n"
            f"TEXTO DE LA NORMA (única fuente válida):\n\n" + "\n\n".join(textos) + "\n"
        )
    salida = RAIZ / "eval" / f"paquete_validacion_{desde}-{hasta}.txt"
    salida.write_text("\n\n".join(partes), encoding="utf-8")
    print(f"{len(preguntas)} preguntas -> {salida.relative_to(RAIZ)} ({len(salida.read_text(encoding='utf-8')) // 1000} mil caracteres)")


if __name__ == "__main__":
    main()
