"""
Métricas de recuperación (deterministas: no usan ningún LLM)
------------------------------------------------------------
Cada elemento recuperado (una hoja en naive RAG; una hoja o un resumen en
RAPTOR) trae la lista de controles a los que pertenece. Un elemento es
RELEVANTE si alguno de sus controles está entre los controles esperados de
la pregunta. Para RAPTOR, un resumen hereda los controles de todas las hojas
que resume (regla que se documenta en la metodología).

- Recall@k: qué fracción de los controles esperados aparece en los k primeros elementos.
- MRR: 1 / posición del primer elemento relevante (0 si no hay ninguno).
- nDCG@k (Järvelin y Kekäläinen, 2002): premia que los relevantes estén arriba.
  Relevancia binaria; el ideal se calcula con todos los elementos relevantes
  que existen en el índice para esa pregunta.
"""

import math


def relevante(controles: list[str], esperados: set[str]) -> bool:
    return bool(set(controles) & esperados)


def recall_at_k(ranking: list[list[str]], esperados: set[str], k: int) -> float:
    encontrados = set()
    for controles in ranking[:k]:
        encontrados |= set(controles) & esperados
    return len(encontrados) / len(esperados)


def mrr(ranking: list[list[str]], esperados: set[str]) -> float:
    for i, controles in enumerate(ranking, start=1):
        if relevante(controles, esperados):
            return 1 / i
    return 0.0


def ndcg_at_k(ranking: list[list[str]], esperados: set[str], k: int, total_relevantes: int) -> float:
    dcg = sum(1 / math.log2(i + 1)
              for i, controles in enumerate(ranking[:k], start=1)
              if relevante(controles, esperados))
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, total_relevantes) + 1))
    return dcg / ideal if ideal else 0.0


# ---------------------------------------------------------------- citas en la respuesta
import re

FAMILIAS_NIST = ("AC|AT|AU|CA|CM|CP|IA|IR|MA|MP|PE|PL|PM|PS|PT|RA|SA|SC|SI|SR")
_CONECTOR = r"(?:[\s,:;(]|\bcontrol\b|\bcláusula\b|\bapartado\b|\bsección\b|\bsection\b|\bclause\b)*"
_PATRONES_CITA = [
    # NIST: el código de familia (AC-2, IA-5(1)) es único aunque no diga "NIST"
    ("NIST", re.compile(rf"\b((?:{FAMILIAS_NIST})-\d+)(?:\s*\(\d+\))?")),
    # ISO: el número solo no basta (5.17 existe en 27001 y 27002): debe nombrar la norma, antes o después
    ("ISO27002", re.compile(rf"\bISO(?:/IEC)?\s*27002(?::2022)?{_CONECTOR}(\d+\.\d+(?:\.\d+)?)")),
    ("ISO27002", re.compile(r"\b(?:control|cláusula|apartado)?\s*(\d+\.\d+)\s+(?:de\s+(?:la\s+)?(?:norma\s+)?|of\s+)ISO(?:/IEC)?\s*27002")),
    ("ISO27001", re.compile(rf"\bISO(?:/IEC)?\s*27001(?::2022)?{_CONECTOR}(A\.\d+\.\d+|\d+(?:\.\d+)*)")),
    ("ISO27001", re.compile(r"\b(A\.\d+\.\d+)\b")),   # "A.5.15" solo existe en el Anexo A de 27001
    # OWASP: "A01:2025" es único; "A01" suelto solo si dice OWASP
    ("OWASP", re.compile(r"\b(A\d{2}):2025\b")),
    ("OWASP", re.compile(r"\bOWASP(?:\s+Top\s+10(?::2025)?)?[\s,:–-]*(A\d{2})\b")),
]
# Guiones y espacios "raros" que algunos modelos escriben (p. ej. gpt-oss usa U+2011 en "AC‑2")
_NORMALIZAR = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
                             "\u2212": "-", "\u00a0": " ", "\u202f": " ", "\u2009": " "})


def extraer_citas(texto: str) -> set:
    """Controles citados en una respuesta, en formato de control base (NIST-AC-2, ISO27002-5.15...).

    Reconoce la cita con o sin corchetes y con o sin el nombre de la norma cuando el
    identificador no es ambiguo: "[NIST IA-2(1)]", "IA-2(1) exige...", "A.5.17" y
    "A07:2025" cuentan. Los números de ISO (5.17) solo cuentan si nombran la norma.
    La misma regla se aplica a todos los modelos y a las dos arquitecturas.
    """
    texto = (texto or "").translate(_NORMALIZAR)
    citas = set()
    for prefijo, patron in _PATRONES_CITA:
        for m in patron.finditer(texto):
            ident = m.group(1).rstrip(".")
            citas.add(f"{prefijo}-{ident}:2025" if prefijo == "OWASP" else f"{prefijo}-{ident}")
    return citas


def metricas_citas(respuesta: str, controles_contexto: set, esperados: set) -> dict:
    """- citas_en_contexto: fracción de lo citado que SÍ estaba en el contexto (citas no inventadas).
       - citas_esperadas: fracción de los controles esperados que la respuesta cita."""
    citadas = extraer_citas(respuesta)
    return {
        "citas": sorted(citadas),
        "citas_en_contexto": len(citadas & controles_contexto) / len(citadas) if citadas else None,
        "citas_esperadas": len(citadas & esperados) / len(esperados) if esperados else None,
        "sin_evidencia": "No se encontró evidencia en el contexto" in (respuesta or ""),
    }
