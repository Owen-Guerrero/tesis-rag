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

_PATRONES_CITA = [
    ("NIST", re.compile(r"\bNIST(?:\s+SP\s+800-53)?(?:\s+Rev\.\s*5)?\s+([A-Z]{2}-\d+)(?:\(\d+\))?")),
    ("ISO27002", re.compile(r"\bISO(?:/IEC)?\s*27002(?::2022)?\s+(\d+\.\d+(?:\.\d+)?)")),
    ("ISO27001", re.compile(r"\bISO(?:/IEC)?\s*27001(?::2022)?\s+(A\.\d+\.\d+|\d+(?:\.\d+)*)")),
    ("OWASP", re.compile(r"\bOWASP(?:\s+Top\s+10(?::2025)?)?\s+(A\d{2})(?::2025)?")),
]


def extraer_citas(texto: str) -> set:
    """Controles citados en una respuesta, en formato de control base (NIST-AC-2, ISO27002-5.15...).
    Reconoce la cita con o sin corchetes: "[NIST IA-2(1)]" y "NIST IA-2(1) exige..." cuentan igual."""
    citas = set()
    for prefijo, patron in _PATRONES_CITA:
        for m in patron.finditer(texto or ""):
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
