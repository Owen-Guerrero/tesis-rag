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
