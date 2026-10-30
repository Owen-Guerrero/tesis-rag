"""
Métricas RAGAS de generación (Es et al., 2024), como en Flores et al. (2025)
-------------------------------------------------------------------------
Calcula, para la 1.ª corrida del piloto de cada arquitectura:
  - Faithfulness:       qué fracción de las afirmaciones de la respuesta está respaldada
                        por el contexto recuperado (necesita el TEXTO de lo recuperado).
  - Answer Relevancy:   qué tanto la respuesta atiende la pregunta.
  - Answer Correctness: corrección frente a la respuesta de referencia
                        (0,75 × corrección factual + 0,25 × similitud semántica).
  - Answer Similarity:  similitud semántica respuesta–referencia (embeddings).
Las métricas de recuperación se miden aparte con métricas clásicas (eval/evaluar.py),
igual que Flores et al. (2025), que aplican RAGAS solo a la generación.

JUEZ: un LLM local DISTINTO del que genera las respuestas (gpt-oss:20b), para evitar
que un modelo se califique a sí mismo. Por defecto qwen2.5:14b. Embeddings: el
mismo modelo del experimento (qwen3-embedding:0.6b).

Texto del contexto: las hojas salen de corpus/procesado/hojas.jsonl; los resúmenes
de RAPTOR, del árbol (corpus/procesado/arbol_raptor__*.jsonl, solo en la máquina
donde se construyó). Si no está el árbol, Faithfulness de RAPTOR queda sin calcular.

Instalación (una vez):
  pip install ragas==0.4.3 "langchain<1" "langchain-core<1" "langchain-community<0.4" "langchain-openai<1" "langchain-ollama<1"
  ollama pull qwen2.5:14b

Uso:
  python eval\\ragas_eval.py --limite 2         (prueba rápida con 2 preguntas)
  python eval\\ragas_eval.py                    (las 20 preguntas x 2 arquitecturas)
  python eval\\ragas_eval.py --juez qwen2.5:32b (en el lab)

Salidas: logs/evaluacion/ragas_por_pregunta.csv y logs/evaluacion/ragas.md
"""

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipelines.comun import cargar_hojas, config, leer_jsonl, modelo_embeddings, ruta, slug  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)
SALIDA = ruta("logs") / "evaluacion"
METRICAS = ["faithfulness", "answer_relevancy", "answer_correctness", "semantic_similarity"]


def primera_corrida(arq: str, maquina: str, ids: set) -> dict:
    llm = slug(config()["llm"]["modelo"])
    for archivo in sorted(ruta("resultados").glob(f"{arq}__*__{llm}__{maquina}__*.jsonl")):
        filas = {x["id"]: x for x in leer_jsonl(archivo) if "pregunta" in x}
        if ids <= set(filas):
            return filas
    raise SystemExit(f"No hay resultados del piloto para {arq}.")


def textos_de_nodos() -> dict:
    textos = {h["id_hoja"]: h["texto"] for h in cargar_hojas()}
    arbol = ruta("corpus/procesado") / f"arbol_raptor__{slug(modelo_embeddings())}__{slug(config()['llm']['modelo'])}.jsonl"
    if arbol.exists():
        for linea in arbol.read_text(encoding="utf-8").splitlines():
            n = json.loads(linea)
            if n["nivel"] > 0:
                textos[n["id"]] = n["texto"]
    else:
        print("AVISO: no está el árbol de RAPTOR en esta máquina -> Faithfulness de RAPTOR no se calcula.")
    return textos


def bootstrap(v, n=2000, semilla=42):
    """IC del 95 % por remuestreo percentil (Flores et al., 2025: 2000 iteraciones)."""
    v = [x for x in v if x is not None and not math.isnan(x)]
    if len(v) < 2:
        return None, None
    rng = random.Random(semilla)
    medias = sorted(statistics.mean(rng.choices(v, k=len(v))) for _ in range(n))
    return medias[int(0.025 * n)], medias[int(0.975 * n) - 1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--juez", default="qwen2.5:14b")
    p.add_argument("--maquina", default="lab_p5")
    p.add_argument("--limite", type=int, help="solo las primeras N preguntas (prueba)")
    a = p.parse_args()
    if a.juez == config()["llm"]["modelo"]:
        raise SystemExit("El juez debe ser distinto del LLM que generó las respuestas.")

    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerCorrectness, Faithfulness, ResponseRelevancy, SemanticSimilarity

    preguntas = {q["id"]: q for q in leer_jsonl(ruta("eval/preguntas_piloto.jsonl"))}
    ids = sorted(preguntas)[: a.limite] if a.limite else sorted(preguntas)
    textos = textos_de_nodos()
    juez = LangchainLLMWrapper(ChatOllama(model=a.juez, temperature=0, seed=42, num_ctx=8192))
    emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=modelo_embeddings()))
    run = RunConfig(max_workers=2, timeout=900, max_retries=3)

    filas = []
    for arq in ("naive", "raptor"):
        res = primera_corrida(arq, a.maquina, set(preguntas))
        muestras, con_contexto = [], True
        for i in ids:
            ctx = [textos.get(n["id"]) for n in res[i]["recuperados"]]
            if any(c is None for c in ctx):
                con_contexto = False
            muestras.append({"user_input": preguntas[i]["pregunta"],
                             "response": res[i].get("respuesta") or "",
                             "retrieved_contexts": [c for c in ctx if c],
                             "reference": preguntas[i]["respuesta_referencia"]})
        metricas = [ResponseRelevancy(), AnswerCorrectness(), SemanticSimilarity()]
        if con_contexto:
            metricas.insert(0, Faithfulness())
        print(f"\n{arq}: {len(ids)} preguntas | juez {a.juez} | métricas: {[m.name for m in metricas]}")
        t0 = time.perf_counter()
        r = evaluate(EvaluationDataset.from_list(muestras), metrics=metricas, llm=juez, embeddings=emb,
                     run_config=run, show_progress=True)
        print(f"  {time.perf_counter() - t0:.0f} s")
        df = r.to_pandas()
        for k, i in enumerate(ids):
            fila = {"arquitectura": arq, "id": i, "tipo": preguntas[i]["tipo"]}
            for m in METRICAS:
                v = df[m].iloc[k] if m in df.columns else None
                fila[m] = None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
            filas.append(fila)

    SALIDA.mkdir(parents=True, exist_ok=True)
    with (SALIDA / "ragas_por_pregunta.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    nombres = {"faithfulness": "Faithfulness", "answer_relevancy": "Answer Relevancy",
               "answer_correctness": "Answer Correctness", "semantic_similarity": "Answer Similarity"}
    md = [f"# Métricas RAGAS de generación (juez: {a.juez}; embeddings: {modelo_embeddings()})", "",
          f"Preguntas: {len(ids)} por arquitectura. Media [IC 95 % bootstrap, 2000 iteraciones] (n válidas).", "",
          "| Métrica | Naive RAG | RAPTOR |", "|---|---|---|"]
    for m in METRICAS:
        celdas = []
        for arq in ("naive", "raptor"):
            v = [x[m] for x in filas if x["arquitectura"] == arq and x[m] is not None]
            if not v:
                celdas.append("—")
                continue
            lo, hi = bootstrap(v)
            ic = f" [{lo:.3f}, {hi:.3f}]" if lo is not None else ""
            celdas.append(f"{statistics.mean(v):.3f}{ic} (n={len(v)})")
        md.append(f"| {nombres[m]} | {celdas[0]} | {celdas[1]} |")
    md += ["", "Las respuestas sin puntaje (n < total) corresponden a salidas del juez que RAGAS no pudo interpretar."]
    (SALIDA / "ragas.md").write_text("\n".join(md), encoding="utf-8")
    print("\n" + "\n".join(md))
    print(f"\nArchivos: logs/evaluacion/ragas_por_pregunta.csv y ragas.md")


if __name__ == "__main__":
    main()
