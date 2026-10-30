"""
Evaluación del piloto: naive RAG vs RAPTOR (las dos familias de métricas)
-------------------------------------------------------------------------
Lee los resultados del piloto (resultados/naive__... y resultados/raptor__...,
todas las repeticiones) y calcula, SIN volver a llamar a ningún LLM:

1) Recuperación (deterministas):
   - Recall@5, Recall@10, MRR y nDCG@10 sobre el ranking de 100 candidatos.
   - Recall del contexto: qué fracción de los controles esperados llegó de
     verdad al LLM (dentro del presupuesto de 2000 tokens).
2) Calidad de la respuesta (deterministas):
   - Citas en el contexto (no inventó citas) y controles esperados citados.
   - ROUGE-L (F1) frente a la respuesta de referencia.
   - Similitud semántica respuesta–referencia (coseno con el modelo de embeddings).
   - BERTScore (F1), opcional con --bertscore (requiere: pip install bert-score).
3) Desempeño del equipo: latencia total, tiempo de generación, primer token,
   tokens/s, tokens de entrada, VRAM; media ± desviación entre repeticiones y
   % de respuestas idénticas entre repeticiones (reproducibilidad).

REGLAS DE RELEVANCIA (documentar en la metodología):
  - ISO/IEC 27001 A.x y ISO/IEC 27002 x son el MISMO control (enunciado y guía):
    se cuentan como uno solo.
  - Hoja: contiene el control al que pertenece.
  - Resumen de RAPTOR (regla principal, "mención explícita"): contiene un control
    solo si su TEXTO lo menciona por su identificador y ese control está entre
    las hojas que resume. Así un resumen no recibe crédito por los cientos de
    controles que tiene debajo sin mencionarlos.
  - Regla "herencia" (cota superior, solo como referencia): el resumen contiene
    todos los controles de sus hojas.

Uso (desde la raíz del repo):
  python eval\\evaluar.py --menciones   (SOLO en el lab: exporta qué controles menciona
                                        cada resumen a logs/, sin texto; luego commit)
  python eval\\evaluar.py               (en cualquier máquina, con ese archivo de logs/)
  python eval\\evaluar.py --bertscore   (agrega BERTScore)

Salidas: logs/evaluacion/por_pregunta.csv, logs/evaluacion/resumen.csv y
         logs/evaluacion/tablas.md (tablas listas para el documento).
"""

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.metricas import extraer_citas  # noqa: E402
from pipelines.comun import (cargar_hojas, config, control_base, embeber, leer_jsonl,  # noqa: E402
                             modelo_embeddings, ruta, slug)

K = (5, 10)
SALIDA = ruta("logs") / "evaluacion"


# ================================================================ utilidades
def norm(c: str) -> str:
    """ISO 27001 A.x == ISO 27002 x."""
    return "ISO27002-" + c[len("ISO27001-A."):] if c.startswith("ISO27001-A.") else c


def normset(cs) -> set:
    return {norm(c) for c in cs}


def media(v):
    v = [x for x in v if x is not None]
    return statistics.mean(v) if v else None


def desv(v):
    v = [x for x in v if x is not None]
    return statistics.stdev(v) if len(v) > 1 else 0.0


def f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def sufijo_arbol() -> str:
    return f"{slug(modelo_embeddings())}__{slug(config()['llm']['modelo'])}"


# ================================================================ menciones de los resúmenes
def archivo_menciones() -> Path:
    return ruta("logs") / f"arbol_menciones__{sufijo_arbol()}.jsonl"


def exportar_menciones() -> None:
    arbol = ruta("corpus/procesado") / f"arbol_raptor__{sufijo_arbol()}.jsonl"
    if not arbol.exists():
        raise SystemExit(f"No está {arbol.relative_to(ruta('.'))}: --menciones se corre en la máquina del árbol.")
    filas = []
    for linea in arbol.read_text(encoding="utf-8").splitlines():
        n = json.loads(linea)
        if n["nivel"] == 0:
            continue
        debajo = normset(n["conteo"].keys())
        mencionados = sorted(normset(extraer_citas(n["texto"])) & debajo)
        filas.append({"id": n["id"], "nivel": n["nivel"], "n_tokens": n["n_tokens"],
                      "controles_mencionados": mencionados, "controles_debajo": sorted(debajo)})
    with archivo_menciones().open("w", encoding="utf-8") as fh:
        for x in filas:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    prom = statistics.mean(len(x["controles_mencionados"]) for x in filas)
    prom_d = statistics.mean(len(x["controles_debajo"]) for x in filas)
    print(f"{len(filas)} resúmenes: mencionan en promedio {prom:.1f} controles; tienen debajo {prom_d:.1f}.")
    print(f"Guardado: {archivo_menciones().relative_to(ruta('.'))} (sin texto, sí se sube)")


def cargar_controles_nodos() -> tuple:
    """id de nodo -> controles, con las dos reglas (mención explícita y herencia)."""
    estricta, herencia = {}, {}
    for h in cargar_hojas():
        c = {norm(control_base(h))}
        estricta[h["id_hoja"]] = herencia[h["id_hoja"]] = c
    if archivo_menciones().exists():
        for x in leer_jsonl(archivo_menciones()):
            estricta[x["id"]] = set(x["controles_mencionados"])
            herencia[x["id"]] = set(x["controles_debajo"])
    return estricta, herencia


# ================================================================ métricas de recuperación
def recall_k(ranking, nodos, esperados, k):
    encontrados = set().union(*[nodos.get(i, set()) for i in ranking[:k]]) if ranking else set()
    return len(encontrados & esperados) / len(esperados)


def mrr(ranking, nodos, esperados):
    for pos, i in enumerate(ranking, start=1):
        if nodos.get(i, set()) & esperados:
            return 1 / pos
    return 0.0


def ndcg_k(ranking, nodos, esperados, k):
    rel = [1 if nodos.get(i, set()) & esperados else 0 for i in ranking]
    dcg = sum(r / math.log2(p + 1) for p, r in enumerate(rel[:k], start=1))
    ideal = sum(1 / math.log2(p + 1) for p in range(1, min(k, sum(rel)) + 1))   # ideal sobre el pool de 100
    return dcg / ideal if ideal else 0.0


def precision_k(ranking, nodos, esperados, k):
    top = ranking[:k]
    return sum(1 for i in top if nodos.get(i, set()) & esperados) / len(top) if top else 0.0


def average_precision(ranking, nodos, esperados):
    """Average Precision sobre el pool de 100 candidatos (MAP = media entre preguntas)."""
    aciertos, suma = 0, 0.0
    for pos, i in enumerate(ranking, start=1):
        if nodos.get(i, set()) & esperados:
            aciertos += 1
            suma += aciertos / pos
    return suma / aciertos if aciertos else 0.0


def bootstrap(v, n=2000, semilla=42):
    """IC del 95 % por remuestreo percentil, 2000 iteraciones (Flores et al., 2025)."""
    v = [x for x in v if x is not None]
    if len(v) < 2:
        return None, None
    rng = random.Random(semilla)
    medias = sorted(statistics.mean(rng.choices(v, k=len(v))) for _ in range(n))
    return medias[int(0.025 * n)], medias[int(0.975 * n) - 1]


def percentil(v, q):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    pos = (len(v) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


# ================================================================ métricas de la respuesta
_PAL = re.compile(r"\w+", re.UNICODE)


def _tok(t):
    return [x.lower() for x in _PAL.findall(t or "")]


def _ngramas(t, n):
    from collections import Counter
    return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))


def rouge_n(candidato: str, referencia: str, n: int) -> float:
    a, b = _ngramas(_tok(candidato), n), _ngramas(_tok(referencia), n)
    comunes = sum((a & b).values())
    if not comunes:
        return 0.0
    p, r = comunes / sum(a.values()), comunes / sum(b.values())
    return 2 * p * r / (p + r)


def bleu(candidato: str, referencia: str) -> float:
    """BLEU-4 a nivel de oración con suavizado +1 (Lin y Och, 2004) y penalización por brevedad."""
    c, r = _tok(candidato), _tok(referencia)
    if not c or not r:
        return 0.0
    logs = []
    for n in range(1, 5):
        cn, rn = _ngramas(c, n), _ngramas(r, n)
        total = max(sum(cn.values()), 0)
        coinc = sum((cn & rn).values())
        logs.append(math.log((coinc + (1 if n > 1 else 0)) / (total + (1 if n > 1 else 0))) if total and (coinc or n > 1) else float("-inf"))
    if any(x == float("-inf") for x in logs):
        return 0.0
    bp = 1.0 if len(c) > len(r) else math.exp(1 - len(r) / len(c))
    return bp * math.exp(sum(logs) / 4)


def rouge_l(candidato: str, referencia: str) -> float:
    a = [t.lower() for t in _PAL.findall(candidato or "")]
    b = [t.lower() for t in _PAL.findall(referencia or "")]
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


# ================================================================ principal
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--menciones", action="store_true", help="exportar menciones de los resúmenes (lab)")
    ap.add_argument("--bertscore", action="store_true")
    ap.add_argument("--maquina", default="lab_p5")
    a = ap.parse_args()
    if a.menciones:
        exportar_menciones()
        return

    preguntas = {q["id"]: q for q in leer_jsonl(ruta("eval/preguntas_piloto.jsonl"))}
    llm = slug(config()["llm"]["modelo"])
    corridas = defaultdict(list)    # arquitectura -> [ {id: fila} por repetición ]
    for arq in ("naive", "raptor"):
        for archivo in sorted(ruta("resultados").glob(f"{arq}__*__{llm}__{a.maquina}__*.jsonl")):
            filas = {x["id"]: x for x in leer_jsonl(archivo) if x.get("id") in preguntas}
            if len(filas) == len(preguntas):
                corridas[arq].append(filas)
    if not corridas["naive"] or not corridas["raptor"]:
        raise SystemExit("Faltan resultados del piloto (naive y raptor) en resultados/.")
    if not archivo_menciones().exists():
        print("AVISO: no está logs/" + archivo_menciones().name +
              " -> la regla de mención explícita no se puede aplicar; corre --menciones en el lab.")
    estricta, herencia = cargar_controles_nodos()
    print(f"Repeticiones: naive {len(corridas['naive'])} | raptor {len(corridas['raptor'])}")

    # Similitud semántica: un solo lote de embeddings (respuestas de la 1.ª corrida + referencias)
    ids = sorted(preguntas)
    textos = [preguntas[i]["respuesta_referencia"] for i in ids]
    for arq in ("naive", "raptor"):
        textos += [corridas[arq][0][i].get("respuesta", "") or " " for i in ids]
    vec = embeber(textos, "documento")
    ref_vec = dict(zip(ids, vec[:len(ids)]))
    resp_vec = {"naive": dict(zip(ids, vec[len(ids):2 * len(ids)])),
                "raptor": dict(zip(ids, vec[2 * len(ids):]))}

    bert = {}
    if a.bertscore:
        from bert_score import score
        for arq in ("naive", "raptor"):
            cands = [corridas[arq][0][i].get("respuesta", "") or " " for i in ids]
            refs = [preguntas[i]["respuesta_referencia"] for i in ids]
            _, _, F = score(cands, refs, lang="es", verbose=False)
            bert[arq] = dict(zip(ids, F.tolist()))

    filas_out = []
    for arq in ("naive", "raptor"):
        reps = corridas[arq]
        for i in ids:
            q, r0 = preguntas[i], reps[0][i]
            esp = normset(q["controles_esperados"])
            ranking = r0.get("ranking_ids") or [n["id"] for n in r0["recuperados"]]
            contexto_ids = [n["id"] for n in r0["recuperados"]]
            en_ctx = set().union(*[estricta.get(n, set()) for n in contexto_ids]) if contexto_ids else set()
            en_ctx_her = set().union(*[herencia.get(n, set()) for n in contexto_ids]) if contexto_ids else set()
            citadas = normset(extraer_citas(r0.get("respuesta", "")))
            # Lo que el LLM pudo ver: controles mencionados en el texto + los de las líneas "Cita:" del prompt
            visibles = en_ctx | normset(extraer_citas(" ".join(c for n in r0["recuperados"] for c in n.get("citas", []))))
            fila = {
                "arquitectura": arq, "id": i, "tipo": q["tipo"], "n_esperados": len(esp),
                **{f"recall@{k}": recall_k(ranking, estricta, esp, k) for k in K},
                "mrr": mrr(ranking, estricta, esp),
                "ndcg@10": ndcg_k(ranking, estricta, esp, 10),
                "precision@10": precision_k(ranking, estricta, esp, 10),
                "map": average_precision(ranking, estricta, esp),
                "recall_contexto": len(en_ctx & esp) / len(esp),
                "recall@10_herencia": recall_k(ranking, herencia, esp, 10),
                "mrr_herencia": mrr(ranking, herencia, esp),
                "recall_contexto_herencia": len(en_ctx_her & esp) / len(esp),
                "nodos_contexto": len(contexto_ids),
                "resumenes_contexto": sum(n["nivel"] > 0 for n in r0["recuperados"]),
                "tokens_contexto": r0["tokens_contexto"],
                "citas_en_contexto": (len(citadas & visibles) / len(citadas)) if citadas else None,
                "esperados_citados": len(citadas & esp) / len(esp),
                "sin_citas": not citadas,
                "rouge_l": rouge_l(r0.get("respuesta", ""), q["respuesta_referencia"]),
                "rouge_1": rouge_n(r0.get("respuesta", ""), q["respuesta_referencia"], 1),
                "rouge_2": rouge_n(r0.get("respuesta", ""), q["respuesta_referencia"], 2),
                "bleu": bleu(r0.get("respuesta", ""), q["respuesta_referencia"]),
                "similitud_semantica": float(np.dot(resp_vec[arq][i], ref_vec[i])),
                "bertscore_f1": bert.get(arq, {}).get(i),
                "respuesta_vacia": not (r0.get("respuesta") or "").strip(),
                "cortada": r0.get("fin") == "length",
                # equipo (media entre repeticiones)
                "t_total_s": media([rep[i]["t_total_s"] for rep in reps]),
                "t_total_sd": desv([rep[i]["t_total_s"] for rep in reps]),
                "t_generacion_s": media([rep[i]["t_generacion_s"] for rep in reps]),
                "t_primer_token_s": media([rep[i]["t_primer_token_aprox_s"] for rep in reps]),
                "tokens_por_s": media([rep[i]["tokens_por_s"] for rep in reps]),
                "tokens_entrada": r0["tokens_entrada"],
                "tokens_salida": r0["tokens_salida"],
                "vram_gpu_mb": media([rep[i].get("vram_gpu_mb") for rep in reps]),
                "identica_en_repeticiones": len({rep[i].get("respuesta", "") for rep in reps}) == 1,
            }
            filas_out.append(fila)

    SALIDA.mkdir(parents=True, exist_ok=True)
    with (SALIDA / "por_pregunta.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas_out[0]))
        w.writeheader()
        w.writerows(filas_out)

    # ---------------------------------------------------------- resúmenes
    metricas = ["recall@5", "recall@10", "precision@10", "mrr", "map", "ndcg@10", "recall_contexto",
                "citas_en_contexto", "esperados_citados", "rouge_1", "rouge_2", "rouge_l", "bleu",
                "similitud_semantica",
                "bertscore_f1", "nodos_contexto", "resumenes_contexto", "tokens_contexto",
                "t_total_s", "t_generacion_s", "t_primer_token_s", "tokens_por_s", "tokens_entrada",
                "tokens_salida", "vram_gpu_mb", "recall@10_herencia", "mrr_herencia", "recall_contexto_herencia"]
    resumen = []
    for grupo in ["todas", "puntual", "multicontrol", "transversal"]:
        for arq in ("naive", "raptor"):
            sel = [x for x in filas_out if x["arquitectura"] == arq and (grupo == "todas" or x["tipo"] == grupo)]
            fila = {"grupo": grupo, "arquitectura": arq, "n": len(sel)}
            for m in metricas:
                fila[m] = media([x[m] for x in sel])
                fila[m + "_sd"] = desv([x[m] for x in sel])
                fila[m + "_ic_inf"], fila[m + "_ic_sup"] = bootstrap([x[m] for x in sel])
            fila["sin_citas"] = sum(x["sin_citas"] for x in sel)
            fila["respuestas_vacias"] = sum(x["respuesta_vacia"] for x in sel)
            fila["respuestas_cortadas"] = sum(x["cortada"] for x in sel)
            fila["identicas_en_repeticiones"] = sum(x["identica_en_repeticiones"] for x in sel)
            todas_lat = [rep[x["id"]]["t_total_s"] for rep in corridas[arq] for x in sel]
            fila["t_total_p95"] = percentil(todas_lat, 0.95)
            resumen.append(fila)
    with (SALIDA / "resumen.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(resumen[0]))
        w.writeheader()
        w.writerows(resumen)

    # ---------------------------------------------------------- tablas en markdown
    R = {(x["grupo"], x["arquitectura"]): x for x in resumen}

    def linea(nombre, m, d=3, sd=True, ic=False):
        n, r = R[("todas", "naive")], R[("todas", "raptor")]
        def celda(x):
            if x[m] is None:
                return "—"
            if ic and x.get(m + "_ic_inf") is not None:
                return f"{f(x[m], d)} [{f(x[m + '_ic_inf'], d)}, {f(x[m + '_ic_sup'], d)}]"
            return f(x[m], d) + (f" ± {f(x[m + '_sd'], d)}" if sd else "")
        return f"| {nombre} | {celda(n)} | {celda(r)} |"

    md = ["# Resultados del piloto (20 preguntas, lab_p5)", "",
          f"Repeticiones: naive {len(corridas['naive'])}, RAPTOR {len(corridas['raptor'])}. "
          "Tablas 1 y 2: media [IC 95 % bootstrap percentil, 2000 iteraciones]. "
          "Tabla 3: media ± desviación estándar.", "",
          "## Tabla 1. Recuperación", "", "| Métrica | Naive RAG | RAPTOR |", "|---|---|---|",
          linea("Recall@5", "recall@5", ic=True), linea("Recall@10", "recall@10", ic=True),
          linea("Precision@10", "precision@10", ic=True), linea("MRR", "mrr", ic=True), linea("MAP", "map", ic=True),
          linea("nDCG@10", "ndcg@10", ic=True),
          linea("Recall del contexto (lo que recibe el LLM)", "recall_contexto", ic=True),
          linea("Nodos en el contexto", "nodos_contexto", 1), linea("Resúmenes en el contexto", "resumenes_contexto", 1),
          linea("Tokens de contexto", "tokens_contexto", 0), "",
          "Regla de relevancia: mención explícita para los resúmenes; ISO 27001 A.x = ISO 27002 x.", "",
          "## Tabla 2. Calidad de la respuesta", "", "| Métrica | Naive RAG | RAPTOR |", "|---|---|---|",
          linea("Citas en el contexto", "citas_en_contexto", ic=True),
          linea("Controles esperados citados", "esperados_citados", ic=True),
          linea("ROUGE-1 (F1)", "rouge_1", ic=True), linea("ROUGE-2 (F1)", "rouge_2", ic=True),
          linea("ROUGE-L (F1)", "rouge_l", ic=True), linea("BLEU-4", "bleu", ic=True),
          linea("Answer Similarity (similitud semántica con la referencia)", "similitud_semantica", ic=True)]
    if bert:
        md.append(linea("BERTScore (F1)", "bertscore_f1", ic=True))
    n, r = R[("todas", "naive")], R[("todas", "raptor")]
    md += [f"| Respuestas sin ninguna cita | {n['sin_citas']} | {r['sin_citas']} |",
           f"| Respuestas vacías / cortadas | {n['respuestas_vacias']} / {n['respuestas_cortadas']} | "
           f"{r['respuestas_vacias']} / {r['respuestas_cortadas']} |", "",
           "## Tabla 3. Desempeño del equipo (media entre repeticiones)", "",
           "| Métrica | Naive RAG | RAPTOR |", "|---|---|---|",
           linea("Latencia total por pregunta (s)", "t_total_s", 2),
           f"| Latencia p95 (s) | {f(R[('todas', 'naive')]['t_total_p95'], 2)} | {f(R[('todas', 'raptor')]['t_total_p95'], 2)} |",
           linea("Tiempo de generación (s)", "t_generacion_s", 2),
           linea("Primer token, aprox. (s)", "t_primer_token_s", 2), linea("Tokens/s", "tokens_por_s", 1),
           linea("Tokens de entrada al LLM", "tokens_entrada", 0), linea("Tokens de salida", "tokens_salida", 0),
           linea("VRAM de la GPU (MB)", "vram_gpu_mb", 0, sd=False),
           f"| Respuestas idénticas en todas las repeticiones | {n['identicas_en_repeticiones']}/20 | "
           f"{r['identicas_en_repeticiones']}/20 |", "",
           "## Tabla 4. Por tipo de pregunta", "",
           "| Tipo | n | Recall ctx. naive | Recall ctx. RAPTOR | Esperados citados naive | Esperados citados RAPTOR | ROUGE-L naive | ROUGE-L RAPTOR |",
           "|---|---|---|---|---|---|---|---|"]
    for g in ("puntual", "multicontrol", "transversal"):
        n, r = R[(g, "naive")], R[(g, "raptor")]
        md.append(f"| {g} | {n['n']} | {f(n['recall_contexto'])} | {f(r['recall_contexto'])} | "
                  f"{f(n['esperados_citados'])} | {f(r['esperados_citados'])} | {f(n['rouge_l'])} | {f(r['rouge_l'])} |")
    n, r = R[("todas", "naive")], R[("todas", "raptor")]
    md += ["", "## Tabla 5. Sensibilidad a la regla de relevancia (RAPTOR)", "",
           "| Métrica | Mención explícita (principal) | Herencia (cota superior) |", "|---|---|---|",
           f"| Recall@10 | {f(r['recall@10'])} | {f(r['recall@10_herencia'])} |",
           f"| MRR | {f(r['mrr'])} | {f(r['mrr_herencia'])} |",
           f"| Recall del contexto | {f(r['recall_contexto'])} | {f(r['recall_contexto_herencia'])} |", ""]
    (SALIDA / "tablas.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nArchivos: {SALIDA.relative_to(ruta('.'))}\\por_pregunta.csv, resumen.csv, tablas.md")


if __name__ == "__main__":
    main()
