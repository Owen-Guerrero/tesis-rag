"""
OE5 - Naive RAG (línea base)
----------------------------
Flujo: pregunta (español) -> embedding -> búsqueda por similitud en ChromaDB
       -> se agregan hojas en orden de similitud hasta llenar el presupuesto de
       tokens -> prompt común -> LLM local -> respuesta en español con citas.

Por qué un presupuesto de tokens y no un top-k fijo:
  RAPTOR (árbol colapsado) recupera nodos hasta un máximo de tokens. Para que
  ninguna arquitectura le pase más texto al LLM que la otra, ambas usan el
  mismo presupuesto (Sarthi et al., 2024; Laitenberger et al., 2025).

Uso (desde la raíz del repo, después de pipelines\\indexar.py):
  python pipelines\\naive_rag.py --maquina pc_owen "¿Qué exige NIST para la gestión de cuentas?"
  python pipelines\\naive_rag.py --maquina pc_owen --preguntas eval\\preguntas_seleccion.jsonl
  python pipelines\\naive_rag.py --maquina lab_p5 --preguntas ... --solo-recuperacion   (sin LLM)

En modo --preguntas guarda un JSONL en resultados/ con, por pregunta: ids y
similitud de lo recuperado, respuesta y tiempos. NO guarda el texto del
contexto (contiene texto ISO); se reconstruye con los ids.
"""

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.metricas import mrr, recall_at_k  # noqa: E402
from pipelines.comun import (abrir_coleccion, armar_prompt, calentar, config, escribir_jsonl,  # noqa: E402
                             generar, info_maquina, leer_jsonl, modelo_embeddings,
                             nombre_coleccion, recuperar, ruta, slug, vram_usada_mb)

ARQUITECTURA = "naive"


def responder(col, pregunta: str, modelo_emb: str, solo_recuperacion: bool = False) -> dict:
    t0 = time.perf_counter()
    rec = recuperar(col, pregunta, modelo_emb)
    salida = {
        "recuperados": [{k: c[k] for k in ("id", "controles", "citas", "nivel", "similitud", "n_tokens")}
                        for c in rec["seleccion"]],
        "ranking_controles": [c["controles"] for c in rec["candidatos"]],
        "tokens_contexto": rec["tokens_contexto"],
        "t_embedding_consulta_s": rec["t_embedding_consulta_s"],
        "t_busqueda_s": rec["t_busqueda_s"],
    }
    if not solo_recuperacion:
        salida.update(generar(armar_prompt(pregunta, rec["seleccion"])))
        salida["vram_gpu_mb"] = vram_usada_mb()
    salida["t_total_s"] = round(time.perf_counter() - t0, 3)
    return salida


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pregunta", nargs="*", help="pregunta suelta (entre comillas)")
    p.add_argument("--maquina", required=True, help="pc_owen o lab_p5")
    p.add_argument("--preguntas", help="archivo JSONL con preguntas (campos id, pregunta, controles_esperados)")
    p.add_argument("--modelo-emb", help="modelo de embeddings (por defecto, el de config.yaml)")
    p.add_argument("--solo-recuperacion", action="store_true", help="no llama al LLM (OE4)")
    a = p.parse_args()

    modelo_emb = a.modelo_emb or modelo_embeddings()
    modelo_llm = config()["llm"]["modelo"]
    col = abrir_coleccion(nombre_coleccion(ARQUITECTURA, modelo_emb))

    # ---- pregunta suelta: se muestra en pantalla (incluye la carga del modelo si no estaba en la GPU)
    if not a.preguntas:
        pregunta = " ".join(a.pregunta) or "¿Qué exige NIST para gestionar el ciclo de vida de las cuentas?"
        r = responder(col, pregunta, modelo_emb, a.solo_recuperacion)
        print(f"Hojas en el contexto: {len(r['recuperados'])} ({r['tokens_contexto']} tokens)")
        for c in r["recuperados"][:10]:
            print(f"  {c['similitud']:.3f}  {c['citas'][0]:<20} {c['id']}")
        if not a.solo_recuperacion:
            print(f"\n--- Respuesta de {modelo_llm} ({r['t_total_s']} s, "
                  f"{r['tokens_por_s']} tokens/s) ---\n{r['respuesta']}")
        return

    # ---- lote de preguntas: se guarda en resultados/
    preguntas = leer_jsonl(a.preguntas)
    if not a.solo_recuperacion:
        print(f"Cargando {modelo_llm} en la GPU (corrida de calentamiento, se descarta)...")
        calentar()
    filas, r5, r10, rr = [], [], [], []
    for q in preguntas:
        r = responder(col, q["pregunta"], modelo_emb, a.solo_recuperacion)
        fila = {"id": q["id"], "pregunta": q["pregunta"], "arquitectura": ARQUITECTURA,
                "modelo_embeddings": modelo_emb, "modelo_llm": None if a.solo_recuperacion else modelo_llm,
                **r}
        esperados = set(q.get("controles_esperados", []))
        if esperados:
            fila["recall@5"] = recall_at_k(r["ranking_controles"], esperados, 5)
            fila["recall@10"] = recall_at_k(r["ranking_controles"], esperados, 10)
            fila["mrr"] = mrr(r["ranking_controles"], esperados)
            r5.append(fila["recall@5"]); r10.append(fila["recall@10"]); rr.append(fila["mrr"])
        filas.append(fila)
        extra = f" | R@5 {fila['recall@5']:.2f} MRR {fila['mrr']:.2f}" if esperados else ""
        print(f"{q['id']}: {len(r['recuperados'])} hojas, {r['t_total_s']} s{extra}")

    fecha = datetime.now().strftime("%Y%m%d_%H%M")
    llm = "sin-llm" if a.solo_recuperacion else slug(modelo_llm)
    salida = ruta("resultados") / f"{ARQUITECTURA}__{slug(modelo_emb)}__{llm}__{a.maquina}__{fecha}.jsonl"
    escribir_jsonl(salida, [{"maquina": info_maquina(a.maquina), "fecha": fecha}] + filas)
    if rr:
        print(f"\nPromedio: Recall@5 {statistics.mean(r5):.3f} | Recall@10 {statistics.mean(r10):.3f} "
              f"| MRR {statistics.mean(rr):.3f}")
    print(f"Resultados: {salida.relative_to(ruta('.'))}")


if __name__ == "__main__":
    main()
