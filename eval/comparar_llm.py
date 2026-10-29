"""
OE2 - Comparación de calidad de los LLM candidatos (sobre resultados ya generados)
---------------------------------------------------------------------------------
Lee los archivos de resultados/ de naive RAG (uno por LLM, MISMA recuperación) y
vuelve a calcular las métricas de citas con la versión actual de eval/metricas.py,
para que todos los modelos se midan con exactamente la misma regla.
No vuelve a llamar a ningún LLM.

Uso (desde la raíz del repo):
  python eval\\comparar_llm.py resultados\\naive__qwen3-embedding-0.6b__*__lab_p5__20261029_*.jsonl
  python eval\\comparar_llm.py <archivos> --ver S04      (muestra la respuesta de cada modelo a S04)

Guarda la tabla en logs/comparacion_llm_calidad.csv.
"""

import argparse
import csv
import glob
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.metricas import metricas_citas  # noqa: E402
from pipelines.comun import leer_jsonl, ruta  # noqa: E402


def esperados_por_id() -> dict:
    salida = {}
    for f in ("eval/preguntas_seleccion.jsonl", "eval/preguntas_piloto.jsonl"):
        if ruta(f).exists():
            salida.update({q["id"]: set(q.get("controles_esperados", [])) for q in leer_jsonl(ruta(f))})
    return salida


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("archivos", nargs="+")
    p.add_argument("--ver", help="id de pregunta cuya respuesta se muestra para cada modelo")
    a = p.parse_args()
    archivos = sorted({f for patron in a.archivos for f in glob.glob(patron)})
    esperados = esperados_por_id()

    tabla = []
    for archivo in archivos:
        filas = [f for f in leer_jsonl(archivo) if "pregunta" in f]
        if not filas or not filas[0].get("modelo_llm"):
            continue
        modelo = filas[0]["modelo_llm"]
        cc, ce, sin_cita, detalle = [], [], 0, []
        for f in filas:
            contexto = {c for n in f["recuperados"] for c in n["controles"]}
            m = metricas_citas(f.get("respuesta", ""), contexto, esperados.get(f["id"], set()))
            if m["citas_en_contexto"] is not None:
                cc.append(m["citas_en_contexto"])
            else:
                sin_cita += 1
            if m["citas_esperadas"] is not None:
                ce.append(m["citas_esperadas"])
            detalle.append(f"{f['id']}: {', '.join(m['citas']) or 'ninguna'}")
            if a.ver == f["id"]:
                print(f"\n===== {modelo} | {f['id']} =====\n{f.get('respuesta', '')}\n")
        fila = {
            "modelo_llm": modelo,
            "preguntas": len(filas),
            "citas_en_contexto": round(statistics.mean(cc), 3) if cc else None,
            "controles_esperados_citados": round(statistics.mean(ce), 3) if ce else None,
            "respuestas_sin_ninguna_cita": sin_cita,
            "t_generacion_media_s": round(statistics.mean(f["t_generacion_s"] for f in filas), 2),
            "tokens_por_s_medio": round(statistics.mean(f["tokens_por_s"] for f in filas if f.get("tokens_por_s")), 1),
            "tokens_salida_medio": round(statistics.mean(f["tokens_salida"] for f in filas), 0),
            "archivo": Path(archivo).name,
        }
        tabla.append(fila)
        print(f"{modelo:<14} citas en contexto {fila['citas_en_contexto']} | esperados citados "
              f"{fila['controles_esperados_citados']} | sin cita {sin_cita}/{len(filas)} | "
              f"{fila['t_generacion_media_s']} s | {fila['tokens_por_s_medio']} tok/s")
        print("               " + " | ".join(detalle))

    if tabla:
        ruta("logs").mkdir(exist_ok=True)
        with open(ruta("logs") / "comparacion_llm_calidad.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(tabla[0]))
            w.writeheader()
            w.writerows(tabla)
        print("\nTabla: logs/comparacion_llm_calidad.csv")


if __name__ == "__main__":
    main()
