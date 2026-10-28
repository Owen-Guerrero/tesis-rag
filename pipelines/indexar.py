"""
OE5 - Indexación de las hojas para naive RAG
--------------------------------------------
Qué hace:
  1. Lee las hojas (corpus/procesado/hojas.jsonl).
  2. Convierte cada hoja en un vector con el modelo de embeddings (vía Ollama).
  3. Las guarda (texto + vector + metadatos) en la colección "naive-<modelo>" de ChromaDB.
La colección se crea desde cero cada vez, para que el índice corresponda
exactamente al corpus y a la configuración actuales.

El tiempo de indexación es una métrica de equipo: se compara después con el
costo de construir el árbol de RAPTOR.

Uso (desde la raíz del repo):
  python pipelines\\indexar.py --maquina pc_owen
  python pipelines\\indexar.py --maquina lab_p5 --modelo-emb qwen3-embedding:0.6b   (OE4)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipelines.comun import (agregar_nodos, cargar_hojas, cita, control_base, crear_coleccion,  # noqa: E402
                             digest_ollama, embeber, info_maquina, modelo_embeddings,
                             nombre_coleccion, ruta, slug)


def indexar(maquina: str, modelo_emb: str = None) -> dict:
    modelo_emb = modelo_emb or modelo_embeddings()
    hojas = cargar_hojas()
    nodos = [{"id": h["id_hoja"], "texto": h["texto"], "n_tokens": h["n_tokens"], "nivel": 0,
              "controles": [control_base(h)], "citas": [cita(h)]} for h in hojas]

    print(f"Generando embeddings de {len(nodos)} hojas con {modelo_emb} ...")
    inicio = time.perf_counter()
    vectores = embeber([n["texto"] for n in nodos], "documento", modelo_emb)
    t_emb = time.perf_counter() - inicio

    col = crear_coleccion(nombre_coleccion("naive", modelo_emb))
    agregar_nodos(col, nodos, vectores)
    t_total = time.perf_counter() - inicio

    registro = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "maquina": info_maquina(maquina),
        "coleccion": col.name,
        "modelo_embeddings": modelo_emb,
        "digest_modelo": digest_ollama(modelo_emb),
        "dimension": int(vectores.shape[1]),
        "nodos_indexados": col.count(),
        "t_embeddings_s": round(t_emb, 2),
        "t_total_s": round(t_total, 2),
        "hojas_por_s": round(len(nodos) / t_emb, 1),
    }
    ruta("logs").mkdir(exist_ok=True)
    salida = ruta("logs") / f"indexacion_naive_{slug(modelo_emb)}_{maquina}.json"
    salida.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Colección {col.name}: {col.count()} hojas | dimensión {registro['dimension']} | "
          f"{registro['t_total_s']} s ({registro['hojas_por_s']} hojas/s)")
    print(f"Registro: {salida.relative_to(ruta('.'))}")
    return registro


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--maquina", required=True, help="pc_owen o lab_p5 (archivo en maquinas/)")
    p.add_argument("--modelo-emb", help="modelo de embeddings (por defecto, el de config.yaml)")
    a = p.parse_args()
    indexar(a.maquina, a.modelo_emb)
