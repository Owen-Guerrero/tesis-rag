"""
Cargar un árbol RAPTOR ya construido en ChromaDB (sin volver a resumir)
----------------------------------------------------------------------
Para correr la demo en otra máquina (p. ej. la PC de Owen) con EXACTAMENTE el mismo
árbol que se construyó en el laboratorio, sin volver a llamar al LLM:
  1. Lee corpus/procesado/arbol_raptor__<embeddings>__<llm>.jsonl
     (hojas + resúmenes; NO está en GitHub porque contiene texto ISO: se copia por canal privado).
  2. Calcula los embeddings de todos los nodos con el modelo de config.yaml.
  3. Crea las dos colecciones que usan los pipelines y la demo:
       naive-<embeddings>          solo las hojas (nivel 0)
       raptor-<embeddings>-<llm>   hojas + resúmenes (árbol colapsado)

Uso (desde la raíz del repo):
  python pipelines\\cargar_arbol.py
  python pipelines\\cargar_arbol.py --archivo ruta\\al\\arbol.jsonl
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipelines.comun import (agregar_nodos, config, crear_coleccion, embeber, leer_jsonl,  # noqa: E402
                             modelo_embeddings, nombre_coleccion, ruta, slug)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--archivo", help="árbol .jsonl (por defecto el de config.yaml)")
    a = p.parse_args()
    emb, llm = modelo_embeddings(), config()["llm"]["modelo"]
    archivo = Path(a.archivo) if a.archivo else \
        ruta("corpus/procesado") / f"arbol_raptor__{slug(emb)}__{slug(llm)}.jsonl"
    if not archivo.exists():
        raise SystemExit(f"No está {archivo}. Cópialo desde la carpeta privada (Drive) a corpus\\procesado\\")

    nodos = leer_jsonl(archivo)
    for n in nodos:
        if "controles" not in n:
            n["controles"] = [c for c, _ in Counter(n.get("conteo", {})).most_common()]
    hojas = [n for n in nodos if n["nivel"] == 0]
    print(f"Árbol: {archivo.name} | {len(hojas)} hojas + {len(nodos) - len(hojas)} resúmenes "
          f"| capas: {dict(sorted(Counter(n['nivel'] for n in nodos).items()))}")

    print(f"Calculando embeddings con {emb} (unos minutos)...")
    t0 = time.perf_counter()
    vectores = []
    for i in range(0, len(nodos), 500):
        vectores.append(embeber([n["texto"] for n in nodos[i:i + 500]], "documento", emb))
        print(f"  {min(i + 500, len(nodos))}/{len(nodos)} | {time.perf_counter() - t0:.0f} s")
    vectores = np.vstack(vectores)

    es_hoja = np.array([n["nivel"] == 0 for n in nodos])
    col_naive = crear_coleccion(nombre_coleccion("naive", emb))
    agregar_nodos(col_naive, hojas, vectores[es_hoja])
    col_raptor = crear_coleccion(nombre_coleccion("raptor", emb))
    agregar_nodos(col_raptor, nodos, vectores)
    print(f"\nListo en {time.perf_counter() - t0:.0f} s:")
    print(f"  {col_naive.name}: {col_naive.count()} nodos")
    print(f"  {col_raptor.name}: {col_raptor.count()} nodos")
    print('Prueba: python pipelines\\demo.py "¿Qué exige NIST para bloquear cuentas tras intentos fallidos?"')


if __name__ == "__main__":
    main()
