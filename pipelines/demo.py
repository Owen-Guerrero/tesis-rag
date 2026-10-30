"""
Demostración en vivo: una pregunta, las dos arquitecturas, lado a lado
--------------------------------------------------------------------
Uso (desde la raíz del repo):
  python pipelines\\demo.py --calentar                     (antes de exponer: carga los modelos)
  python pipelines\\demo.py "¿Qué exige NIST para bloquear cuentas tras intentos fallidos?"
  python pipelines\\demo.py --solo naive "pregunta..."      (si no está el árbol de RAPTOR)
  python pipelines\\demo.py --modelo-llm llama3.1:8b "..."  (respaldo rápido si la GPU es chica)

Muestra para cada arquitectura: qué recuperó (hojas y resúmenes), la respuesta
con sus citas y el tiempo. Usa exactamente la misma configuración del experimento.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipelines.comun import (abrir_coleccion, armar_prompt, config, embeber,  # noqa: E402
                             generar, nombre_coleccion, recuperar)
from pipelines import comun  # noqa: E402


def correr(arquitectura: str, pregunta: str, modelo_llm: str) -> None:
    col = abrir_coleccion(nombre_coleccion(arquitectura))
    t0 = time.perf_counter()
    rec = recuperar(col, pregunta)
    r = generar(armar_prompt(pregunta, rec["seleccion"]), modelo_llm)
    total = time.perf_counter() - t0
    sel = rec["seleccion"]
    resumenes = sum(n["nivel"] > 0 for n in sel)
    print("\n" + "=" * 90)
    print(f"{arquitectura.upper()}  |  {len(sel)} nodos en el contexto ({resumenes} resúmenes, "
          f"{rec['tokens_contexto']} tokens)  |  {total:.1f} s")
    print("=" * 90)
    print("Lo más parecido a la pregunta:")
    for n in sel[:6]:
        tipo = "hoja" if n["nivel"] == 0 else f"resumen capa {n['nivel']}"
        print(f"  {n['similitud']:.3f}  {tipo:<16} {' '.join(n['citas'][:3])}")
    print("\nRespuesta:\n")
    print(r["respuesta"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pregunta", nargs="*")
    p.add_argument("--solo", choices=["naive", "raptor"])
    p.add_argument("--modelo-llm")
    p.add_argument("--calentar", action="store_true")
    a = p.parse_args()
    modelo = a.modelo_llm or config()["llm"]["modelo"]
    if a.calentar:
        print(f"Cargando {modelo} y el modelo de embeddings...")
        if hasattr(comun, "calentar"):
            comun.calentar(modelo)
        else:
            generar([{"role": "user", "content": "Hola"}], modelo)
        embeber(["prueba"], "consulta")
        print("Listo.")
        return
    pregunta = " ".join(a.pregunta)
    if not pregunta:
        raise SystemExit('Escribe la pregunta entre comillas: python pipelines\\demo.py "..."')
    print(f"Pregunta: {pregunta}\nLLM: {modelo}  |  embeddings: {config()['embeddings']['modelo']}")
    for arq in ([a.solo] if a.solo else ["naive", "raptor"]):
        correr(arq, pregunta, modelo)


if __name__ == "__main__":
    main()
