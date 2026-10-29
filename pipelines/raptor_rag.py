"""
OE6 - RAPTOR con recuperación en árbol colapsado (Sarthi et al., 2024)
---------------------------------------------------------------------
Es EXACTAMENTE el mismo flujo de naive RAG (mismo embedding de la pregunta,
mismo presupuesto de 2000 tokens, mismo prompt y mismo LLM). Lo único que
cambia es el índice: aquí la colección contiene las hojas Y los resúmenes de
todas las capas del árbol, y la búsqueda elige entre todos ellos por similitud.

Uso (después de pipelines\\raptor_arbol.py):
  python pipelines\\raptor_rag.py --maquina pc_owen "¿Qué exige NIST para la gestión de cuentas?"
  python pipelines\\raptor_rag.py --maquina lab_p5 --preguntas eval\\preguntas_piloto.jsonl
  python pipelines\\raptor_rag.py --maquina pc_owen --coleccion <colección de prueba> "..."
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipelines.naive_rag import main  # noqa: E402

if __name__ == "__main__":
    main("raptor")
