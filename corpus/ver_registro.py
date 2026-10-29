"""
Muestra uno o más registros del corpus por su id de control (útil para revisar y para exponer).

Uso (desde la raíz del repo, con el .venv activo):
  python corpus\\ver_registro.py AC-2            (NIST)
  python corpus\\ver_registro.py A01:2025         (OWASP, todas sus secciones)
  python corpus\\ver_registro.py 5.15             (ISO 27002 control 5.15)
  python corpus\\ver_registro.py A.5.15           (ISO 27001 Anexo A)
  python corpus\\ver_registro.py AC-2 --completo  (sin recortar el texto)
"""

import json
import sys
from pathlib import Path

PROCESADO = Path(__file__).resolve().parent / "procesado"
# Solo los registros extraídos de cada norma (no las hojas ni el árbol RAPTOR)
FUENTES = ["nist_sp800-53.jsonl", "owasp_top10_2025.jsonl", "iso27001_2022.jsonl", "iso27002_2022.jsonl"]


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    buscado, completo = sys.argv[1], "--completo" in sys.argv
    encontrados = 0
    for archivo in [PROCESADO / f for f in FUENTES if (PROCESADO / f).exists()]:
        for linea in archivo.open(encoding="utf-8"):
            r = json.loads(linea)
            if r.get("id_control") != buscado:
                continue
            encontrados += 1
            texto = r.pop("texto")
            r["relacionados"] = ", ".join(r["relacionados"])   # en una sola línea
            print(f"\n=== {archivo.name} ===")
            print(json.dumps(r, ensure_ascii=False, indent=2))
            print("--- texto ---")
            print(texto if completo or len(texto) < 1500 else texto[:1500] + "\n[...] (usar --completo)")
    print(f"\n{encontrados} registro(s) encontrados para '{buscado}'.")


if __name__ == "__main__":
    main()
