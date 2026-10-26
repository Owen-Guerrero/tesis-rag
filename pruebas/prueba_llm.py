"""
OE2 / OE3 - Verificación y comparación de LLM locales
------------------------------------------------------
Qué hace, para cada modelo indicado:
  1. Descarga de la GPU cualquier modelo que Ollama tenga cargado, para medir la
     VRAM "en reposo" de forma correcta.
  2. Hace 1 corrida de calentamiento (se descarta) y N corridas medidas
     (N = verificacion.repeticiones en config.yaml) con la misma consulta.
  3. Mide: tiempo total, tiempo hasta el primer token (aprox.), tokens/s,
     VRAM total de la GPU y VRAM que Ollama reporta para el modelo.
  4. Guarda un JSON detallado en logs/ y agrega una fila a logs/resumen_llm.csv
     (esa tabla es la base de la tabla de selección del OE2).

La consulta de prueba: párrafo en inglés del control AC-2 + pregunta en español
(escenario real de la tesis; ver Marchisio et al., 2024).

Uso (desde la raíz del repo, con el .venv activo):
  python pruebas\\prueba_llm.py --maquina pc_owen                  (modelo de config.yaml)
  python pruebas\\prueba_llm.py --maquina lab_p5 --candidatos      (todos los candidatos)
  python pruebas\\prueba_llm.py --maquina lab_p5 --modelo qwen2.5:14b gpt-oss:20b
"""

import argparse
import csv
import json
import platform
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

import ollama
import yaml

RAIZ = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
LLM = CONFIG["llm"]
REPETICIONES = CONFIG["verificacion"]["repeticiones"]
NS = 1e9  # Ollama reporta tiempos en nanosegundos

# Paráfrasis en inglés del control AC-2 (no es el texto oficial; solo para la prueba).
CONTEXTO = (
    "AC-2 Account Management. The organization defines the types of system "
    "accounts that are allowed and prohibited, assigns account managers, and "
    "establishes conditions for group and role membership. Accounts are created, "
    "enabled, modified, disabled, and removed in accordance with organizational "
    "policy. Account managers are notified when accounts are no longer required, "
    "when users are terminated or transferred, and when system usage or "
    "need-to-know changes. The organization monitors the use of accounts and "
    "reviews them for compliance with account management requirements at a "
    "defined frequency."
)
PREGUNTA = (
    "¿Qué debería revisar un auditor para verificar que la organización cumple "
    "con la gestión de cuentas descrita en AC-2?"
)
INSTRUCCION_SISTEMA = (
    "Eres un asistente de apoyo a auditorías de seguridad de la información. "
    "Responde siempre en español y únicamente con base en el contexto proporcionado."
)


# ---------- hardware ----------
def nvidia(campos: str) -> list:
    salida = subprocess.run(
        ["nvidia-smi", f"--query-gpu={campos}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return [c.strip() for c in salida.stdout.strip().splitlines()[0].split(",")]


def vram_usada_mb() -> int:
    return int(nvidia("memory.used")[0])


def info_maquina(nombre: str) -> dict:
    datos = yaml.safe_load((RAIZ / "maquinas" / f"{nombre}.yaml").read_text(encoding="utf-8"))
    gpu, driver, vram_total = nvidia("name,driver_version,memory.total")
    datos.update({
        "gpu": gpu,
        "driver_nvidia": driver,
        "vram_total_mb": int(vram_total),
        "sistema_operativo": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "ollama": subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip(),
    })
    return datos


# ---------- modelo ----------
def descargar_modelos_de_gpu() -> None:
    """Libera la GPU: pide a Ollama descargar todo modelo que tenga cargado."""
    for m in ollama.ps().models:
        ollama.generate(model=m.model, keep_alive=0)


def info_modelo(modelo: str) -> dict:
    d = ollama.show(modelo).details
    return {
        "modelo": modelo,
        "familia": d.family,
        "parametros": d.parameter_size,
        "cuantizacion": d.quantization_level,
        "formato": d.format,
    }


def consultar(modelo: str) -> dict:
    extra = {} if LLM.get("razonamiento") is None else {"think": LLM["razonamiento"]}
    r = ollama.chat(
        model=modelo,
        messages=[
            {"role": "system", "content": INSTRUCCION_SISTEMA},
            {"role": "user", "content": f"Contexto:\n{CONTEXTO}\n\nPregunta: {PREGUNTA}"},
        ],
        options={
            "temperature": LLM["temperatura"],
            "top_p": LLM["top_p"],
            "seed": LLM["semilla"],
            "num_ctx": LLM["num_ctx"],
            "num_predict": LLM["max_tokens"],
        },
        **extra,
    )
    vram_gpu = vram_usada_mb()
    vram_modelo = next((m.size_vram for m in ollama.ps().models if m.model == modelo), 0) or 0

    tokens_salida = r.eval_count or 0
    t_gen = (r.eval_duration or 0) / NS
    return {
        "respuesta": r.message.content,
        "tokens_entrada": r.prompt_eval_count or 0,
        "tokens_salida": tokens_salida,
        "t_primer_token_aprox_s": round(((r.load_duration or 0) + (r.prompt_eval_duration or 0)) / NS, 3),
        "t_total_s": round((r.total_duration or 0) / NS, 3),
        "tokens_por_s": round(tokens_salida / t_gen, 2) if t_gen else None,
        "vram_gpu_mb": vram_gpu,
        "vram_modelo_mb": round(vram_modelo / 1024**2),
    }


def resumen(corridas: list, campo: str) -> dict:
    v = [c[campo] for c in corridas if c[campo] is not None]
    return {"media": round(statistics.mean(v), 3),
            "desv": round(statistics.stdev(v), 3) if len(v) > 1 else 0.0}


def verificar(modelo: str, maquina: dict) -> dict:
    print(f"\n=== {modelo} en {maquina['nombre']} ===")
    descargar_modelos_de_gpu()
    vram_reposo = vram_usada_mb()
    print(f"VRAM en reposo (sin modelos cargados): {vram_reposo} MB")

    print("Corrida de calentamiento (se descarta)...")
    calentamiento = consultar(modelo)

    corridas = []
    for i in range(1, REPETICIONES + 1):
        c = consultar(modelo)
        corridas.append(c)
        print(f"Corrida {i}: {c['t_total_s']} s | {c['tokens_por_s']} tokens/s | "
              f"VRAM GPU {c['vram_gpu_mb']} MB | modelo {c['vram_modelo_mb']} MB")

    registro = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "maquina": maquina,
        "modelo": info_modelo(modelo),
        "parametros_inferencia": {**LLM, "modelo": modelo},
        "vram_reposo_mb": vram_reposo,
        "carga_en_frio_s": calentamiento["t_primer_token_aprox_s"],
        "resumen": {
            "t_total_s": resumen(corridas, "t_total_s"),
            "t_primer_token_s": resumen(corridas, "t_primer_token_aprox_s"),
            "tokens_por_s": resumen(corridas, "tokens_por_s"),
            "vram_gpu_max_mb": max(c["vram_gpu_mb"] for c in corridas),
            "vram_modelo_mb": max(c["vram_modelo_mb"] for c in corridas),
        },
        "corridas": corridas,
    }

    logs = RAIZ / "logs"
    logs.mkdir(exist_ok=True)
    nombre = f"verif_llm_{maquina['nombre']}_{modelo.replace(':', '-')}_{datetime.now():%Y%m%d_%H%M}.json"
    (logs / nombre).write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")

    fila = {
        "fecha": registro["fecha"], "maquina": maquina["nombre"], "gpu": maquina["gpu"],
        "modelo": modelo, "parametros": registro["modelo"]["parametros"],
        "cuantizacion": registro["modelo"]["cuantizacion"],
        "vram_reposo_mb": vram_reposo,
        "vram_modelo_mb": registro["resumen"]["vram_modelo_mb"],
        "vram_gpu_max_mb": registro["resumen"]["vram_gpu_max_mb"],
        "carga_en_frio_s": registro["carga_en_frio_s"],
        "t_primer_token_s": registro["resumen"]["t_primer_token_s"]["media"],
        "t_total_s_media": registro["resumen"]["t_total_s"]["media"],
        "t_total_s_desv": registro["resumen"]["t_total_s"]["desv"],
        "tokens_por_s_media": registro["resumen"]["tokens_por_s"]["media"],
        "tokens_por_s_desv": registro["resumen"]["tokens_por_s"]["desv"],
        "log": nombre,
    }
    csv_path = logs / "resumen_llm.csv"
    nuevo = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fila))
        if nuevo:
            w.writeheader()
        w.writerow(fila)

    print("--- Respuesta (revisar que esté en español y basada en el contexto) ---")
    print(corridas[-1]["respuesta"][:1200])
    print(f"Registro: logs/{nombre}")
    return registro


def main() -> None:
    ap = argparse.ArgumentParser(description="Verificación y comparación de LLM locales (OE2/OE3)")
    ap.add_argument("--maquina", required=True, help="nombre del archivo en maquinas/ (pc_owen, lab_p5)")
    ap.add_argument("--modelo", nargs="+", help="uno o más modelos de Ollama")
    ap.add_argument("--candidatos", action="store_true", help="probar todos los candidatos de config.yaml")
    args = ap.parse_args()

    maquina = info_maquina(args.maquina)
    print(f"Máquina: {maquina['nombre']} | GPU: {maquina['gpu']} ({maquina['vram_total_mb']} MB) "
          f"| {maquina['sistema_operativo']}")

    modelos = CONFIG["candidatos_llm"] if args.candidatos else (args.modelo or [LLM["modelo"]])
    disponibles = {m.model for m in ollama.list().models}
    for modelo in modelos:
        if modelo not in disponibles:
            print(f"\n[!] {modelo} no está descargado. Ejecuta primero: ollama pull {modelo}")
            continue
        try:
            verificar(modelo, maquina)
        except Exception as e:  # p. ej. el modelo no entra en la VRAM
            print(f"\n[!] {modelo} falló: {e}")
    descargar_modelos_de_gpu()


if __name__ == "__main__":
    main()
