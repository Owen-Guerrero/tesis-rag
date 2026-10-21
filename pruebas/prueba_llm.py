"""
Etapa 1 - Prueba mínima del LLM local
--------------------------------------
Qué verifica:
  1. Que el modelo responde desde Python (no solo desde la terminal).
  2. Que responde en ESPAÑOL cuando el contexto está en INGLÉS
     (el escenario real de la tesis; ver Marchisio et al., 2024).
  3. Registra métricas para la ficha técnica: latencia, tokens/s y VRAM.

Cómo se mide:
  - 1 corrida de calentamiento que se descarta (carga el modelo en la GPU).
  - N corridas medidas (N = verificacion.repeticiones en config.yaml).
  - Todo se guarda en logs/ como JSON: es evidencia para el artículo.

Ubicación: pruebas/prueba_llm.py  (config.yaml va en la raíz del proyecto)
"""

import json
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

# Contexto de prueba: paráfrasis en inglés del control AC-2 de NIST SP 800-53.
# NO es el texto oficial; solo sirve para probar el comportamiento del modelo.
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

NS = 1e9  # Ollama reporta tiempos en nanosegundos


def vram_usada_mb() -> int:
    """Lee la VRAM ocupada en la GPU con nvidia-smi."""
    salida = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return int(salida.stdout.strip().splitlines()[0])


def info_modelo() -> dict:
    """Datos del modelo para la ficha técnica (tamaño y cuantización)."""
    d = ollama.show(LLM["modelo"]).details
    version_ollama = subprocess.run(
        ["ollama", "--version"], capture_output=True, text=True
    ).stdout.strip()
    return {
        "modelo": LLM["modelo"],
        "familia": d.family,
        "parametros": d.parameter_size,
        "cuantizacion": d.quantization_level,
        "formato": d.format,
        "ollama": version_ollama,
    }


def consultar() -> dict:
    """Envía una consulta al modelo y devuelve la respuesta con sus métricas."""
    r = ollama.chat(
        model=LLM["modelo"],
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
    )
    vram = vram_usada_mb()  # el modelo sigue cargado justo después de responder

    tokens_salida = r.eval_count or 0
    t_generacion = (r.eval_duration or 0) / NS
    t_entrada = (r.prompt_eval_duration or 0) / NS
    t_carga = (r.load_duration or 0) / NS
    return {
        "respuesta": r.message.content,
        "tokens_entrada": r.prompt_eval_count or 0,
        "tokens_salida": tokens_salida,
        "t_primer_token_aprox_s": round(t_carga + t_entrada, 3),
        "t_total_s": round((r.total_duration or 0) / NS, 3),
        "tokens_por_s": round(tokens_salida / t_generacion, 2) if t_generacion else None,
        "vram_mb": vram,
    }


def main() -> None:
    print(f"Modelo: {LLM['modelo']}  |  num_ctx: {LLM['num_ctx']}")
    print(f"VRAM antes de cargar el modelo: {vram_usada_mb()} MB\n")

    print("Corrida de calentamiento (se descarta)...")
    consultar()

    corridas = []
    for i in range(1, REPETICIONES + 1):
        res = consultar()
        corridas.append(res)
        print(f"Corrida {i}: total {res['t_total_s']} s | "
              f"{res['tokens_por_s']} tokens/s | VRAM {res['vram_mb']} MB")

    def resumen(campo):
        valores = [c[campo] for c in corridas if c[campo] is not None]
        return {
            "media": round(statistics.mean(valores), 3),
            "desv": round(statistics.stdev(valores), 3) if len(valores) > 1 else 0.0,
        }

    registro = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelo": info_modelo(),
        "parametros_inferencia": LLM,
        "hardware": CONFIG["hardware"],
        "resumen": {
            "t_total_s": resumen("t_total_s"),
            "tokens_por_s": resumen("tokens_por_s"),
            "vram_mb_max": max(c["vram_mb"] for c in corridas),
        },
        "corridas": corridas,
    }

    carpeta_logs = RAIZ / "logs"
    carpeta_logs.mkdir(exist_ok=True)
    nombre = f"etapa1_llm_{LLM['modelo'].replace(':', '-')}_{datetime.now():%Y%m%d_%H%M}.json"
    (carpeta_logs / nombre).write_text(
        json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n--- Respuesta del modelo (revisa que esté en español) ---")
    print(corridas[-1]["respuesta"])
    print(f"\nRegistro guardado en logs/{nombre}")


if __name__ == "__main__":
    main()
