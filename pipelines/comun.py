"""
Funciones compartidas por naive RAG y RAPTOR
--------------------------------------------
Aquí vive todo lo que las dos arquitecturas deben hacer EXACTAMENTE igual:
leer la configuración, generar embeddings, consultar ChromaDB, aplicar el
presupuesto de tokens, armar el prompt y llamar al LLM.
Si esto estuviera copiado en cada pipeline podría diferir sin querer y la
comparación dejaría de ser controlada: lo único que debe cambiar entre
naive RAG y RAPTOR es QUÉ nodos hay en el índice (solo hojas, o hojas + resúmenes).
"""

import json
import platform
import re
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

RAIZ = Path(__file__).resolve().parent.parent
NS = 1e9  # Ollama reporta tiempos en nanosegundos


# ---------------------------------------------------------------- configuración y archivos
@lru_cache(maxsize=1)
def config() -> dict:
    return yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))


def ruta(relativa: str) -> Path:
    return RAIZ / relativa


def leer_jsonl(path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def escribir_jsonl(path, filas) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for fila in filas:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")


def cargar_hojas() -> list:
    archivo = ruta("corpus/procesado/hojas.jsonl")
    if not archivo.exists():
        raise SystemExit("Falta corpus/procesado/hojas.jsonl: corre antes corpus\\fragmentar.py")
    return leer_jsonl(archivo)


def slug(texto: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", texto).strip("-.")


# ---------------------------------------------------------------- identificador de control
PREFIJO_NORMA = {
    "NIST SP 800-53": "NIST",
    "OWASP Top 10": "OWASP",
    "ISO/IEC 27001": "ISO27001",
    "ISO/IEC 27002": "ISO27002",
}


def control_base(h: dict) -> str:
    """Identificador único del control al que pertenece una hoja (unidad de relevancia).

    Las mejoras de NIST (p. ej. AC-2(1)) cuentan como su control base (AC-2):
    así las anotaciones de relevancia y las citas se hacen por control, no por mejora.
    Ejemplos: NIST-AC-2, ISO27002-5.15, ISO27001-A.5.15, OWASP-A01:2025.
    """
    base = h["parent_id"] if h.get("tipo") == "mejora" and h.get("parent_id") else h["id_control"]
    return f"{PREFIJO_NORMA[h['estandar']]}-{base}"


PREFIJO_CITA = {
    "NIST SP 800-53": "NIST",
    "OWASP Top 10": "OWASP",
    "ISO/IEC 27001": "ISO 27001",
    "ISO/IEC 27002": "ISO 27002",
}


def cita(h: dict) -> str:
    """Cómo debe citar el LLM una hoja: [NIST IA-2(1)], [ISO 27002 8.5], [OWASP A01:2025]."""
    return f"[{PREFIJO_CITA[h['estandar']]} {h['id_control']}]"


def cita_de_control(control: str) -> str:
    """NIST-AC-2 -> [NIST AC-2]; ISO27002-5.15 -> [ISO 27002 5.15] (citas de los resúmenes de RAPTOR)."""
    prefijo, ident = control.split("-", 1)
    norma = {v: PREFIJO_CITA[k] for k, v in PREFIJO_NORMA.items()}[prefijo]
    return f"[{norma} {ident}]"


@lru_cache(maxsize=1)
def _codificador():
    import tiktoken
    return tiktoken.get_encoding(config()["fragmentacion"]["tokenizador"])


def contar_tokens(texto: str) -> int:
    """Mismo tokenizador que las hojas (cl100k_base): el presupuesto se mide igual para todo nodo."""
    return len(_codificador().encode(texto))


# ---------------------------------------------------------------- embeddings
def modelo_embeddings() -> str:
    return config()["embeddings"]["modelo"]


def prefijos(modelo: str) -> tuple:
    p = config()["candidatos_embeddings"].get(modelo, {})
    return p.get("prefijo_consulta", ""), p.get("prefijo_documento", "")


def embeber(textos: list, tipo: str, modelo: str = None) -> np.ndarray:
    """Vectores de longitud 1 (con ellos, distancia coseno = 1 − producto punto).

    tipo = "documento" para hojas/resúmenes, "consulta" para preguntas:
    algunos modelos exigen un prefijo distinto para cada caso.
    """
    import ollama

    modelo = modelo or modelo_embeddings()
    p_consulta, p_documento = prefijos(modelo)
    prefijo = p_consulta if tipo == "consulta" else p_documento
    lote = config()["embeddings"]["lote"]
    vectores = []
    for i in range(0, len(textos), lote):
        r = ollama.embed(model=modelo, input=[prefijo + t for t in textos[i:i + lote]])
        vectores.extend(r.embeddings)
    m = np.asarray(vectores, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def digest_ollama(modelo: str) -> str:
    """Huella del archivo del modelo en Ollama: prueba qué versión exacta se usó."""
    import ollama

    nombres = {modelo, f"{modelo}:latest"}
    return next((m.digest for m in ollama.list().models if m.model in nombres), "desconocido")


# ---------------------------------------------------------------- ChromaDB
def cliente_chroma():
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(ruta(config()["almacen"]["ruta"])),
                                     settings=Settings(anonymized_telemetry=False))


def nombre_coleccion(arquitectura: str, modelo: str = None) -> str:
    """Una colección por arquitectura y modelo de embeddings, p. ej. naive-qwen3-embedding-0.6b.

    La de RAPTOR también lleva el LLM, porque los resúmenes del árbol dependen de él:
    raptor-qwen3-embedding-0.6b-llama3.1-8b.
    """
    nombre = f"{arquitectura}-{slug(modelo or modelo_embeddings())}"
    if arquitectura.startswith("raptor"):
        nombre += f"-{slug(config()['llm']['modelo'])}"
    return nombre


def crear_coleccion(nombre: str):
    cliente = cliente_chroma()
    if nombre in [c.name for c in cliente.list_collections()]:
        cliente.delete_collection(nombre)
    # Espacio coseno y búsqueda HNSW con ef alto: con ~5 000 nodos el resultado
    # coincide con una búsqueda exacta (verificado contra fuerza bruta).
    return cliente.create_collection(
        nombre,
        configuration={"hnsw": {"space": "cosine", "ef_construction": 200,
                                "ef_search": 200, "max_neighbors": 32}},
        embedding_function=None,
    )


def abrir_coleccion(nombre: str):
    cliente = cliente_chroma()
    if nombre not in [c.name for c in cliente.list_collections()]:
        raise SystemExit(f"No existe la colección '{nombre}'. Corre antes pipelines\\indexar.py")
    return cliente.get_collection(nombre, embedding_function=None)


def agregar_nodos(col, nodos: list, vectores: np.ndarray) -> None:
    """nodos: dicts con id, texto, n_tokens, nivel, controles (control_base) y citas."""
    for i in range(0, len(nodos), 1000):
        parte = nodos[i:i + 1000]
        col.add(
            ids=[n["id"] for n in parte],
            embeddings=vectores[i:i + 1000],
            documents=[n["texto"] for n in parte],
            metadatas=[{"n_tokens": n["n_tokens"], "nivel": n["nivel"],
                        "controles": "|".join(n["controles"]),
                        "citas": "|".join(n["citas"])} for n in parte],
        )


# ---------------------------------------------------------------- recuperación (común)
def recuperar(col, pregunta: str, modelo_emb: str = None) -> dict:
    """Pregunta -> embedding -> candidatos por similitud -> presupuesto de tokens.

    Regla del árbol colapsado (Sarthi et al., 2024): se recorren los nodos en
    orden de similitud y se detiene en el primero que ya no cabe.
    """
    cfg = config()["recuperacion"]
    t0 = time.perf_counter()
    vector = embeber([pregunta], "consulta", modelo_emb)[0]
    t1 = time.perf_counter()
    r = col.query(query_embeddings=[vector], n_results=min(cfg["candidatos"], col.count()),
                  include=["documents", "metadatas", "distances"])
    t2 = time.perf_counter()

    candidatos = [
        {"id": i, "texto": d, "n_tokens": m["n_tokens"], "nivel": m["nivel"],
         "controles": m["controles"].split("|") if m["controles"] else [],
         "citas": m["citas"].split("|") if m.get("citas") else [],
         "similitud": round(1 - dist, 4)}
        for i, d, m, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])
    ]
    seleccion, usados = [], 0
    for c in candidatos:
        if usados + c["n_tokens"] > cfg["presupuesto_tokens"]:
            break
        seleccion.append(c)
        usados += c["n_tokens"]
    return {
        "seleccion": seleccion,
        "candidatos": candidatos,          # ranking completo (para Recall@k, MRR, nDCG)
        "tokens_contexto": usados,
        "t_embedding_consulta_s": round(t1 - t0, 4),
        "t_busqueda_s": round(t2 - t1, 4),
    }


# ---------------------------------------------------------------- generación (común)
INSTRUCCION_SISTEMA = (
    "Eres un asistente de apoyo a auditorías de seguridad de la información. "
    "Responde siempre en español y únicamente con base en los fragmentos del contexto, "
    "que provienen de normas en inglés (NIST SP 800-53, ISO/IEC 27001, ISO/IEC 27002 "
    "y OWASP Top 10).\n"
    "Instrucciones:\n"
    "1. Redacta una respuesta explicativa y concreta (viñetas o párrafos breves) que "
    "responda directamente la pregunta. No copies los encabezados de los fragmentos.\n"
    "2. Al final de cada afirmación escribe la cita del fragmento que la respalda, "
    "exactamente como aparece en su línea \"Cita:\" (por ejemplo [NIST IA-2(1)] o "
    "[ISO 27002 8.5]). No cites fragmentos que no usaste.\n"
    "3. No agregues requisitos, controles ni recomendaciones que no estén en el contexto.\n"
    "4. Si el contexto no contiene la respuesta, responde exactamente: "
    "\"No se encontró evidencia en el contexto.\""
)


def armar_prompt(pregunta: str, nodos: list) -> list:
    """Plantilla ÚNICA de prompt para ambas arquitecturas (variable controlada).

    Cada nodo va numerado y con su línea "Cita:" para que el LLM sepa cómo citarlo.
    En RAPTOR, un resumen lleva las citas de todos los controles que resume.
    """
    contexto = "\n\n".join(
        f"--- Fragmento {i} | Cita: {' '.join(n['citas'])}\n{n['texto']}"
        for i, n in enumerate(nodos, start=1)
    )
    return [
        {"role": "system", "content": INSTRUCCION_SISTEMA},
        {"role": "user", "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"},
    ]


def generar(mensajes: list, modelo: str = None, max_tokens: int = None) -> dict:
    """LLM local con los parámetros fijos de config.yaml; devuelve respuesta + métricas de equipo.
    max_tokens solo se cambia para los resúmenes de RAPTOR (config: raptor.max_tokens_resumen)."""
    import ollama

    llm = config()["llm"]
    extra = {} if llm.get("razonamiento") is None else {"think": llm["razonamiento"]}
    r = ollama.chat(
        model=modelo or llm["modelo"],
        messages=mensajes,
        options={"temperature": llm["temperatura"], "top_p": llm["top_p"], "seed": llm["semilla"],
                 "num_ctx": llm["num_ctx"], "num_predict": max_tokens or llm["max_tokens"]},
        **extra,
    )
    t_gen = (r.eval_duration or 0) / NS
    return {
        "respuesta": r.message.content,
        "tokens_entrada": r.prompt_eval_count or 0,
        "tokens_salida": r.eval_count or 0,
        "t_primer_token_aprox_s": round(((r.load_duration or 0) + (r.prompt_eval_duration or 0)) / NS, 3),
        "t_generacion_s": round((r.total_duration or 0) / NS, 3),
        "tokens_por_s": round((r.eval_count or 0) / t_gen, 2) if t_gen else None,
        "fin": getattr(r, "done_reason", None),   # "length" = se cortó al llegar al tope de tokens
    }


def calentar(modelo: str = None) -> None:
    """Carga el LLM en la GPU con una consulta corta que se descarta, para que el
    tiempo de carga del modelo no se sume a la primera pregunta medida."""
    generar([{"role": "user", "content": "Responde solo: OK"}], modelo)


# ---------------------------------------------------------------- máquina
def info_maquina(nombre: str) -> dict:
    datos = yaml.safe_load((ruta("maquinas") / f"{nombre}.yaml").read_text(encoding="utf-8"))
    try:
        s = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True).stdout.splitlines()[0]
        gpu, driver, vram = [x.strip() for x in s.split(",")]
        datos.update({"gpu": gpu, "driver_nvidia": driver, "vram_total_mb": int(vram)})
    except Exception:
        datos["gpu"] = "no detectada"
    datos.update({"sistema_operativo": f"{platform.system()} {platform.release()}",
                  "python": platform.python_version()})
    return datos


def vram_usada_mb():
    try:
        s = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True).stdout
        return int(s.splitlines()[0].strip())
    except Exception:
        return None
