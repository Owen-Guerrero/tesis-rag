"""
OE6 - Construcción del árbol RAPTOR (Sarthi et al., 2024)
---------------------------------------------------------
Qué hace:
  1. Parte de las MISMAS hojas y los MISMOS vectores del índice de naive RAG
     (los lee de la colección naive-<modelo>): así naive RAG es una ablación
     exacta de RAPTOR (Laitenberger et al., 2025).
  2. Capa por capa:
       a) reduce los vectores con UMAP (global y luego local dentro de cada grupo);
       b) agrupa con mezclas gaussianas (GMM). El número de grupos se elige con el
          BIC y el agrupamiento es suave: un nodo puede estar en varios grupos;
       c) si un grupo pasa de 3500 tokens, lo vuelve a agrupar;
       d) el LLM local resume cada grupo -> esos resúmenes son los nodos de la capa siguiente.
     Se detiene a las 5 capas o cuando quedan 11 nodos o menos (código oficial).
  3. Calcula el embedding de cada resumen y guarda TODO el árbol (hojas + resúmenes)
     en la colección raptor-<embeddings>-<llm>: es el "árbol colapsado".

Cada resumen hereda los controles de todas las hojas que tiene debajo (para la
relevancia y para su línea "Cita:" del prompt).

Si se interrumpe, se puede volver a correr: los resúmenes ya hechos se leen de
una caché (corpus/procesado/arbol_cache_*.jsonl) y no se repiten.

Uso:
  python pipelines\\raptor_arbol.py --maquina pc_owen --limite 300     (prueba rápida)
  python pipelines\\raptor_arbol.py --maquina lab_p5                   (árbol definitivo)

Salidas:
  corpus/procesado/arbol_raptor_*.jsonl    árbol completo (NO se sube: tiene texto ISO)
  logs/arbol_raptor_*_<maquina>.json       costo de construcción y estadísticas (sí se sube)
  logs/arbol_raptor_estructura_*.jsonl     hijos y controles de cada resumen, sin texto (sí se sube)
"""

import argparse
import hashlib
import json
import re
import sys
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipelines.comun import (abrir_coleccion, agregar_nodos, calentar, cargar_hojas, cita,  # noqa: E402
                             cita_de_control, config, contar_tokens, control_base,
                             crear_coleccion, digest_ollama, embeber, generar, info_maquina,
                             modelo_embeddings, nombre_coleccion, ruta, slug, vram_usada_mb)

CFG = config()["raptor"]
warnings.filterwarnings("ignore", message=".*n_jobs value.*")  # aviso normal de UMAP al fijar semilla


# ================================================================ agrupamiento
def reducir(vectores: np.ndarray, vecinos: int, dim: int) -> np.ndarray:
    import umap

    return umap.UMAP(n_neighbors=vecinos, n_components=dim, metric="cosine",
                     random_state=CFG["semilla"]).fit_transform(vectores)


def gmm_suave(x: np.ndarray) -> list:
    """Elige el número de grupos con el BIC y devuelve, por nodo, los grupos a los que pertenece."""
    from sklearn.mixture import GaussianMixture

    candidatos = np.arange(1, min(CFG["max_grupos"], len(x)))
    if len(candidatos) == 0:
        return [np.array([0]) for _ in x]
    bics = [GaussianMixture(n_components=n, random_state=CFG["semilla"]).fit(x).bic(x)
            for n in candidatos]
    n = int(candidatos[int(np.argmin(bics))])
    gm = GaussianMixture(n_components=n, random_state=CFG["semilla"]).fit(x)
    return [np.where(p > CFG["umbral_gmm"])[0] for p in gm.predict_proba(x)]


def agrupar_una_vez(vectores: np.ndarray) -> list:
    """Agrupamiento global + local del código oficial. Devuelve listas de índices."""
    dim = CFG["dimension_reducida"]
    n = len(vectores)
    if n <= dim + 1:
        return [list(range(n))]
    globales = gmm_suave(reducir(vectores, int((n - 1) ** 0.5), min(dim, n - 2)))
    grupos = []
    for g in sorted({int(e) for gs in globales for e in gs}):
        idx = [i for i, gs in enumerate(globales) if g in gs]
        if len(idx) <= dim + 1:
            grupos.append(idx)
            continue
        locales = gmm_suave(reducir(vectores[idx], CFG["vecinos_locales"], dim))
        for l_ in sorted({int(e) for ls in locales for e in ls}):
            grupos.append([idx[i] for i, ls in enumerate(locales) if l_ in ls])
    return [g for g in grupos if g]


def agrupar(nodos: list, vectores: np.ndarray) -> list:
    """Agrupa y vuelve a agrupar todo grupo que pase de max_tokens_grupo (recursivo, como el oficial)."""
    salida = []
    for g in agrupar_una_vez(vectores):
        tokens = sum(nodos[i]["n_tokens"] for i in g)
        if len(g) == 1 or tokens <= CFG["max_tokens_grupo"]:
            salida.append(g)
        elif len(g) == len(nodos):
            # Salvaguarda propia: si no logra dividirlo, se parte en orden sin pasar el límite.
            actual, t = [], 0
            for i in g:
                if actual and t + nodos[i]["n_tokens"] > CFG["max_tokens_grupo"]:
                    salida.append(actual)
                    actual, t = [], 0
                actual.append(i)
                t += nodos[i]["n_tokens"]
            salida.append(actual)
        else:
            sub = agrupar([nodos[i] for i in g], vectores[g])
            salida.extend([[g[i] for i in s] for s in sub])
    return salida


# ================================================================ resúmenes
def clave(hijos: list, llm: str) -> str:
    base = llm + "|" + CFG["prompt_resumen"] + "|" + "|".join(sorted(hijos))
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def resumir(textos: list, llm: str) -> dict:
    contexto = "\n\n".join(" ".join(t.splitlines()) for t in textos)
    mensajes = [{"role": "system", "content": CFG["sistema_resumen"]},
                {"role": "user", "content": CFG["prompt_resumen"].format(contexto=contexto)}]
    return generar(mensajes, llm, max_tokens=CFG["max_tokens_resumen"])


INTRO = re.compile(r"^\s*(here is|here's|this is)[^\n]*?:\s*", re.IGNORECASE)


def limpiar(texto: str) -> str:
    """Quita una frase de introducción del tipo "Here is a summary...:" si el LLM la agrega."""
    return INTRO.sub("", texto.strip(), count=1).strip()


def recortar(texto: str) -> str:
    """Si el resumen se cortó por el tope de tokens, se deja hasta su última oración completa
    (decisión propia: ningún nodo del árbol queda terminando a mitad de una oración)."""
    fin = max(texto.rfind(". "), texto.rfind(".\n"), texto.rfind("."))
    return texto[:fin + 1] if fin > 0 else texto


def citas_de(conteo: Counter) -> list:
    return [cita_de_control(c) for c, _ in conteo.most_common(CFG["max_citas_resumen"])]


# ================================================================ principal
def construir(maquina: str, limite: int = None) -> None:
    llm = config()["llm"]["modelo"]
    modelo_emb = modelo_embeddings()
    sufijo = f"{slug(modelo_emb)}__{slug(llm)}" + (f"__prueba{limite}" if limite else "")
    arch_arbol = ruta("corpus/procesado") / f"arbol_raptor__{sufijo}.jsonl"
    arch_cache = ruta("corpus/procesado") / f"arbol_cache__{slug(llm)}.jsonl"

    # ---- capa 0: las hojas, con los vectores de naive RAG
    hojas = cargar_hojas()[:limite] if limite else cargar_hojas()
    col_naive = abrir_coleccion(nombre_coleccion("naive", modelo_emb))
    guardado = col_naive.get(ids=[h["id_hoja"] for h in hojas], include=["embeddings"])
    por_id = dict(zip(guardado["ids"], guardado["embeddings"]))
    vectores = np.asarray([por_id[h["id_hoja"]] for h in hojas], dtype=np.float32)
    nodos = [{"id": h["id_hoja"], "nivel": 0, "texto": h["texto"], "n_tokens": h["n_tokens"],
              "hijos": [], "conteo": Counter({control_base(h): 1}), "citas": [cita(h)]}
             for h in hojas]
    todos, todos_vec = list(nodos), [vectores]

    cache = {}
    if arch_cache.exists():
        for linea in arch_cache.read_text(encoding="utf-8").splitlines():
            r = json.loads(linea)
            cache[r["clave"]] = r
    print(f"Árbol RAPTOR | hojas: {len(nodos)} | embeddings: {modelo_emb} | LLM: {llm} | "
          f"resúmenes en caché: {len(cache)}")
    calentar(llm)

    t_agrupar = t_resumir = t_embeber = 0.0
    tokens_in = tokens_out = llamadas = cortados = 0
    vram_max = vram_usada_mb() or 0
    capas = []
    inicio = time.perf_counter()

    for capa in range(1, CFG["max_capas"] + 1):
        if len(nodos) <= CFG["dimension_reducida"] + 1:
            break
        t0 = time.perf_counter()
        grupos = agrupar(nodos, vectores)
        t_agrupar += time.perf_counter() - t0
        print(f"\nCapa {capa}: {len(nodos)} nodos -> {len(grupos)} grupos. Resumiendo...")

        nuevos, t0 = [], time.perf_counter()
        with arch_cache.open("a", encoding="utf-8") as fcache:
            for k, g in enumerate(grupos):
                hijos = [nodos[i] for i in g]
                ck = clave([h["id"] for h in hijos], llm)
                guardado = cache.get(ck)
                if guardado and (not guardado.get("cortado") or guardado.get("recortado")):
                    texto = guardado["texto"]
                    cortados += guardado.get("cortado", False)
                else:
                    r = resumir([h["texto"] for h in hijos], llm)
                    texto = limpiar(r["respuesta"])
                    if r.get("fin") == "length":
                        texto = recortar(texto)
                    tokens_in += r["tokens_entrada"]
                    tokens_out += r["tokens_salida"]
                    llamadas += 1
                    cache[ck] = {"clave": ck, "texto": texto, "cortado": r.get("fin") == "length",
                                 "recortado": r.get("fin") == "length"}
                    cortados += cache[ck]["cortado"]
                    fcache.write(json.dumps(cache[ck], ensure_ascii=False) + "\n")
                    fcache.flush()
                conteo = Counter()
                for h in hijos:
                    conteo.update(h["conteo"])
                nuevos.append({"id": f"raptor-c{capa}-{k}", "nivel": capa, "texto": texto,
                               "n_tokens": contar_tokens(texto), "hijos": [h["id"] for h in hijos],
                               "conteo": conteo, "citas": citas_de(conteo)})
                if (k + 1) % 25 == 0 or k + 1 == len(grupos):
                    vram_max = max(vram_max, vram_usada_mb() or 0)
                    seg = time.perf_counter() - t0
                    print(f"  {k + 1}/{len(grupos)} resúmenes | {seg:.0f} s | "
                          f"faltan ~{seg / (k + 1) * (len(grupos) - k - 1):.0f} s")
        t_resumir += time.perf_counter() - t0

        t0 = time.perf_counter()
        vectores = embeber([n["texto"] for n in nuevos], "documento", modelo_emb)
        t_embeber += time.perf_counter() - t0
        capas.append({"capa": capa, "nodos_entrada": len(nodos), "resumenes": len(nuevos),
                      "tokens_media_resumen": round(float(np.mean([n["n_tokens"] for n in nuevos])), 1),
                      "hijos_media": round(float(np.mean([len(n["hijos"]) for n in nuevos])), 1)})
        nodos = nuevos
        todos.extend(nuevos)
        todos_vec.append(vectores)

    t_total = time.perf_counter() - inicio

    # ---- árbol colapsado en ChromaDB (hojas + todos los resúmenes)
    for n in todos:
        n["controles"] = [c for c, _ in n["conteo"].most_common()]
    nombre_col = nombre_coleccion("raptor", modelo_emb) + (f"-prueba{limite}" if limite else "")
    col = crear_coleccion(nombre_col)
    agregar_nodos(col, todos, np.vstack(todos_vec))

    with arch_arbol.open("w", encoding="utf-8") as f:
        for n in todos:
            f.write(json.dumps({k: (dict(v) if k == "conteo" else v) for k, v in n.items()},
                               ensure_ascii=False) + "\n")

    # Estructura sin texto (sí se sube): qué hijos y qué controles tiene cada resumen.
    ruta("logs").mkdir(exist_ok=True)
    arch_estructura = ruta("logs") / f"arbol_raptor_estructura__{sufijo}__{maquina}.jsonl"
    with arch_estructura.open("w", encoding="utf-8") as f:
        for n in todos:
            if n["nivel"] > 0:
                f.write(json.dumps({"id": n["id"], "nivel": n["nivel"], "n_tokens": n["n_tokens"],
                                    "hijos": n["hijos"], "conteo_controles": dict(n["conteo"])},
                                   ensure_ascii=False) + "\n")

    registro = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "maquina": info_maquina(maquina),
        "prueba_con_hojas": limite,
        "coleccion": nombre_col,
        "modelo_embeddings": modelo_emb, "digest_embeddings": digest_ollama(modelo_emb),
        "modelo_llm": llm, "digest_llm": digest_ollama(llm),
        "parametros": CFG,
        "hojas": len(hojas),
        "resumenes": len(todos) - len(hojas),
        "nodos_totales": col.count(),
        "capas": capas,
        "llamadas_llm": llamadas,
        "resumenes_cortados_por_tope": cortados,
        "resumenes_desde_cache": len(todos) - len(hojas) - llamadas,
        "tokens_entrada_llm": tokens_in,
        "tokens_salida_llm": tokens_out,
        "t_agrupamiento_s": round(t_agrupar, 1),
        "t_resumenes_s": round(t_resumir, 1),
        "t_embeddings_resumenes_s": round(t_embeber, 1),
        "t_total_s": round(t_total, 1),
        "vram_max_mb": vram_max,
        "sha256_arbol": hashlib.sha256(arch_arbol.read_bytes()).hexdigest(),
    }
    salida = ruta("logs") / f"arbol_raptor__{sufijo}__{maquina}.json"
    salida.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nÁrbol listo: {len(hojas)} hojas + {registro['resumenes']} resúmenes en "
          f"{len(capas)} capas | {registro['t_total_s']} s | colección {nombre_col}")
    print(f"Resúmenes que llegaron al tope de {CFG['max_tokens_resumen']} tokens: {cortados} "
          f"(recortados hasta su última oración completa)")
    for c in capas:
        print(f"  capa {c['capa']}: {c['resumenes']} resúmenes "
              f"(media {c['tokens_media_resumen']} tokens, {c['hijos_media']} hijos)")
    print(f"Registro: {salida.relative_to(ruta('.'))}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--maquina", required=True, help="pc_owen o lab_p5")
    p.add_argument("--limite", type=int, help="solo las primeras N hojas (prueba rápida)")
    a = p.parse_args()
    construir(a.maquina, a.limite)
