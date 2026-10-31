# Naive RAG vs. RAPTOR para auditorías sobre normas de ciberseguridad

Código, datos de evaluación y resultados del trabajo de investigación
**"Comparación de arquitecturas RAG estándar y jerárquica (naive RAG vs. RAPTOR) para
auditorías sobre normas de ciberseguridad"**. Seminario de Investigación I, Universidad de Lima (2026).

- Autores: Franklin Owen Guerrero Angaspilco, Irwin Scott Crisóstomo Porras
- Asesor: José Valdivia Caballero

## De qué trata

Un auditor hace preguntas **en español** sobre control de acceso y gestión de identidades.
El sistema busca la respuesta en normas escritas **en inglés** y responde citando el control
de donde sale cada afirmación. Todo corre en local: LLM y embeddings con Ollama, sin servicios en la nube.

Se comparan dos arquitecturas que usan **los mismos fragmentos, el mismo modelo, el mismo prompt
y el mismo presupuesto de contexto (2000 tokens)**. Lo único que cambia es cómo se recupera la evidencia:

| | Naive RAG (línea base) | RAPTOR (Sarthi et al., 2024) |
|---|---|---|
| Índice | Fragmentos ("hojas") de ≤ 100 tokens | Hojas + resúmenes jerárquicos (árbol de 4 capas) |
| Recuperación | Similitud coseno sobre las hojas | Árbol colapsado: hojas y resúmenes compiten por similitud |

### Corpus

| Norma | Fuente | ¿Incluida en este repositorio? |
|---|---|---|
| NIST SP 800-53 Rev. 5 | Catálogo OSCAL oficial (dominio público) | Sí |
| OWASP Top 10:2025 | Repositorio oficial OWASP (CC BY-SA 4.0) | Sí |
| ISO/IEC 27001:2022 y 27002:2022 | PDF con licencia | **No.** Se requiere una copia con licencia propia (ver más abajo) |

## Configuración final

| Componente | Valor |
|---|---|
| LLM generador | `gpt-oss:20b` (razonamiento *low*, temperatura 0, semilla 42, `num_ctx` 8192) |
| Embeddings | `qwen3-embedding:0.6b` |
| Base vectorial | ChromaDB 1.5.9, distancia coseno |
| RAPTOR | UMAP + GMM (número de grupos por BIC), resúmenes ≤ 800 tokens |
| Juez RAGAS | `qwen2.5:32b` (distinto del generador) |
| Equipo de las mediciones | NVIDIA RTX 6000 Ada 48 GB, Intel Xeon w7-2495X, 255 GB RAM (`logs/entorno_lab.txt`) |

Toda la configuración está en `config.yaml`.

## Estructura del repositorio

```
config.yaml            parámetros del experimento (modelos, presupuesto, RAPTOR)
requirements.txt       dependencias con versiones fijas
maquinas/              descripción de los equipos (desarrollo y laboratorio)
corpus/
  extraer_nist.py      NIST SP 800-53 (OSCAL JSON) -> registros por control
  extraer_owasp.py     OWASP Top 10:2025 (Markdown) -> registros por sección
  extraer_iso.py       ISO 27001/27002 (PDF) -> registros por control (requiere los PDF)
  fragmentar.py        registros -> hojas de ≤ 100 tokens (iguales para las dos arquitecturas)
  procesado/           corpus procesado y manifiestos (sin texto ISO)
pipelines/
  comun.py             funciones compartidas (embeddings, recuperación, prompt, generación)
  indexar.py           índice de naive RAG
  naive_rag.py         naive RAG
  raptor_arbol.py      construcción del árbol RAPTOR
  raptor_rag.py        RAPTOR (árbol colapsado)
  ver_arbol.py         inspección y muestra para auditar el árbol
  demo.py              demostración: una pregunta, las dos arquitecturas lado a lado
eval/
  preguntas_piloto.jsonl  conjunto de evaluación (20 preguntas con respuesta de referencia)
  metricas.py          métricas de recuperación y de citas
  evaluar.py           todas las métricas clásicas + IC bootstrap -> logs/evaluacion/
  ragas_eval.py        métricas RAGAS de generación
  anotacion.py         anotación humana a ciegas y kappa de Cohen
  comparar_llm.py      selección del LLM (OE2)
resultados/            salidas de cada corrida (ids recuperados, respuestas, tiempos)
logs/evaluacion/       tablas finales, RAGAS, auditoría del árbol y anotación
```

## Dataset de evaluación

`eval/preguntas_piloto.jsonl`: 20 preguntas en español sobre control de acceso e identidades,
de tres tipos:

- **Puntual** (6): un solo control.
- **Multicontrol** (8): varios controles de dos o más normas.
- **Transversal** (6): un caso de auditoría que requiere integrar varias normas.

Cada pregunta trae su tipo, sus controles esperados (ISO 27001 A.x emparejado con ISO 27002 x) y una
respuesta de referencia.

## Cómo reproducirlo

Requisitos: Python 3.13, Git y [Ollama](https://ollama.com).

```cmd
git clone <url-del-repositorio>
cd tesis-rag
python -m pip install -r requirements.txt
ollama pull gpt-oss:20b
ollama pull qwen3-embedding:0.6b
```

1. **Corpus** (NIST y OWASP se descargan de sus fuentes oficiales):
   ```cmd
   python corpus\extraer_nist.py
   python corpus\extraer_owasp.py
   python corpus\extraer_iso.py        (solo si se tienen los PDF en corpus\fuentes\iso\)
   python corpus\fragmentar.py
   ```
2. **Índices**:
   ```cmd
   python pipelines\indexar.py --maquina lab_p5
   python pipelines\raptor_arbol.py --maquina lab_p5
   ```
3. **Experimento** (3 repeticiones de cada arquitectura):
   ```cmd
   python pipelines\naive_rag.py --maquina lab_p5 --preguntas eval\preguntas_piloto.jsonl
   python pipelines\raptor_rag.py --maquina lab_p5 --preguntas eval\preguntas_piloto.jsonl
   ```
4. **Evaluación**:
   ```cmd
   python eval\evaluar.py
   python -m pip install ragas==0.4.3 "langchain<1" "langchain-core<1" "langchain-community<0.4" "langchain-openai<1" "langchain-ollama<1"
   python eval\ragas_eval.py --juez qwen2.5:32b
   ```
5. **Demostración**:
   ```cmd
   python pipelines\demo.py "¿Qué exige NIST para bloquear cuentas tras intentos fallidos?"
   ```

Sin los PDF de ISO el sistema funciona igual con NIST y OWASP, pero los resultados no serán
idénticos a los reportados.

## Resultados principales

Fuente: `logs/evaluacion/tablas.md` y `logs/evaluacion/ragas.md`. Valores: media de 20 preguntas
[IC 95 % bootstrap, 2000 iteraciones].

| Métrica | Naive RAG | RAPTOR |
|---|---|---|
| Recall@10 | 0.696 [0.571, 0.817] | 0.787 [0.679, 0.896] |
| MAP | 0.443 [0.343, 0.543] | 0.495 [0.412, 0.578] |
| nDCG@10 | 0.529 [0.402, 0.658] | 0.614 [0.510, 0.714] |
| Recall del contexto entregado al LLM | 0.900 [0.800, 0.988] | 0.792 [0.662, 0.908] |
| RAGAS Faithfulness | 0.932 [0.856, 0.986] | 0.884 [0.769, 0.964] |
| RAGAS Answer Correctness | 0.613 [0.555, 0.671] | 0.597 [0.531, 0.665] |
| Latencia media por pregunta | 3.09 s | 2.91 s |

RAPTOR ordena mejor la evidencia, pero sus resúmenes ocupan buena parte del presupuesto de contexto
y las respuestas finales de ambas arquitecturas resultan equivalentes: los intervalos de confianza
se superponen. La auditoría de 33 resúmenes del árbol encontró un 3,65 % de afirmaciones no
respaldadas por sus hijos (`logs/evaluacion/auditoria_arbol.md`).

## Sobre el texto de ISO/IEC

ISO/IEC 27001 y 27002 son normas con licencia. Este repositorio **no contiene** su texto: ni los PDF,
ni el texto extraído, ni las hojas, ni el árbol RAPTOR, ni la base vectorial (ver `.gitignore`).
Las respuestas generadas guardadas en `resultados/` y `eval/` pueden incluir citas breves de los controles,
como lo haría un informe de auditoría.

## Uso de IA

Durante el desarrollo se usaron asistentes de IA como apoyo para programar y redactar. La
declaración completa de su uso está en el artículo.
