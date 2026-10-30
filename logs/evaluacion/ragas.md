# Métricas RAGAS de generación (juez: qwen2.5:32b; embeddings: qwen3-embedding:0.6b)

Preguntas: 20 por arquitectura. Media [IC 95 % bootstrap, 2000 iteraciones] (n válidas).

| Métrica | Naive RAG | RAPTOR |
|---|---|---|
| Faithfulness | 0.932 [0.856, 0.986] (n=20) | 0.884 [0.769, 0.964] (n=20) |
| Answer Relevancy | 0.728 [0.687, 0.766] (n=20) | 0.742 [0.704, 0.778] (n=20) |
| Answer Correctness | 0.613 [0.555, 0.671] (n=20) | 0.597 [0.531, 0.665] (n=20) |
| Answer Similarity | 0.810 [0.777, 0.840] (n=20) | 0.807 [0.779, 0.833] (n=20) |

Las respuestas sin puntaje (n < total) corresponden a salidas del juez que RAGAS no pudo interpretar.