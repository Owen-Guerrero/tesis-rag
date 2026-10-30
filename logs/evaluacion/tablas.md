# Resultados del piloto (20 preguntas, lab_p5)

Repeticiones: naive 3, RAPTOR 3. Tablas 1 y 2: media [IC 95 % bootstrap percentil, 2000 iteraciones]. Tabla 3: media ± desviación estándar.

## Tabla 1. Recuperación

| Métrica | Naive RAG | RAPTOR |
|---|---|---|
| Recall@5 | 0.654 [0.512, 0.796] | 0.667 [0.529, 0.796] |
| Recall@10 | 0.696 [0.571, 0.817] | 0.787 [0.679, 0.896] |
| Precision@10 | 0.490 [0.365, 0.615] | 0.585 [0.480, 0.690] |
| MRR | 0.721 [0.583, 0.858] | 0.772 [0.631, 0.902] |
| MAP | 0.443 [0.343, 0.543] | 0.495 [0.412, 0.578] |
| nDCG@10 | 0.529 [0.402, 0.658] | 0.614 [0.510, 0.714] |
| Recall del contexto (lo que recibe el LLM) | 0.900 [0.800, 0.988] | 0.792 [0.662, 0.908] |
| Nodos en el contexto | 24.8 ± 1.1 | 11.8 ± 5.3 |
| Resúmenes en el contexto | 0.0 ± 0.0 | 4.4 ± 1.6 |
| Tokens de contexto | 1957 ± 28 | 1787 ± 229 |

Regla de relevancia: mención explícita para los resúmenes; ISO 27001 A.x = ISO 27002 x.

## Tabla 2. Calidad de la respuesta

| Métrica | Naive RAG | RAPTOR |
|---|---|---|
| Citas en el contexto | 0.992 [0.977, 1.000] | 0.968 [0.915, 1.000] |
| Controles esperados citados | 0.742 [0.592, 0.875] | 0.767 [0.621, 0.887] |
| ROUGE-1 (F1) | 0.421 [0.381, 0.458] | 0.405 [0.370, 0.436] |
| ROUGE-2 (F1) | 0.174 [0.145, 0.204] | 0.160 [0.138, 0.181] |
| ROUGE-L (F1) | 0.234 [0.204, 0.272] | 0.229 [0.201, 0.259] |
| BLEU-4 | 0.099 [0.077, 0.123] | 0.091 [0.075, 0.108] |
| Answer Similarity (similitud semántica con la referencia) | 0.810 [0.777, 0.840] | 0.807 [0.779, 0.833] |
| Respuestas sin ninguna cita | 1 | 1 |
| Respuestas vacías / cortadas | 0 / 1 | 0 / 1 |

## Tabla 3. Desempeño del equipo (media entre repeticiones)

| Métrica | Naive RAG | RAPTOR |
|---|---|---|
| Latencia total por pregunta (s) | 3.09 ± 1.30 | 2.91 ± 1.13 |
| Latencia p95 (s) | 4.91 | 4.71 |
| Tiempo de generación (s) | 2.94 ± 1.20 | 2.80 ± 1.13 |
| Primer token, aprox. (s) | 0.19 ± 0.00 | 0.18 ± 0.02 |
| Tokens/s | 165.0 ± 0.7 | 164.8 ± 0.5 |
| Tokens de entrada al LLM | 2724 ± 33 | 2395 ± 294 |
| Tokens de salida | 441 ± 223 | 432 ± 214 |
| VRAM de la GPU (MB) | 19304 | 19354 |
| Respuestas idénticas en todas las repeticiones | 2/20 | 8/20 |

## Tabla 4. Por tipo de pregunta

| Tipo | n | Recall ctx. naive | Recall ctx. RAPTOR | Esperados citados naive | Esperados citados RAPTOR | ROUGE-L naive | ROUGE-L RAPTOR |
|---|---|---|---|---|---|---|---|
| puntual | 6 | 1.000 | 1.000 | 0.833 | 0.833 | 0.269 | 0.269 |
| multicontrol | 8 | 1.000 | 0.854 | 0.792 | 0.854 | 0.221 | 0.224 |
| transversal | 6 | 0.667 | 0.500 | 0.583 | 0.583 | 0.217 | 0.196 |

## Tabla 5. Sensibilidad a la regla de relevancia (RAPTOR)

| Métrica | Mención explícita (principal) | Herencia (cota superior) |
|---|---|---|
| Recall@10 | 0.787 | 0.787 |
| MRR | 0.772 | 0.838 |
| Recall del contexto | 0.792 | 0.804 |
