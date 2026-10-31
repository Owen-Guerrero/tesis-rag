# Contraste entre el juez automático (RAGAS) y la anotación humana

Respuestas emparejadas: 40 (misma 1.ª corrida del piloto). Escala humana: correcta = 1, parcial = 0,5, incorrecta = 0. Spearman rho [IC 95 % bootstrap, 2000 iteraciones].

| Métrica RAGAS | vs. media de anotadores | vs. anotador 1 | vs. anotador 2 | n |
|---|---|---|---|---|
| Answer Correctness | 0.408 [0.095, 0.673] | 0.342 [0.012, 0.612] | 0.274 [-0.046, 0.576] | 40 |
| Answer Similarity | 0.033 [-0.312, 0.401] | -0.225 [-0.577, 0.168] | 0.212 [-0.104, 0.527] | 40 |

Faithfulness (RAGAS) vs. fidelidad humana: rho = -0.005 (n = 40).
Nota: el criterio humano de fidelidad resultó demasiado estricto (ver anotacion.md), por lo que este contraste se interpreta con cautela.

## Answer Correctness (RAGAS) según la calificación humana de consenso

| Calificación (ambos anotadores coinciden) | n | Answer Correctness media | Faithfulness media |
|---|---|---|---|
| correcta | 7 | 0.623 | 1.000 |
| parcial | 20 | 0.562 | 0.850 |
| incorrecta | 0 | — | — |
| sin consenso | 13 | | |

Lectura: rho cercano a 1 = el juez ordena las respuestas como los humanos; cercano a 0 = no hay relación. Con n = 40 los intervalos son amplios.