# Auditoría de fidelidad de los resúmenes del árbol RAPTOR

Muestra estratificada de 33 resúmenes de 613 (semilla 42; `pipelines/ver_arbol.py --auditoria 30`),
capas 1–4. Cada resumen se comparó con el texto de sus hijos (única fuente válida).
Revisión asistida por un modelo de lenguaje independiente del que generó los resúmenes
(prompt en el registro del proyecto), con criterio: una afirmación = un control + un
requisito o elemento verificable.

## Resultados

| Capa | Resúmenes | Afirmaciones revisadas | No respaldadas | % |
|---|---|---|---|---|
| 1 | 25 | 332 | 15 | 4,52 % |
| 2 | 4 | 86 | 4 | 4,65 % |
| 3 | 2 | 73 | 1 | 1,37 % |
| 4 | 2 | 84 | 1 | 1,19 % |
| **Total** | **33** | **575** | **21** | **3,65 %** |

Resúmenes con algún identificador de control incorrecto: 6 de 33 (18 %).

## Tipos de error observados

1. **Atribución a un control equivocado** (p. ej., el acuerdo de confidencialidad atribuido
   a PS-3(3) cuando solo corresponde a PS-6(2); CAPTCHA atribuido a AC-7 cuando proviene de
   ISO/IEC 27002 8.5).
2. **Rangos de identificadores que incluyen mejoras inexistentes en los hijos**
   (p. ej., "AU-5(a–e)", "SC-12(1–6)", "PE-3(1)–(8)").
3. **Generalizaciones o consecuencias añadidas** ("can lead to data breaches",
   "thereby affecting network reliability").
4. **Filtración del prompt**: un resumen (raptor-c1-377) reprodujo los identificadores de
   ejemplo del prompt de resumen ("e.g., AC-2, IA-2(1), 5.15, A.8.2, A01:2025"), que no
   están en sus hijos. Es un efecto de la frase añadida al prompt oficial.

## Hallazgos cualitativos

- **Propagación de errores hacia arriba**: un error de la capa 1 (c1-140) se copia en las
  capas 2, 3 y 4; en las capas superiores se cuenta como "respaldado" porque el hijo lo dice.
  Por eso el % de las capas altas subestima el error acumulado respecto de la norma.
- **Capas altas extractivas e incompletas**: varios resúmenes de capas 2–4 copian casi
  literalmente a sus hijos y cubren solo parte de ellos (p. ej., c3-8 cubre 3 de 7 hijos).
  Su baja tasa de error refleja fidelidad, no calidad de síntesis.

## Implicaciones para la Etapa II

- Quitar los identificadores de ejemplo del prompt de resumen (evita la filtración).
- Auditar cada capa contra el texto normativo original, no solo contra sus hijos,
  para medir el error acumulado.
- Controlar la longitud de los resúmenes de capas altas (ver Tabla 1 de tablas.md).
