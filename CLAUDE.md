# CLAUDE.md — ecommerce-review-risk

Contexto permanente del proyecto. Léelo entero antes de proponer o escribir código.

---

## 1. Qué es esto

Proyecto de machine learning para portfolio, orientado a candidaturas de **Data Scientist**.
No es un producto: es una demostración de criterio técnico. Eso condiciona todas las decisiones
que siguen — el objetivo no es maximizar una métrica, es que cada decisión sea defendible en
una entrevista de una hora.

**Autor:** Rubén — analista de datos con experiencia en Python (Pandas, Spark), SQL, pipelines
ETL en Azure y Power BI. Formación en Física. El proyecto existe para cubrir el hueco entre
"analista" y "científico de datos": modelo entrenado, validado con rigor temporal, interpretado
y traducido a impacto de negocio.

**Presupuesto:** ~24 h en 8 sesiones de 3 h. Ampliable después (ver §8).

---

## 2. El problema de negocio

Marketplace de e-commerce. Una reseña de 1-2 estrellas cuesta el margen del pedido más el daño
reputacional al vendedor. Intervenir antes (aviso proactivo, cupón, priorización logística)
cuesta bastante menos.

**Pregunta:** ¿qué pedidos van a acabar en reseña negativa, con antelación suficiente para actuar?

**Target:** `review_score <= 2`, binario. Prevalencia **13,42%** sobre la población de
análisis: 96.636 pedidos que alcanzan t₁, tienen reseña y caen en la ventana 2017-01 a 2018-08.
Ver `docs/decisiones.md` D-06 y D-07.

**La prevalencia no es estacionaria.** Oscila entre 9,9% y 22,1% según el mes y sigue a la tasa
de retraso de entrega (correlación mensual 0,87). La del bloque de test es **10,19%**, y es esa
—no la global— la que acompaña a cualquier métrica de test. Ver D-08.

**Dos momentos de decisión.** Se modelan los dos y se comparan en la misma tabla:

| Momento | Cuándo | Información disponible | Trade-off |
|---|---|---|---|
| **t₀** | Al aprobarse el pedido | Precio, categoría, vendedor, distancia comprador-vendedor, forma de pago, plazo estimado | Señal más débil, ventana de actuación amplia |
| **t₁** | Al salir el envío | Todo lo de t₀ + transportista, fecha real de envío, desviación sobre el plazo prometido | Señal mucho mejor, ventana corta |

La comparación t₀ vs t₁ es el núcleo intelectual del proyecto. No la elimines ni la simplifiques.

---

## 3. Datos

**Brazilian E-Commerce Public Dataset by Olist** (Kaggle, CC BY-NC-SA 4.0).
~100.000 pedidos, 9 tablas relacionadas, ventana temporal 2016-2018.

Tablas principales: `orders`, `order_items`, `order_payments`, `order_reviews`, `products`,
`sellers`, `customers`, `geolocation`, `product_category_name_translation`.

Reglas:

- Los datos **no se commitean**. `data/` está en `.gitignore`, subdividido en
  `raw/` (CSVs originales, inmutables), `interim/` y `processed/`.
- La descarga se hace con un script reproducible (`src/data.py`), vía API de Kaggle.
  El paquete `kaggle` 2.x usa un único **`KAGGLE_API_TOKEN`** en `.env`; las variables
  `KAGGLE_USERNAME`/`KAGGLE_KEY` del modelo antiguo ya no existen.
- La descarga automática es una conveniencia, no un requisito: si falla, hay alternativa
  manual documentada y el resto del pipeline funciona igual.
- `data/manifest.json` (versionado) guarda filas y hash por tabla. La carga valida contra él,
  para detectar que alguien esté usando una versión distinta del dataset.
- Las reseñas incluyen comentario en texto libre en portugués: **no se tocan en la fase 1**,
  son la materia prima de la fase 2 (NLP).

---

## 4. Reglas metodológicas — no negociables

Estas son la razón de ser del proyecto. Si una propuesta las incumple, no la hagas: dilo.

1. **Split temporal.** Entrenamiento con los meses antiguos, evaluación con los recientes.
   Nunca `train_test_split` aleatorio.
2. **Cero fuga temporal.** Ninguna feature puede usar información posterior al momento de
   predicción. La fecha real de entrega no existe en t₀ ni en t₁.
3. **Agregados con ventana hacia atrás.** Cualquier estadística de vendedor o producto
   (nota media histórica, volumen, tasa de retraso) se calcula solo con pedidos anteriores al
   que se está puntuando. Esta es la fuga más silenciosa que hay.
4. **`docs/features.md` es obligatorio.** Tabla `feature → momento en que está disponible`,
   actualizada cada vez que se añade una feature.
5. **Baseline primero.** Regla trivial, luego regresión logística. El modelo complejo tiene que
   batirlos con un número explícito o no se justifica.
6. **Métricas: PR-AUC y recall@k.** Siempre acompañadas de la prevalencia. Accuracy no aparece
   como resultado, solo como ejemplo de por qué no se usa.
7. **Sin rebalanceo por defecto.** Nada de SMOTE ni `class_weight` en la línea principal. El
   desbalance se gestiona con el umbral optimizado por coste esperado. Si se prueban pesos de
   clase, es como comparación documentada, y con recalibración posterior.
8. **Calibración obligatoria.** Sin probabilidades calibradas, el cálculo de impacto en euros
   es ficción. Curva de calibración + Brier score en el notebook de negocio.
9. **Semillas fijas** en todo lo que tenga aleatoriedad.

---

## 5. Stack y convenciones

- **El desarrollo ocurre en WSL2 / Ubuntu 24.04, no en Windows nativo.** Smart App Control
  bloquea los binarios nativos sin firmar que trae toda wheel científica. Ver `docs/decisiones.md`
  D-01. El proyecto vive en `~/projects/ecommerce-review-risk`, **nunca en `/mnt/c`** (la E/S
  a través del puente 9p es lenta y aquí se leen CSVs de ~120 MB).
- Python 3.11, entorno con `uv` (`pyproject.toml` + `uv.lock`). `requirements.txt` se genera
  del lockfile con `uv export`, no se edita a mano.
- `libgomp1` es prerrequisito **de sistema** para LightGBM (`sudo apt-get install -y libgomp1`).
  Sin él, el fallo aparece al importar, no al instalar.
- **Nunca `n_jobs=-1` en LightGBM aquí.** La CPU es heterogénea (núcleos P/E/LP-E) y pedir los 16
  hilos multiplica el tiempo por 200: 50 árboles pasan de 0,15 s a 34 s. Se usa la mitad de los
  núcleos. Ver `docs/decisiones.md` D-12.
- pandas, scikit-learn, lightgbm, matplotlib, shap, pytest, ruff.
- Antes de dar por hecho que una herramienta es nativa de Linux, comprobar con `command -v`:
  el PATH de WSL hereda ejecutables de Windows por interop.
- **Notebooks cuentan la historia; `src/` es lo que se ejecuta.** Nada de lógica de negocio
  viviendo solo en una celda. Feature engineering va en `src/features.py`, se importa desde el
  notebook.
- Nombres de variables, funciones, ficheros y documentación pública en **inglés**.
  Esta conversación y `CLAUDE.md` en español.
- Commits pequeños, en inglés, con mensaje que explique el porqué. Nada de `update` × 40.

### Estructura

```
ecommerce-review-risk/
├── README.md              # en inglés; lo primero que lee un reclutador
├── CLAUDE.md              # este fichero
├── Makefile               # make data | features | train | eval | test | lint
├── pyproject.toml         # dependencias y config de ruff/pytest
├── uv.lock                # versiones exactas; se commitea
├── requirements.txt       # derivado del lock con `uv export`
├── .env.example           # plantilla; .env con el token está en .gitignore
├── .gitignore
├── data/                  # vacío en git salvo .gitkeep y manifest.json
│   ├── raw/               # CSVs de Kaggle, inmutables
│   ├── interim/           # spine.parquet: una fila por pedido, target y t₀/t₁
│   └── processed/         # features.parquet: matriz lista para modelo
├── docs/
│   ├── problema.md        # pregunta de negocio, target, matriz de costes
│   ├── decisiones.md      # registro de decisiones con su porqué
│   └── features.md        # feature → disponibilidad temporal
├── notebooks/
│   ├── 01_eda.ipynb       # target en el tiempo, cortes del split
│   ├── 02_signal.ipynb    # señal candidata en t₀ y t₁
│   ├── 03_baseline.ipynb
│   ├── 04_model.ipynb
│   └── 05_business.ipynb  # calibración, euros y análisis de sensibilidad
├── src/                   # paquete instalable: `from src.x import y` sin tocar sys.path
│   ├── config.py          # rutas, SEED, tablas, parámetros de coste
│   ├── data.py            # descarga y carga
│   ├── features.py        # feature engineering, sin fugas
│   ├── train.py           # baselines y LightGBM, misma matriz y mismo medidor
│   ├── evaluate.py        # métricas, costes por fila, bootstrap emparejado
│   ├── calibrate.py       # sigmoide ajustada en validación
│   └── business.py        # umbral, euros y la frontera donde deja de compensar
├── tests/
│   ├── conftest.py        # matriz sintética compartida
│   ├── test_features.py   # espina, features y no fuga de historial
│   ├── test_evaluate.py   # métricas, costes y bootstrap
│   ├── test_train.py      # qué columnas ve cada momento, no fuga en el modelo
│   ├── test_calibrate.py  # el test no participa en la calibración
│   └── test_business.py   # los euros, calculados a mano
├── models/                # artefactos entrenados; no se versionan en fase 1
└── reports/figures/       # PNG que se incrustan en el README
```

---

## 6. Cómo quiero que trabajes conmigo

Esto importa tanto como lo técnico. El proyecto solo sirve si **puedo defender cada línea en
una entrevista**.

- **Explica antes de escribir.** Para cualquier decisión no trivial (elección de feature, de
  métrica, de validación), dime primero qué propones y por qué, y espera confirmación.
- **No generes bloques enormes de código de golpe.** Prefiero incrementos que pueda leer,
  entender y criticar.
- **Señala los trade-offs.** Si hay dos caminos razonables, dímelo en vez de elegir en silencio.
- **Discrépame.** Si propongo algo metodológicamente flojo, dilo directamente. Un "sí" cómodo
  aquí me cuesta la entrevista.
- **Pregunta antes de añadir dependencias.**
- **Nada de código que no entienda.** Si algo requiere un concepto que no he mencionado nunca,
  explícamelo en dos frases antes de usarlo.
- Directo, sin relleno. Nivel intermedio: doy por sabidos pandas, SQL y estadística básica.

---

## 7. Estado actual

**Sesión 4 de 8 terminada — FASE 1 COMPLETA.** Al día 2026-09-13.

`docs/decisiones.md` es el documento de traspaso: 16 decisiones con su porqué. Léelo junto a
este fichero y a `docs/features.md` antes de proponer nada.

### Hecho

| Sesión | Alcance | Decisiones |
|---|---|---|
| **S1** | Estructura, entorno `uv`, descarga reproducible con manifiesto, matriz de costes | D-01 a D-06 |
| **S2** | Población (96.636 pedidos), split temporal por régimen logístico, 27 features, prueba de no fuga | D-07 a D-09 |
| **S3** | Historial de vendedor con doble reloj, métricas y bootstrap, baselines, LightGBM contra barra fechada, lectura única del test | D-10 a D-13 |
| **S4** | Calibración, umbral en euros, sensibilidad y frontera de rentabilidad | D-14 a D-16 |

102 tests. Notebooks 01 a 05. `make data | features | train | eval | test | lint`.

### Cifras vigentes

| | |
|---|---|
| Test | 2018-06 .. 2018-08 · 18.664 pedidos · prevalencia **10,19%** |
| Modelo de la fase 1 | **Regresión logística calibrada con sigmoide** |
| PR-AUC en test | 0,1802 (t₀) · **0,2281** (t₁) |
| recall@10% | 21,8% (t₀) · **26,1%** (t₁), frente al 9,9% del azar |
| t₁ − t₀ | +0,0480 [+0,0380, +0,0590] |
| Error de calibración | 7,0% → **1,7%** (t₀) · 6,9% → **2,4%** (t₁) |
| Ahorro en el punto de operación | 23,31 €/mil pedidos (t₀) · **58,51 €/mil** (t₁) |
| Frontera de rentabilidad | e = 0,254 (t₀) · **e = 0,230** (t₁) |

### Lo que la fase 1 concluye

- **La señal es aditiva.** Trocear las variables empeoró (D-11) y LightGBM no encuentra
  interacciones que valgan 0,01 de PR-AUC, sobreajustando desde la ronda 25 (D-12). No propongas
  modelos más complejos sin un argumento nuevo: en test el árbol **pierde** contra la logística.
- **El caso de negocio se sostiene sin holgura.** Se supuso eficacia 0,30 y el equilibrio está en
  0,254. La conclusión del proyecto no es sobre el modelo: **antes de montar esto hay que medir
  la eficacia real de la intervención con un A/B**, porque de ese número cuelga todo.
- **La deriva no se detiene.** El calibrador aprende de un bloque al 11,86% y se aplica a uno al
  10,19%; queda un sesgo residual que no se toca porque recalibrar sobre test sería trampa.
- Una semilla fija **no basta** en un modelo multihilo: hace falta `deterministic=True`.

### Siguiente: S5 — hay que elegir, y la elección es de Rubén

La fase 1 está cerrada y las tres continuaciones son legítimas. Por orden de lo que más añade a
una candidatura:

1. **Fase 2, NLP sobre las reseñas.** Es lo que más diferencia el portfolio y lo que la hoja de
   ruta pone como siguiente. Las reseñas en portugués siguen intactas en el pipeline.
2. **Fase 3, servicio.** FastAPI + contenedor + demo pública. Un enlace que un reclutador puede
   pulsar vale mucho, y el modelo que se serviría es un vector de coeficientes más un calibrador
   de dos parámetros: trivial de desplegar.
3. **Rematar la fase 1.** Queda la feature candidata que encontró la logística —
   `seller_prior_orders` menos `seller_prior_reviews`, el atasco del vendedor sin juzgar — y
   subir la precisión compra margen frente al supuesto de eficacia (del 32,6% al 40% bajaría el
   equilibrio del 23,0% al 18,8%). **Ojo:** el test ya se ha leído dos veces; tocar features
   obliga a decidir explícitamente cuántas lecturas más se permite el proyecto y a decirlo en el
   registro.

### Cómo trabaja Rubén

Bloque a bloque, revisando cada incremento antes del siguiente, para poder defender cada línea
en una entrevista. Explica antes de escribir y espera confirmación. Esto va por delante de la
velocidad. Los notebooks los ejecuta él en VS Code; déjalos formateados con `ruff` antes de que
los abra, porque un buffer abierto de antes sobrescribe el formato al guardar.

---

## 8. Hoja de ruta

| Fase | Alcance | Estado |
|---|---|---|
| 1 | Riesgo tabular: t₀ vs t₁, calibración, impacto en euros, repo reproducible | En curso |
| 2 | NLP sobre las reseñas: motivo de queja, embeddings, extracción de aspectos con LLM vs clasificador clásico (con coste y latencia medidos) | Pendiente |
| 3 | Servicio: FastAPI + contenedor + demo en Streamlit con enlace público | Pendiente |
| 4 | Operación: drift entre meses, reentrenamiento, MLflow | Pendiente |

Al escribir código de la fase 1, no cierres puertas a la fase 2: las reseñas en texto libre
deben seguir accesibles desde el pipeline de datos.