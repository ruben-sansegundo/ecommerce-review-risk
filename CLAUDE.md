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

**Target:** `review_score <= 2`, binario. Prevalencia **medida: 14,7%**
(11,5% de 1 estrella + 3,2% de 2 estrellas, sobre reseñas). Ver `docs/decisiones.md` D-06.

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
   es ficción. Curva de calibración + Brier score en el notebook 04.
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
│   ├── interim/           # tablas unidas
│   └── processed/         # matrices listas para modelo
├── docs/
│   ├── problema.md        # pregunta de negocio, target, matriz de costes
│   ├── decisiones.md      # registro de decisiones con su porqué
│   └── features.md        # feature → disponibilidad temporal
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_model.ipynb
│   └── 04_business.ipynb
├── src/                   # paquete instalable: `from src.x import y` sin tocar sys.path
│   ├── config.py          # rutas, SEED, tablas, parámetros de coste
│   ├── data.py            # descarga y carga
│   ├── features.py        # feature engineering, sin fugas
│   ├── train.py
│   └── evaluate.py        # métricas + umbral por coste esperado
├── tests/
│   └── test_features.py
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

**Sesión 1 de 8 — encuadre y esqueleto.** Al día 2026-09-08.

`docs/decisiones.md` es el documento de traspaso entre sesiones: recoge las decisiones
tomadas y su porqué. Léelo junto a este fichero antes de proponer nada.

Hecho en S1:

- [x] Estructura de carpetas y `.gitignore`
- [x] Entorno reproducible (`pyproject.toml` + `uv.lock`), verificado en WSL2
- [x] `docs/decisiones.md` — D-01 a D-05, incluida la matriz de costes cerrada
- [x] `README.md`, `Makefile`, `requirements.txt` derivado
- [ ] `src/config.py` y `src/data.py` — **ficheros creados pero vacíos a propósito**
- [ ] Exploración del esquema: filas por tabla, claves de unión, rango de fechas,
      prevalencia real del target
- [ ] `docs/problema.md`
- [ ] `data/manifest.json`
- [ ] Remoto de git y primer commit

Bloqueado por: el `KAGGLE_API_TOKEN` en `.env`, y la URL del repositorio en GitHub.

**Rubén quiere escribir `config.py` y `data.py` paso a paso para aprenderlos**, no recibirlos
hechos. Explica antes de cada incremento y espera confirmación. Esto va por delante de la
velocidad.

**No** se hace EDA en profundidad, ni features, ni modelo. Eso es S2 y S3.

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