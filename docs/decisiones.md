# Decisiones de diseño

Registro de decisiones tomadas durante el proyecto, con su contexto y sus consecuencias.
Complementa a `CLAUDE.md` (que fija las reglas) explicando **por qué** se tomó cada camino.

Formato: una decisión por sección — contexto, decisión, consecuencias.
Las decisiones no se borran; si una se revierte, se añade una nota al final de su sección.

---

## D-01 · Entorno de desarrollo: WSL2, no Windows nativo

**Fecha:** 2026-09-08 (S1)

### Contexto

El desarrollo empezó en Windows 11 nativo. Al instalar el stack científico, cualquier
`import` de una librería con extensiones compiladas fallaba:

```
ImportError: DLL load failed while importing properties:
Una directiva de Control de aplicaciones bloqueó este archivo.
```

Diagnóstico por eliminación:

| Hipótesis | Prueba | Resultado |
|---|---|---|
| Problema de `uv` | Instalación con `pip` en venv de la stdlib | Falla igual |
| Ruta no confiable | Mismo paquete instalado en `%TEMP%` | Falla igual |
| Falta de firma de código | `.pyd` sin firmar **que sí funcionaba** en el sistema | Descartada |
| Mark-of-the-Web | Sin stream `Zone.Identifier` en los ficheros | Descartada |
| Versión de Python | numpy para 3.14 en venv nuevo | Falla igual |

Causa real: **Smart App Control en modo enforced** bloquea la carga de cualquier binario
nativo sin firmar instalado después de activarse la política. Como toda wheel científica
(numpy, pandas, scipy, scikit-learn, lightgbm) distribuye módulos `.pyd` sin firmar, el
stack completo es inviable en Windows nativo en esta máquina.

Los `numpy` y `pandas` preexistentes del sistema funcionaban solo por ser anteriores a la
política; `scikit-learn`, instalado después, ya estaba roto.

### Decisión

Todo el desarrollo ocurre en **WSL2 con Ubuntu 24.04**. El proyecto vive en el sistema de
ficheros de Linux (`~/projects/ecommerce-review-risk`), **no** en `/mnt/c`.

Alternativa descartada: desactivar Smart App Control. Funciona, pero es **irreversible** —
Microsoft solo permite reactivarlo reinstalando Windows. No se degrada la seguridad del
equipo por una preferencia de tooling cuando existe una salida limpia.

### Consecuencias

- El entorno de desarrollo pasa a ser Linux, igual al de un CI. Menos sorpresas al
  contenerizar en la fase 3.
- `libgomp1` (runtime de OpenMP) es **prerrequisito de sistema** para LightGBM: no viene en
  la imagen mínima de Ubuntu y el fallo es un `OSError` en tiempo de import, no de
  instalación. Documentado en el README.
- El proyecto **no** debe colocarse en `/mnt/c`: el puente 9p entre Windows y Linux penaliza
  seriamente la E/S, y este proyecto lee CSVs de ~120 MB de forma repetida.
- VS Code requiere la extensión Remote-WSL y que las extensiones de workspace se instalen
  del lado Linux.
- Aviso: el PATH de WSL hereda ejecutables de Windows por interop. En esta máquina, `npm`
  resuelve a `/mnt/c/Program Files/nodejs/npm` sin que exista Node en Linux. Conviene
  verificar con `command -v` antes de asumir que una herramienta es nativa.

---

## D-02 · Dependencias: uv + pyproject.toml + lockfile

**Fecha:** 2026-09-08 (S1)

### Contexto

La reproducibilidad es un objetivo declarado del proyecto, no un adorno. Un
`requirements.txt` escrito a mano no la garantiza: fija las dependencias directas y deja
flotar las transitivas. `pip freeze` sí las fija, pero mezcla plataformas y produce un
fichero ilegible.

Factor adicional: el Python del sistema era la versión 3.14, demasiado reciente para parte
del stack en el momento de arrancar. Con `venv` habría hecho falta instalar un intérprete
3.11 a mano.

### Decisión

`uv` con `pyproject.toml` y `uv.lock` versionado. `requires-python = ">=3.11,<3.12"`; `uv`
descarga y gestiona el intérprete, aislado del sistema.

Se versiona además un `requirements.txt` derivado (`uv export --no-hashes`) para quien
prefiera no instalar `uv`.

### Consecuencias

- `uv sync` reconstruye el entorno exacto desde el lockfile.
- Dependencias de desarrollo (`pytest`, `ruff`, `ipykernel`) separadas en un grupo `dev`.
- El lockfile ya ha demostrado su valor: la librería `kaggle` cambió su modelo de
  autenticación entre versiones mayores (ver D-03). Sin lockfile, el repositorio se rompería
  solo con el tiempo.

---

## D-03 · Adquisición de datos: API de Kaggle con alternativa manual

**Fecha:** 2026-09-08 (S1)

### Contexto

El dataset (Olist, ~120 MB, 9 tablas) no se versiona: ocupa demasiado y su licencia es
CC BY-NC-SA 4.0. Hay que poder reproducir su obtención.

La versión actual del paquete `kaggle` (2.x) **rompió compatibilidad** con el modelo de
autenticación anterior: ya no existen `KAGGLE_USERNAME` / `KAGGLE_KEY`, sino un único
`KAGGLE_API_TOKEN`. Muchos tutoriales en circulación describen el modelo antiguo.

### Decisión

1. Descarga programática vía API de Kaggle en `src/data.py`, idempotente (no vuelve a
   descargar si los ficheros ya están).
2. Credenciales en `.env` (ignorado por git), con `.env.example` versionado. Se cargan con
   `python-dotenv` **antes** de importar `kaggle`.
3. Si no hay credenciales: mensaje accionable con la alternativa de descarga manual, no un
   traceback.
4. `data/manifest.json` versionado con el número de filas y el hash de cada tabla.
   `check_raw_data()` valida contra él.

### Consecuencias

- La descarga automática es una **conveniencia, no un requisito**. Quien no quiera lidiar
  con credenciales descarga el zip a mano y el resto del pipeline funciona igual.
- El manifest hace que la adquisición quede **verificada** aunque sea manual: si alguien usa
  una versión distinta del dataset, el proyecto lo detecta en vez de producir números
  silenciosamente distintos a los del README.
- Las reseñas en texto libre (`review_comment_title`, `review_comment_message`) se cargan
  íntegras aunque la fase 1 no las use, para no cerrar la puerta a la fase 2 (NLP).

---

## D-04 · Matriz de costes

**Fecha:** 2026-09-08 (S1)

### Contexto

El proyecto optimiza un umbral por coste esperado y traduce el resultado a euros. Eso exige
una matriz de costes. **El dataset no la contiene**: Olist no publica cuánto le cuesta una
reseña negativa, ni el margen por pedido, ni la eficacia de ninguna intervención.

La respuesta correcta no es evitar el problema, sino la que se da en la práctica: **fijar
supuestos razonables, documentarlos como supuestos y medir la sensibilidad del resultado a
cada uno**. El error grave sería presentar estos números como si fueran datos observados.

> Todo lo que sigue son **supuestos**, no medidas. Ninguno procede del dataset.

### Decisión

#### Moneda

Se trabaja en **euros**, con `BRL_PER_EUR = 3.6` como constante documentada en
`src/config.py`. No se modela la evolución del tipo de cambio.

*Razón:* el destinatario del informe es europeo. La alternativa (mantener BRL, la moneda
real del marketplace) es igual de defendible; se eligió el euro por comunicación, y la
constante queda explícita para que la conversión sea auditable.

#### Parámetros

| Parámetro | Símbolo | Valor | Derivación |
|---|---|---|---|
| Coste de intervención | `c_int` | **3 €** | Cupón de 5 € por una tasa de canje del 60%. La comunicación y el marcado logístico son despreciables a escala. |
| Coste de reseña negativa | `C_neg` | **40 €** | 10 € de margen en riesgo más 30 € de efecto aguas abajo |
| Eficacia de la intervención | `e` | **0,30** | De cada 10 pedidos que acabarían en 1-2 estrellas, la intervención evita 3 |

**Descomposición de `C_neg`.** Un pedido con reseña negativa está pagado: el margen no se
pierde entero. Por eso el componente de margen es parcial (unos 10 €: devoluciones parciales,
coste de atención, reembolsos) y el grueso (unos 30 €) es el daño aguas abajo — menor
probabilidad de recompra de ese cliente y conversión deprimida del vendedor por el rating
visible.

*Contraste con los datos:* el pedido medio en Olist ronda los 160 BRL, unos 45 €. Con un
margen bruto del 25%, son unos 11 € de margen por pedido. Que una reseña negativa cueste
unas 3,6 veces el margen del pedido es agresivo pero razonable en un marketplace donde el
rating es el activo principal.

**Justificación de `e = 0,30`.** Deliberadamente conservador:

- *No más alto:* buena parte de las reseñas negativas vienen de producto defectuoso, artículo
  equivocado o expectativa incumplida. Un cupón y una llamada no arreglan un producto roto.
- *No más bajo:* el retraso de entrega es el factor dominante de las reseñas de 1-2 estrellas
  en este dataset, y ahí la comunicación proactiva y la priorización logística sí actúan.
- Si el caso de negocio se sostiene con un 30%, se sostiene mejor con un 50%. Un supuesto
  optimista que hace cuadrar los números es la vía rápida a perder credibilidad.

Es el parámetro con **menos evidencia** detrás de los tres, y por eso encabeza el análisis
de sensibilidad.

#### La intervención

Una sola acción compuesta, para mantener una matriz limpia: **priorización logística del
pedido, comunicación proactiva y cupón de fidelidad**.

Modelar tres intervenciones con tres eficacias distintas multiplicaría los parámetros
inventados sin añadir rigor.

#### La matriz

Todos los costes son **incrementales respecto a no hacer nada**. Por eso la celda de
"no intervenir con reseña OK" vale cero.

| | Realidad: reseña OK | Realidad: reseña 1-2 estrellas |
|---|---|---|
| **No intervenir** | 0 € | `C_neg` = 40 € |
| **Intervenir** | `c_int` = 3 € | `c_int + (1-e) * C_neg` = 31 € |

#### Umbral óptimo

Intervenir sobre un pedido con probabilidad `p` de reseña negativa compensa cuando el coste
esperado de intervenir es menor que el de no hacerlo:

```
c_int + (1-e) * p * C_neg  <  p * C_neg
  =>  c_int  <  e * p * C_neg
  =>  p  >  c_int / (e * C_neg)
```

Con los valores fijados:

```
p* = 3 / (0,30 * 40) = 0,25
```

Comprobación por el otro lado: intervenir sobre un acierto ahorra 40 - 31 = 9 €; sobre un
falso positivo cuesta 3 €. El equilibrio está en `9p = 3(1-p)`, que da `p* = 0,25`. Coincide.

**Dos consecuencias importantes:**

1. **El umbral depende solo del ratio `c_int / (e * C_neg)`, no de los niveles absolutos.**
   Si se discute si el coste reputacional son 40 € o 25 €, la respuesta es que el umbral se
   mueve únicamente con el ratio. Esto acota mucho el impacto de equivocarse en un supuesto.
2. Un umbral del 25% frente a la prevalencia medida del 14,7% significa actuar sobre el
   segmento de aproximadamente **1,7 veces** el riesgo base. Operativamente sensato: ni
   intervenir sobre todo el mundo, ni un umbral tan alto que no se actúe nunca.

#### Coste plano frente a proporcional al ticket

Se adopta **coste plano** como línea principal, y el proporcional al valor del pedido como
extensión en S6.

*Razón:* el coste plano da una cifra única de ahorro, comunicable, con un umbral global.
La versión proporcional (`C_FN = margen * valor_pedido + reputacional_fijo`) es más realista
y convierte el umbral en **una decisión por pedido**, pero complica el titular.

`src/evaluate.py` se diseña desde el principio para aceptar **costes por fila** (un vector),
no solo escalares. Así la extensión no requiere refactorizar.

#### Misma matriz para t0 y t1

La comparación entre los dos momentos de decisión usa **la misma matriz de costes**.

*Razón, metodológica y no de comodidad:* si varían a la vez la información disponible y la
estructura de costes, la diferencia entre t0 y t1 deja de ser atribuible a nada. Se fija la
economía y se varía solo el conjunto de información, de modo que la comparación mide
exactamente lo que dice medir: **el valor de la señal adicional**.

*Matiz de negocio, que no se oculta:* en t0 el pedido aún no se ha movido y hay más palancas
(reencaminar, avisar al vendedor, cambiar transportista), así que la eficacia real
probablemente sea mayor. En t1 el paquete ya va en camino y quedan sobre todo comunicación y
buena voluntad. Si la eficacia en t0 supera a la de t1, parte de la ventaja de señal de t1
se compensa. Este escenario se evalúa explícitamente en S6.

### Consecuencias

- Los parámetros viven en `src/config.py` como constantes con nombre, nunca incrustados en
  un notebook.
- `docs/problema.md` presenta esta matriz señalando en cada cifra que es un supuesto.
- **S6 incluye análisis de sensibilidad obligatorio:**

| Parámetro | Rango | Motivo del rango |
|---|---|---|
| `C_neg` | 15 - 80 € | Factor 2 en ambos sentidos sobre el ancla |
| `e` | 0,15 - 0,50 | El más incierto: de escéptico a optimista |
| `c_int` | 1 - 8 € | De solo comunicación a cupón generoso |
| Eficacia en t0 frente a t1 | Escenario asimétrico | El matiz de negocio descrito arriba |

Salida esperada: ahorro neto y umbral óptimo en función del ratio, y **la frontera donde el
proyecto deja de compensar**. Poder decir "esto deja de tener sentido por debajo de un 8% de
eficacia" vale más que cualquier PR-AUC.

---

## D-05 · Estructura del repositorio

**Fecha:** 2026-09-08 (S1)

### Decisión

`data/` se subdivide en `raw/` (CSVs originales, inmutables), `interim/` (tablas unidas) y
`processed/` (matrices listas para modelo). Todo ignorado por git salvo los `.gitkeep` y el
manifest.

`src/` es un paquete instalable (`uv sync` lo instala en modo editable), de forma que
`from src.features import ...` funciona igual desde un notebook, un test o la línea de
comandos, sin manipular `sys.path`.

`src/config.py` centraliza rutas, semilla y parámetros de coste.

### Consecuencias

- La separación entre datos crudos y procesados deja explícito qué es inmutable y qué es
  derivado.
- Las figuras de `reports/figures/` **sí** se versionan: se incrustan en el README.
- Los modelos entrenados **no** se versionan en la fase 1; son artefactos reproducibles. Se
  revisará en la fase 3, cuando la demo necesite arrancar sin reentrenar.

---

## D-06 · Definición operativa del target y población

**Fecha:** 2026-09-10 (S1)

### Contexto

`review_score <= 2` define el target, pero entre esa frase y una columna `y` utilizable hay
varias decisiones que el dataset no resuelve solo. Los hechos que siguen están **medidos**
sobre los CSV descargados, no estimados.

| Hecho medido | Valor |
|---|---|
| Pedidos | 99.441 |
| Filas en `order_reviews` | 99.224 |
| `order_id` distintos con reseña | 98.673 |
| Pedidos sin reseña | 768 (0,77%) |
| Pedidos con más de una reseña | 551 |
| Prevalencia `review_score <= 2` (sobre reseñas) | **14,69%** |
| Reparto | 1★ 11,51% · 2★ 3,18% · 3★ 8,24% · 4★ 19,29% · 5★ 57,78% |
| Rango temporal de compra | 2016-09-04 a 2018-10-17 |

### Decisión

**Unidad de análisis: un pedido, una fila.** La reseña es por pedido. Para pedidos con varios
vendedores se toma el de mayor valor de artículo y se añade una feature `n_sellers`.
`order_items` tiene 112.650 filas para 99.441 pedidos, así que el caso multi-artículo es común.

**Reseñas duplicadas: se conserva la primera por `review_creation_date`.** 551 pedidos tienen
más de una. Quedarse con la peor nota, o con la última, sería usar información posterior al
momento en que el modelo se aplicaría: incumple la regla 2 de `CLAUDE.md`. La primera es la
única compatible con el momento de decisión.

**Pedidos sin reseña: se excluyen**, reportando el número. Tratarlos como no negativos
asumiría que la ausencia de reseña no correlaciona con la insatisfacción, supuesto más fuerte
del necesario. Al ser el 0,77%, el sesgo potencial es pequeño; se documenta como limitación.

**Población de comparación: la misma para t₀ y t₁.** Ambos modelos se evalúan sobre los pedidos
que alcanzaron t₁. Evaluar t₀ sobre todos los pedidos y t₁ solo sobre los enviados compararía
sobre poblaciones distintas, y la diferencia dejaría de ser atribuible a la señal. Como
añadido puede reportarse t₀ sobre su población completa, pero el titular es la comparación
pareada.

**Split temporal en tres bloques, no dos:** entrenamiento → validación → test, cortados por
fecha. La calibración y el umbral se fijan sobre validación; el test queda intacto para el
número final. Calibrar sobre el test sería fuga: el modelo habría visto los datos con los que
se le juzga. Las fechas de corte se fijan en S2, tras ver la distribución mensual.

### Consecuencias

- El umbral de 0,25 **no cambia**: depende solo de `c_int / (e · C_neg)`, no de la prevalencia.
  Lo que cambia es su lectura — **1,7 veces** el riesgo base, no el doble.
- **Prevalencia por pedido: 14,67%**, tras deduplicar y excluir los pedidos sin reseña
  (frente al 14,69% sobre filas de reseña bruta). La deduplicación apenas la mueve, lo
  cual es esperable: 551 duplicados sobre 99.224 filas. Reparto por nota: 1★ 11,50% ·
  2★ 3,17% · 3★ 8,23% · 4★ 19,30% · 5★ 57,80%. Reproducible con `make data`.
- **`geolocation` no comparte nombre de columna con ninguna otra tabla.** El prefijo de
  código postal se llama `geolocation_zip_code_prefix`, `customer_zip_code_prefix` y
  `seller_zip_code_prefix` según la tabla. La feature de distancia comprador-vendedor de
  S2 tendrá que unir por columnas con nombres distintos; no sale del grafo de joins
  automático que imprime `make data`.
- **`customer_id` no identifica a un cliente.** Hay 99.441 valores distintos para 99.441
  pedidos: uno por pedido. El identificador de persona es `customer_unique_id`, con 96.096
  valores distintos. Cualquier feature de historial de cliente calculada sobre `customer_id`
  daría siempre "cliente nuevo, cero pedidos previos". Debe advertirse en `features.md`.
- Los 25 meses de ventana temporal dan margen suficiente para los tres bloques.
