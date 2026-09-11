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

> **Actualización (S2, D-07):** al cerrar el criterio de entrada —solo pedidos que alcanzan
> t₁— la población baja a 96.636 pedidos y la prevalencia al **13,42%**. La cifra vigente es
> la de D-07; la de esta sección es la que se midió antes de esa decisión.

---

## D-07 · Población de análisis y momentos de decisión

**Fecha:** 2026-09-10 (S2)

### Contexto

D-06 fijó la unidad de análisis y el target, pero dejó abiertos tres detalles que solo
aparecen al construir la tabla: qué hacer con los pedidos que nunca llegaron a salir, cómo
materializar t₁ como columna, y dónde empieza y acaba la ventana temporal.

Hechos medidos sobre `data/raw`:

| Hecho | Valor |
|---|---|
| Pedidos con `order_delivered_carrier_date` nula | 1.783 (1,79%) |
| — de los cuales `unavailable`, `invoiced`, `processing` | 1.224, el 100% de esos estados |
| — `canceled` sin salir | 550 de 625 |
| Pedidos con fecha de transportista **anterior** a la de aprobación | 1.359 (1,4%) |
| Pedidos con transportista pero sin `order_approved_at` | 14 |
| Volumen fuera del cuerpo del dataset | 329 pedidos en 2016, 20 en 2018-09/10 |

### Decisión

**El criterio de entrada es haber alcanzado t₁, no el `order_status`.** Un pedido entra si
tiene fecha de aprobación *y* fecha de entrega al transportista.

*Razón:* el estado final es una consecuencia posterior al momento de predicción; filtrar por
él sería seleccionar la muestra con información del futuro. "Alcanzó t₁" es una condición
verificable en el propio instante en que el modelo se ejecutaría. Además mantiene dentro los
75 pedidos cancelados que sí llegaron a salir, que son justamente casos de interés.

Los 14 pedidos con transportista pero sin aprobación también salen: retroceder a la fecha de
compra para rellenar t₀ sería inventar el instante en que el modelo se ejecuta.

**t₁ = `max(order_approved_at, order_delivered_carrier_date)`**, con una bandera
`t1_was_inverted` que marca los 1.350 pedidos afectados dentro de la ventana.

*Razón:* la inversión es un artefacto de registro, no una secuencia real —un paquete no se
entrega al transportista antes de que se apruebe el pago—. Descartar esos pedidos sesgaría la
muestra hacia los vendedores con administración ordenada, que plausiblemente son también los
de menos reseñas negativas: exactamente en la dirección del target. El clamp conserva la fila
y la bandera permite comprobar en S3 si ese grupo se comporta distinto.

**Ventana de análisis: 2017-01-01 a 2018-08-31**, sobre `order_purchase_timestamp`.

*Razón:* los 329 pedidos repartidos por 2016 son un piloto con un patrón de negocio distinto,
y 2018-09/10 son 20 pedidos de un corte de exportación a mitad de mes. Dejar las colas
distorsionaría los bloques del split temporal justo en sus bordes, que es donde se leen los
resultados. Constantes en `src/config.py`, no incrustadas en el código.

**El split temporal se cortará sobre `order_purchase_timestamp`, no sobre t₀ ni t₁.**

*Razón:* un eje único e independiente del modelo garantiza que t₀ y t₁ caigan exactamente en
los mismos bloques. Si cada uno se cortara por su propio momento de decisión, las poblaciones
divergirían y la comparación dejaría de medir solo el valor de la señal adicional, que es lo
que D-06 exige. Las fechas concretas de corte se fijan en D-08.

### Consecuencias

- **La prevalencia baja del 14,67% al 13,42%** sobre 96.636 pedidos. No es un error de
  cálculo: exigir que el pedido alcance t₁ elimina los `unavailable` y los `canceled` que
  nunca salieron, que son desproporcionadamente reseñas de 1★. Es el número que acompaña a
  todas las métricas a partir de aquí, por la regla 6 de `CLAUDE.md`.
- **La lectura del umbral cambia, el umbral no.** Sigue siendo 0,25, porque depende solo de
  `c_int / (e · C_neg)`. Pero frente a un 13,42% de base, actuar por encima de 0,25 es actuar
  sobre el segmento de **1,9 veces** el riesgo base, no 1,7.
- El embudo de exclusiones viaja en `spine.attrs["funnel"]` y se imprime con `make features`,
  de modo que la población es auditable sin abrir un notebook.
- `data/interim/spine.parquet` es a partir de ahora la tabla base: una fila por pedido con el
  target, los dos momentos de decisión y `review_creation_date`, esta última guardada para
  que S3 pueda distinguir "pedido anterior" de "etiqueta ya conocida" al construir agregados.
- Se añade `pyarrow` como dependencia. Parquet conserva los tipos entre escritura y lectura;
  con CSV las tres columnas de fecha volverían como texto en cada carga.

### Nota sobre D-06

Esta decisión **supersede la prevalencia reportada en D-06** (14,67%), que se calculó antes de
fijar el criterio de entrada. D-06 se mantiene sin tocar: describe correctamente lo que se
midió en S1 y con qué población.

---

## D-08 · Fechas de corte del split temporal

**Fecha:** 2026-09-10 (S2)

### Contexto

D-07 fijó que el split se corta por `order_purchase_timestamp` en tres bloques. Faltaban las
fechas. La suposición de partida era que se elegirían por tamaño —tantos meses a cada bloque—,
pero el EDA mensual la invalidó.

**La prevalencia no es estacionaria.** Oscila entre el 9,87% y el 22,13% según el mes, con una
desviación de 3,4 puntos sobre una media del 13,42%. Y no deriva de forma suave: va a golpes
de régimen.

La causa está medida:

| Desenlace de la entrega | Prevalencia |
|---|---|
| Entregado a tiempo | 9,2% |
| Entregado tarde | **54,0%** |
| Nunca entregado | **69,4%** |

La correlación mensual entre prevalencia y tasa de retraso es **0,872**. Los dos picos son
episodios logísticos distintos: noviembre de 2017 es Black Friday (el volumen salta de 4.477 a
7.302 pedidos y la tasa de retraso a 14,0%), mientras que febrero y marzo de 2018 tienen
volumen normal y tasas de retraso del 15,7% y 20,7% — un colapso de reparto sin pico de
demanda detrás.

Esto convierte la elección de los cortes en una decisión sobre **qué régimen ve cada bloque**,
no sobre cuántos meses le tocan a cada uno.

### Decisión

| Bloque | Rango | Pedidos | Positivos | Prevalencia |
|---|---|---|---|---|
| Entrenamiento | 2017-01 .. 2018-03 | 64.360 | 9.458 | 14,70% |
| Validación | 2018-04 .. 2018-05 | 13.612 | 1.614 | 11,86% |
| Test | 2018-06 .. 2018-08 | 18.664 | 1.901 | 10,19% |

Constantes `VAL_START` y `TEST_START` en `src/config.py`. La columna `split` se persiste
**dentro de la espina**: si cada script recalcula el reparto, tarde o temprano dos usan cortes
distintos sin que nadie lo note.

*Razón, en dos partes:*

1. **Las dos crisis quedan en entrenamiento.** Un modelo que nunca ha visto una red logística
   saturada va a fallar precisamente cuando más falta hace intervenir.
2. **Validación y test comparten régimen** (11,86% frente a 10,19%). Como el umbral y la
   calibración se fijan sobre validación, solo transfieren al test si la tasa base de ambos
   es parecida. Esta es la restricción que manda sobre el tamaño de los bloques.

### Alternativas descartadas

| Reparto | Validación | Test | Motivo del descarte |
|---|---|---|---|
| **A** 13/3/4 | 18,26% | 10,45% | La crisis entera cae en validación. El umbral se fijaría sobre un régimen que casi dobla al del test y llegaría desplazado. Descartada de plano. |
| **C** 14/3/3 | 15,37% | 10,19% | Validación el doble de grande (3.180 positivos, curva de calibración más estable), pero calibra sobre un régimen ajeno al test y el modelo no ve marzo de 2018. |

Contra C se aceptó una validación más pequeña porque 1.614 positivos bastan para una curva de
calibración de diez tramos (unos 160 por tramo), y la coincidencia de régimen pesa más que la
estabilidad extra de la curva.

### Consecuencias

- **La cifra de referencia del test es 10,19%**, no el 13,42% global. Toda métrica de test se
  lee contra ella; usar la prevalencia global inflaría la lectura de cualquier lift.
- **Hay deriva de tasa base entre entrenamiento (14,70%) y test (10,19%).** El modelo tenderá
  a sobreestimar el riesgo en test. Es exactamente lo que la calibración sobre validación debe
  corregir, y el residuo se reporta en el notebook de negocio en vez de esconderse.
- **El retraso de entrega es el conductor dominante del target.** Pero `late` se conoce a
  posteriori: el trabajo de modelado en t₀ y t₁ es predecir el *riesgo de retraso* con lo
  visible en ese instante — plazo prometido, distancia, historial del vendedor, velocidad de
  traspaso al transportista. Esto orienta el catálogo de features de S2-B4.
- **Hallazgo lateral, útil para features:** la mediana de entrega real menos plazo prometido es
  de **−13 días**. Los plazos de Olist llevan un colchón grande y sistemático, así que la
  variable relevante no es el plazo en bruto sino cuánto colchón queda.
- El diagnóstico usa `order_delivered_customer_date`, prohibida como feature. Vive
  **solo** en `notebooks/01_eda.ipynb`, con el aviso al lado; `src/features.py` no la toca ni
  para diagnosticar. Explicar el pasado y construir un predictor son actividades distintas, y
  mezclarlas en el mismo módulo es como una fuga acaba entrando.
- Figuras en `reports/figures/`: `target_by_month.png` y `prevalence_vs_delay.png`.

---

## D-09 · Catálogo de features y prueba de no fuga

**Fecha:** 2026-09-10 (S2)

### Contexto

Con la población y el split cerrados, quedaba decidir qué se le da al modelo en cada momento.
La exploración se hizo **solo sobre el bloque de entrenamiento**: mirar la relación entre una
variable y el target en validación o test es elegir features con información de los bloques
que luego juzgan el resultado, aunque no intervenga ningún modelo.

### Decisión

**27 features en una sola construcción, con dos listas que declaran el momento.**
`T0_FEATURES` (22) y `T1_FEATURES` (5) en `src/features.py`; el modelo de t₀ entrena con la
primera, el de t₁ con la suma.

*Razón:* dos pipelines separados para t₀ y t₁ acabarían divergiendo —una corrección aplicada a
uno y no al otro— y la comparación dejaría de ser limpia. Una construcción y dos vistas.
Además convierte la pertenencia temporal en un dato verificable: un test exige que toda columna
construida esté declarada exactamente en una de las dos listas.

**Los nulos se quedan nulos.** LightGBM los trata de forma nativa y un pipeline logístico
necesitará su propia imputación. Imputar en `features.py` enterraría una decisión de modelado
dentro del código de datos.

**Las categóricas se dejan como `category`, sin codificar.** La codificación
(one-hot, target encoding, nativa de LightGBM) es una decisión de modelado de S3, y el target
encoding además tiene su propio riesgo de fuga.

### La prueba de no fuga

`tests/test_features.py::test_build_features_ignores_the_actual_delivery_date` construye la
matriz dos veces: la segunda con `order_delivered_customer_date` puesta a un valor absurdo, y
exige que ambas sean **idénticas**.

*Razón:* la fecha de entrega real no existe ni en t₀ ni en t₁ y predice el target casi
perfectamente (54,0% de prevalencia en los pedidos que llegan tarde frente a 9,2% en los que
llegan a tiempo). Un comentario que diga "no usar esta columna" no impide nada. Este test
convierte la afirmación en una propiedad comprobable: si cualquier ruta de código llega a
leerla, las dos matrices dejan de coincidir.

El diagnóstico que **sí** usa esa columna vive solo en los notebooks, con el aviso al lado.

### Consecuencias

- **Ninguna variable es fuerte por sí sola.** El mejor AUC univariante es 0,590
  (`freight_total`); las de t₁ rondan 0,57. El valor tendrá que salir de la combinación. En
  este dataset, un predictor individual espectacular es una sospecha de fuga, no una
  buena noticia.
- **Hay dos canales de riesgo, no uno.** Además del retraso, existe un canal de complejidad del
  pedido: `n_items` predice la reseña (AUC 0,554) y **no** predice el retraso (0,488). Más
  artículos es más superficie para que algo falle, y eso no es logística. Un modelo planteado
  solo como predictor de retraso perdería esa señal.
- **Las relaciones son no lineales y la señal vive en las colas.** `handover_days` es plana en
  nueve deciles y salta al 25,6% en el décimo; `remaining_days_at_t1` solo informa en su decil
  más bajo. Esto da contenido real a la comparación exigida por la regla 5: LightGBM debería
  ganar a la regresión logística, y para que la logística compita habrá que darle versiones
  troceadas de las peores.
- **La geografía y la categoría pesan más de lo esperado**: estado del comprador del 11,7% al
  23,3%, categoría del 8,9% al 23,4%. Ambas son categóricas de cardinalidad alta.
- **`payment_type` no aporta** (13,9%–15,2%). Se conserva en `docs/features.md` como resultado
  negativo documentado en vez de desaparecer sin dejar rastro.
- Se añade `data/processed/features.parquet` y `make features` lo genera junto a la espina.

> **Nota añadida en S3 (D-11).** La expectativa escrita arriba —que para competir habría que
> darle a la regresión logística las variables troceadas— **quedó falsada al medirla**. Trocear
> todas las numéricas en deciles le cuesta a la logística 0,035 de PR-AUC en t₀ y 0,046 en t₁;
> trocear solo las que sí se doblan no la distingue de la lisa. La observación sobre la forma de
> la señal sigue siendo correcta; la receta que se dedujo de ella, no. Ver D-11.

### Pendiente para S3

Ninguna feature describe el **historial del vendedor**, que es el predictor obvio que falta.
Requiere ventana hacia atrás, y con un matiz que la regla 3 de `CLAUDE.md` no cubre del todo:
no basta con usar pedidos *anteriores*, hay que usar **etiquetas ya conocidas** en ese
instante. La reseña llega una mediana de 10 días después de la compra, así que al puntuar un
pedido existen pedidos anteriores cuya reseña todavía no se ha escrito. Por eso la espina
guarda `review_creation_date` desde S2.

---

## D-10 · Historial de vendedor: el doble reloj y el suavizado

**Fecha:** 2026-09-11 (S3)

### Contexto

D-09 dejó el catálogo sin ninguna feature de historial de vendedor, que es el predictor obvio
que faltaba. Construirlo es el punto del proyecto donde es más fácil meter una fuga, y por una
razón que la regla 3 de `CLAUDE.md` no cubre: **la regla habla de pedidos anteriores, y el
problema son las etiquetas**.

Un pedido pasado entrega su información en dos momentos distintos:

| Reloj | Qué marca | Cuándo se puede usar |
|---|---|---|
| Operativo | El vendedor despachó, dentro o fuera de plazo | En el acto |
| De etiqueta | El cliente escribió su reseña | Mediana de 10,2 días después |

Medido sobre la población de análisis:

| Hecho | Valor |
|---|---|
| Pedidos sin ningún pedido anterior del mismo vendedor | 3,0% |
| **Pedidos sin ninguna etiqueta conocida del mismo vendedor** | **5,6%** |
| Pedidos con menos de 5 etiquetas conocidas | 15,2% |
| Etiquetas conocidas en t₀, mediana | 51 |
| Vendedores distintos · pedidos por vendedor | 2.938 · mediana 7, p90 73, máx 1.814 |

Los 2,6 puntos de diferencia entre las dos primeras filas son exactamente el tamaño del
problema: pedidos cuyo vendedor tiene historial operativo pero del que todavía no se sabe
ninguna reseña. Un filtro por fecha de pedido les inventaría una tasa.

### Decisión

**Una tabla de eventos con dos filas por pedido, una por reloj.** Cada pedido genera un evento
de despacho en t₁ y, si tiene reseña, un evento de etiqueta en `review_creation_date`. Se
ordena por tiempo, se acumula con `cumsum` por vendedor y se recoge el estado con `merge_asof`
sobre t₀.

*Razón:* la no fuga sale de la **estructura del join**, no de un filtro que alguien pueda
borrar. Con `allow_exact_matches=False`, el evento de despacho del propio pedido —que cae en
su t₁, nunca anterior a su t₀— queda fuera por construcción. Coste O(n log n) frente al O(n²)
de preguntar por cada pedido qué pedidos anteriores tiene su vendedor.

**Cinco features, todas en `T0_FEATURES`:** `seller_prior_orders`, `seller_tenure_days`,
`seller_prior_reviews`, `seller_neg_rate`, `seller_late_handover_rate`.

`seller_prior_reviews` es feature por derecho propio y no solo un denominador: mide **cuánto
se sabe** del vendedor, y le permite al modelo desconfiar de una tasa hecha con tres reseñas.

`seller_late_handover_rate` no necesita ninguna etiqueta: que un pedido anterior se despachara
tarde se supo el día en que se despachó. Es señal del canal logístico —el dominante según
D-08— disponible en t₀ sin tocar el target.

**As-of t₀ para los dos momentos de decisión.** El historial no se recalcula en t₁.

*Razón:* entre t₀ y t₁ pasan pocos días y el historial extra es marginal, mientras que tener
dos versiones metería en la diferencia t₀ frente a t₁ una componente que no es la señal del
despacho. D-04 fijó que esa comparación mide **solo** el valor de la información adicional.

**La base del historial son todos los pedidos que alcanzaron t₁, no la espina.** Son 97.658
frente a los 96.636 de la población: 1.022 pedidos más, 281 de ellos de 2016.

*Razón:* el vendedor despachó esos pedidos de verdad y sus reseñas existieron de verdad.
Excluirlos confundiría el **criterio de población** con **lo que se sabía en ese instante**, y
dejaría sin historial a todos los vendedores activos en enero de 2017, justo en el borde
inicial del bloque de entrenamiento.

**La etiqueta se considera conocida un día después de su fecha de reseña**, y nunca antes del
despacho del propio pedido.

*Razón, en dos partes:* `review_creation_date` tiene granularidad de día y llega a medianoche,
así que leerla literalmente haría contar una reseña "de las 00:00" para un pedido puntuado ese
mismo día a mediodía. **Cuando la única granularidad disponible es el día, se redondea en la
dirección que quita información, no en la que la regala.** Y hay 317 pedidos con reseña fechada
antes de su propio despacho, 30 de ellos antes de su propio t₀: tomadas al pie de la letra,
esas fechas dejarían entrar el desenlace de un pedido en su propio agregado.

**Las tasas se suavizan hacia la media del mercado, también acumulada hacia atrás**, con la
forma `(k + m·p̄ₜ) / (n + m)`.

Un vendedor con 2 reseñas conocidas y 1 negativa no tiene una tasa del 50%: tiene ruido. `m`
son observaciones ficticias de la media del mercado que el vendedor tiene que contrapesar
antes de que se crea su propia tasa. Con `n = 0` el resultado es exactamente `p̄ₜ`, que es la
respuesta honesta para un vendedor nuevo — no un 0% de reseñas negativas.

**`p̄ₜ` es la tasa del mercado conocida en t₀**, acumulada con el mismo mecanismo. Usar la
prevalencia del bloque de entrenamiento metería el futuro por la puerta de atrás, en el único
sitio donde nadie mira.

**`m` se estima, no se elige.** `shrinkage_from_moments()` aplica el método de los momentos:
la dispersión que se observa entre vendedores es la dispersión real más el ruido binomial de
estimar una tasa con pocos intentos; se resta el ruido y `m = p(1-p) / var_real − 1`.

| Umbral de pedidos por vendedor | `seller_neg_rate` | `seller_late_handover_rate` |
|---|---|---|
| ≥ 5 | m = 22,9 | m = 2,8 |
| ≥ 10 | m = 28,9 | m = 3,5 |
| ≥ 20 | m = 29,6 | m = 4,2 |

De ahí `SHRINKAGE_NEG = 25` y `SHRINKAGE_LATE = 3`. La curva de AUC univariante sobre
entrenamiento, calculada aparte, apunta en la misma dirección para las dos (mejora monótona
con `m` en la de reseñas, empeoramiento monótono en la de despacho), lo que da dos caminos
independientes que coinciden.

### El hallazgo: las dos tasas no se suavizan igual

Que `m` salga **25 en una y 3 en la otra** es el resultado interesante del bloque, no una
inconsistencia. La desviación típica real entre vendedores es de 0,064-0,072 sobre una media
de 0,146 en la tasa de reseñas, y de 0,127-0,151 sobre una media de 0,092 en la de despacho.

Los vendedores **se diferencian mucho más en puntualidad que en reseñas**. Un puñado de envíos
ya deja claro si un vendedor despacha rápido, mientras que un puñado de reseñas con una base
del 14,7% es casi solo azar. El suavizado es un remedio contra denominadores finos: aplicarlo
donde el denominador no es fino solo destruye la dispersión que lleva la señal.

Se comprobó además que el suavizado fuerte **no** convierte la tasa en un proxy de tamaño: la
correlación de Spearman con `seller_prior_reviews` cae de +0,269 sin suavizar a +0,038 con
`m = 25`, y el AUC restringido a vendedores con historial grueso (≥ 50 reseñas) apenas se
mueve. La ganancia viene de tratar mejor a los vendedores finos, que es para lo que está.

### Consecuencias

- **`seller_neg_rate` es la segunda feature más informativa del proyecto** (AUC 0,565, por
  detrás de `freight_total` con 0,590), con deciles monótonos del 10,6% al 21,8% sobre una base
  del 14,70%. `seller_late_handover_rate` da 0,540. Las tres de volumen y antigüedad rondan
  0,52-0,53: **el historial vale por su tasa, no por su tamaño**.
- Sigue sin haber ninguna variable fuerte por sí sola, lo que D-09 ya avisaba. El historial no
  cambia esa conclusión, la refuerza.
- Cinco tests nuevos sobre un escenario de seis pedidos con todos los valores calculados a
  mano, incluido el del doble reloj y una versión de la prueba de envenenamiento de S2 dirigida
  a los agregados: se ponen a 1 estrella todas las reseñas futuras y se exige que el pedido
  puntuado antes no se mueva **y que el puntuado después sí**. Una comparación que pasa también
  cuando el código está roto no prueba nada.
- Dos refactors para no duplicar convenciones ya fijadas: `handover_moment()` (el clamp de t₁
  de D-07) y `dominant_item()` (el vendedor dominante de D-06) pasan a estar escritos una sola
  vez, porque el historial tiene que fechar sus eventos exactamente igual que la espina.
- `seller_tenure_days` queda nula en el primer pedido de cada vendedor (3,6% en entrenamiento).
  Coherente con D-09: los nulos se quedan nulos, y un cero afirmaría que el vendedor abrió hoy.

### Alternativas descartadas

| Alternativa | Motivo |
|---|---|
| Recalcular el historial as-of t₁ | Ensucia la comparación t₀ vs t₁ con algo que no es la señal del despacho, y duplica código y tests |
| Historial de cliente | Solo el 3% de los clientes repite: sería nulo en casi todas las filas. Documentado como descarte razonado en `features.md` |
| Peor vendedor del pedido en vez del dominante | Defendible —una sola tienda mala arruina la reseña—, pero afecta al 1,30% de los pedidos y duplicaría la convención de D-06 |
| Target encoding de vendedor | Es esto mismo hecho mal: sin ventana hacia atrás y sin el reloj de la etiqueta |
| `m` ajustado por AUC | Se midió la curva, pero elegir por ella es ajustar un hiperparámetro a la métrica con la que luego se defiende el resultado. El método de los momentos lo estima sin mirar al target más que a través de la dispersión |

---

## D-11 · Baselines: qué hay que batir, y con cuánto margen

**Fecha:** 2026-09-11 (S3)

### Contexto

La regla 5 de `CLAUDE.md` exige que el modelo complejo bata a un baseline **con un número
explícito**. Para que ese número signifique algo hacen falta tres cosas que es fácil dar por
supuestas: que ambos se ajusten sobre las mismas filas, que se midan con el mismo código, y que
la diferencia se compare contra el ruido del bloque donde se mide.

### Decisión

**Cuatro baselines en `src/train.py`, en los dos momentos.**

| Baseline | Qué es | Para qué está |
|---|---|---|
| `constant` | La tasa base de entrenamiento para todos | El suelo: su PR-AUC **es** la prevalencia, por construcción |
| `single feature` | La variable más fuerte usada en crudo | Lo que escribiría un analista sin modelo: `freight_total` en t₀, `handover_days` en t₁ |
| `logistic` | Regresión logística sobre todo el momento | El baseline serio |
| `logistic binned` | Igual, con las numéricas en deciles | La receta que D-09 dedujo de la forma de la señal |

**Todo se mide en validación. El test no se toca.** La elección de modelo ocurre en validación;
el bloque de test se lee una sola vez al final de S3, con las decisiones ya congeladas. Además
sus tasas base no son intercambiables: 11,86% frente a 10,19%.

**Tres decisiones de preprocesado que no son obvias:**

1. **La nulidad va por una rama aparte** (`MissingIndicator`), no dentro del imputado. Que un
   vendedor no tenga antigüedad significa que es nuevo. Imputar el valor y no dejar constancia
   de que faltaba es fingir que nunca faltó.
2. **Categóricas con `min_frequency=50`.** `product_category` tiene ~70 niveles con cola larga;
   una columna por nivel serían decenas de columnas casi vacías. **Nada de target encoding**:
   es exactamente lo que D-10 hace bien, hecho mal —sin ventana hacia atrás y sin reloj de
   etiqueta—.
3. **Sin `class_weight`**, por la regla 7. El desbalance lo gestiona el umbral, y reponderar
   distorsionaría justo las probabilidades de las que dependerá el cálculo en euros.

**Un gap solo se reporta con su intervalo.** `evaluate.bootstrap_difference()` remuestrea las
filas de validación con reemplazo y puntúa **los dos modelos sobre las mismas filas
remuestreadas**, de modo que la suerte de qué pedidos salieron se cancela y queda la diferencia
entre modelos. Sin esto, "A gana a B por 0,002" es una frase sin contenido.

### Resultados sobre validación (13.612 pedidos, prevalencia 11,86%)

| Modelo | Momento | PR-AUC | recall@10% | lift@10% |
|---|---|---|---|---|
| `constant` | t₀ y t₁ | 0,1186 | 9,9% | 0,98 |
| `single feature` | t₀ | 0,1645 | 17,2% | 1,72 |
| `logistic binned` | t₀ | 0,1992 | 20,0% | 2,00 |
| **`logistic`** | **t₀** | **0,2342** | **24,5%** | **2,45** |
| `single feature` | t₁ | 0,1788 | 16,7% | 1,67 |
| `logistic binned` | t₁ | 0,2113 | 21,9% | 2,19 |
| **`logistic`** | **t₁** | **0,2572** | **27,0%** | **2,70** |

Todas las distancias contra la constante y contra la variable suelta son reales con holgura: la
logística le saca +0,1156 [+0,0989, +0,1347] a la constante en t₀ y +0,1386 en t₁.

### El hallazgo: trocear empeoró las cosas

D-09 observó que las relaciones no son lineales y **dedujo** que habría que darle a la logística
las variables troceadas. Medido, ocurre lo contrario:

| | t₀ | t₁ |
|---|---|---|
| `logistic` − `logistic binned` | **+0,0350** [+0,0230, +0,0480] | **+0,0459** [+0,0328, +0,0590] |

No es la regularización ni el número de tramos: se barrió `C` ∈ {0,01 … 10} y tramos ∈ {5, 10,
20}, y la troceada pierde en toda la rejilla. Trocear **solo** las features que de verdad se
doblan (`promised_days`, `handover_days`, `remaining_days_at_t1` y las tres de historial) da
+0,0018 [−0,0007, +0,0044] en t₀ y +0,0022 [−0,0005, +0,0052] en t₁: **indistinguible de cero**,
así que no se añade nada a `src/train.py` por ello.

*La lectura:* los deciles en one-hot compran la capacidad de doblarse a cambio de tirar el orden.
La mayoría de estas variables son grosso modo monótonas —`seller_neg_rate` sube del 10,6% al
21,8% a lo largo de sus deciles— y un término lineal captura eso con **un** parámetro, mientras
que diez coeficientes independientes tienen que redescubrirlo desde trozos más ruidosos. Lo que
se gana en dos variables con forma de cola se pierde en veinticinco corrientes.

Esto **cambia lo que se le pide a LightGBM**, con un matiz que conviene no perder. Lo medido es
que la curvatura **servida de la forma más burda** —deciles en one-hot, que compran doblarse
tirando el orden— cuesta más de lo que aporta. Eso no demuestra que no haya curvatura: una
representación más suave (splines) podría capturarla sin ese coste, y no se ha probado.

La afirmación defendible es la estrecha: si el árbol gana, **las interacciones son la hipótesis
principal** —un camino hasta una hoja es una conjunción de condiciones, y eso una logística no lo
expresa por muchos tramos que se le den—, pero no queda aislada de la curvatura suave. Y si el
árbol **no** gana con claridad, la conclusión es igual de publicable: la señal es esencialmente
aditiva, la logística basta, y eso es más barato de mantener y más fácil de explicar.

### La comparación t₀ frente a t₁, medida por primera vez

**+0,0230 de PR-AUC [+0,0132, +0,0316]** a favor de t₁, con la logística. Real, y más pequeño de
lo que los AUC univariantes de S2 dejaban intuir. Es la primera cifra del eje central del
proyecto, y la que S4 tendrá que traducir a euros contra una ventana de actuación más corta.

### Hallazgo lateral, candidato a feature de S4

La logística asigna +0,70 a `seller_prior_orders` y −0,64 a `seller_prior_reviews`, que son
casi la misma columna. No es inestabilidad sin más: el modelo está usando su **diferencia**, o
sea los pedidos que el vendedor ya ha despachado y de los que todavía no se sabe la reseña. Un
vendedor en plena punta de volumen, sin juzgar aún. Merece ser una feature explícita y no algo
que el modelo tenga que reconstruir restando.

### Consecuencias

- **La barra para B4 es 0,2342 en t₀ y 0,2572 en t₁.** Un LightGBM que se quede ahí añade
  complejidad a cambio de nada, y lo honesto será decirlo.
- `make train` imprime la tabla completa; `notebooks/03_baseline.ipynb` cuenta la historia y
  deja `reports/figures/baseline_pr.png`.
- Seis tests nuevos en `tests/test_train.py`, incluido el equivalente de la prueba de fuga de S2
  aplicada al modelo: se envenenan todas las columnas de t₁ y se exige que el modelo de t₀
  puntúe idéntico.
- El notebook no usa `.style` de pandas: requiere `jinja2`, que no está en el entorno. Un
  notebook que no se puede ejecutar no documenta nada.

### Alternativas descartadas

| Alternativa | Motivo |
|---|---|
| Target encoding de las categóricas | La versión con fuga de lo que D-10 hace bien |
| `class_weight` o SMOTE | Regla 7. El desbalance es cosa del umbral, y reponderar rompe la calibración de la que depende el cálculo en euros |
| Ajustar `C` | Se barrió: la curva es plana entre 0,01 y 10 (±0,001). Se deja el valor por defecto en vez de fingir que se ha ajustado algo |
| Medir ya en test | Es el único bloque que queda limpio. Se lee una vez, al final, con todo decidido |

---

## D-12 · LightGBM: criterios de aceptación, fijados antes de entrenar

**Fecha:** 2026-09-11 (S3)

> **Esta sección se escribió y se committeó antes de entrenar el primer modelo.** El historial de
> git lo acredita. La razón es incómoda pero simple: si la barra se decide viendo el resultado,
> siempre acaba puesta justo donde el modelo la salta.

### Contexto

D-11 dejó la barra puesta: **0,2342 de PR-AUC en t₀ y 0,2572 en t₁**, sobre validación, con la
regresión logística. Y dejó además una hipótesis: como trocear las variables empeoró las cosas,
lo que un árbol añada debería venir sobre todo de **interacciones** — un camino hasta una hoja
es una conjunción de condiciones, y eso una logística no lo expresa.

### Qué significa "ganar"

Las tres condiciones, y hacen falta las tres:

| # | Condición | Por qué |
|---|---|---|
| 1 | PR-AUC superior a la logística **en validación** | El test sigue cerrado |
| 2 | El intervalo del bootstrap emparejado **excluye el cero** | Con 1.614 positivos, +0,005 cabe dentro del ruido: ya se vio con el troceado quirúrgico |
| 3 | La diferencia **importa operativamente** | Ver abajo |

**Sobre la tercera.** Intervenir sobre el 10% de validación son 1.361 pedidos, y ahí cada punto
porcentual de `recall@10%` equivale a unos **16 pedidos rescatados**. Una mejora de +0,4 puntos
de recall puede ser estadísticamente real y operativamente irrelevante.

### Los tres desenlaces, decididos de antemano

| Desenlace | Qué se hace |
|---|---|
| Gana con holgura (intervalo lejos de cero, hueco ≳ 0,02) | LightGBM pasa a S4. La pregunta siguiente es **de dónde** sale la ganancia |
| Gana por poco (intervalo excluye cero, hueco ~0,005) | **Se queda la logística**, y se explica por qué: calibra mejor de fábrica, se lee en una tabla de coeficientes y no arrastra artefacto ni dependencia de sistema. El árbol se queda documentado con su número |
| No gana | Se reporta tal cual. La señal es aditiva, y eso es un resultado, no un fracaso |

El caso intermedio es el que hay que tener decidido antes, porque es donde la tentación de
justificar el modelo complejo es máxima.

### Cómo se entrena, decidido también antes

**Categóricas y nulos nativos.** LightGBM los trata sin ayuda: ni one-hot, ni imputación. Es
una de las razones de usarlo, y mantiene la matriz igual que la que ve la logística.

**Sin `class_weight` ni `is_unbalance`**, por la regla 7.

**El número de árboles se busca en un corte interno de entrenamiento, y luego se reentrena con
el bloque entero.** Los últimos dos meses de entrenamiento (2018-02 y 2018-03) hacen de
validación interna para el early stopping; con ese número de rondas fijado, el modelo se vuelve
a ajustar sobre los 64.360 pedidos completos.

*Razón, y son dos:* parar directamente sobre el bloque de validación ensuciaría la comparación
—el modelo habría elegido cuándo parar mirando el mismo bloque que luego lo juzga— y usar solo
el corte interno para entrenar dejaría fuera las dos crisis logísticas, que D-08 metió en
entrenamiento a propósito para que el modelo viera una red saturada. Reentrenar con todo
después recupera las dos cosas.

**Hiperparámetros conservadores y fijos, sin búsqueda.** 64.360 filas, 32 features y ninguna
señal individual por encima de 0,59 de AUC: el riesgo aquí es memorizar ruido, no quedarse
corto. Si el modelo gana, se ajustará en S4; si no gana, ninguna rejilla lo va a salvar.

### Resultado: no gana. Se queda la logística

| Modelo | Momento | PR-AUC | recall@10% |
|---|---|---|---|
| **lightgbm** | t₀ | **0,2387** | 0,2441 |
| logistic | t₀ | 0,2342 | **0,2454** |
| **lightgbm** | t₁ | **0,2647** | 0,2546 |
| logistic | t₁ | 0,2572 | **0,2701** |

| Criterio | Veredicto |
|---|---|
| 1 · PR-AUC superior | **sí** — +0,0045 en t₀, +0,0076 en t₁ |
| 2 · El intervalo excluye el cero | **no** — los cuatro lo cruzan |
| 3 · Diferencia operativamente relevante | **no** — el `recall@10%` es **peor**, en 0,1 y 1,6 puntos |

Los intervalos, sobre 400 remuestreos emparejados:

| | PR-AUC | recall@10% |
|---|---|---|
| t₀ | +0,0045 [−0,0062, +0,0145] | −0,0012 [−0,0169, +0,0131] |
| t₁ | +0,0076 [−0,0040, +0,0190] | −0,0155 [−0,0313, +0,0012] |

El árbol gana en la métrica que integra la curva entera y pierde en la que mira la cabeza del
ranking, que es la única parte sobre la que alguien actuaría. **El criterio 1 es el que se suele
citar, y es el único que el árbol pasa.**

### Por qué no gana: sobreajusta desde la ronda 25

El corte interno eligió **69 árboles**, un número lo bastante pequeño como para sospechar del
propio diseño: la validación interna son febrero y marzo de 2018, la crisis logística que D-08
metió a propósito en entrenamiento, y un desajuste de régimen podría estar parando el modelo
antes de tiempo.

La curva lo descarta. Medida sobre validación **como diagnóstico, no para elegir nada**:

| Árboles | 10 | 25 | 50 | **69** | 100 | 200 | 400 | 800 | 1200 |
|---|---|---|---|---|---|---|---|---|---|
| t₀ | 0,2370 | 0,2398 | 0,2379 | **0,2387** | 0,2376 | 0,2339 | 0,2287 | 0,2161 | 0,2075 |
| t₁ | 0,2667 | 0,2699 | 0,2671 | **0,2647** | 0,2641 | 0,2627 | 0,2577 | 0,2458 | 0,2361 |

El máximo está en torno a 25 árboles y a partir de ahí baja. Las 69 rondas no se quedaron
cortas: ya estaban pasadas del pico. Y aun parando en el punto óptimo elegido con trampa —
mirando el bloque que luego juzga — el árbol llega a 0,2398 y 0,2699, márgenes de +0,006 y
+0,013 que tampoco cambiarían el veredicto.

Que un boosting con tasa de aprendizaje 0,05 sature en 25 árboles dice por sí solo lo fina que
es la señal.

### La importancia por ganancia está plana

Ninguna feature pasa del **7,6%** de la ganancia total. El árbol no encuentra una regla dominante
sobre la que construir: se reparte fino entre todo. Es lo mismo que S2 vio desde el otro lado
cuando ningún AUC univariante superó 0,59. La cabeza de la lista —`seller_neg_rate`,
`product_category`, `customer_state`, `distance_km`— coincide con lo que ya leía la logística en
sus coeficientes.

### Lo que esto dice de los datos

Más interesante que qué modelo ganó: **la señal de estas 32 features es esencialmente aditiva.**

Hay dos sondeos independientes en busca de estructura más allá de una suma ponderada, y los dos
salen vacíos. Trocear las variables buscando **curvatura** empeoró el resultado (D-11). Un árbol
libre de construir cualquier conjunción que quiera no encuentra **interacciones** que valgan
0,01 de PR-AUC. No es que el árbol esté mal ajustado: es que no hay gran cosa que ajustar.

### Segunda opinión sobre t₀ frente a t₁

Con el árbol: **+0,0260 [+0,0115, +0,0391]**, frente a +0,0230 [+0,0132, +0,0316] de la
logística. Las dos clases de modelo coinciden, así que la cifra es una propiedad de la
**información disponible**, no del modelo elegido. Es el resultado central del proyecto y ahora
tiene dos medidas independientes detrás.

### Incidencia de entorno: `n_jobs=-1` en WSL2

El primer `make train` tardó **4 min 43 s** en un problema que debería resolverse en segundos.
No era el modelo, era la configuración de hilos. Medido sobre 50 árboles y 64.360 filas:

| `n_jobs` | Tiempo |
|---|---|
| 8 | 0,15 s |
| 12 | 0,12 s |
| 16 | 34,3 s |
| −1 (que resuelve a 16) | 27,9 s |

La causa es el reparto de OpenMP en una CPU **heterogénea** — un Intel Core Ultra 7 255H mezcla
núcleos de rendimiento, de eficiencia y de bajo consumo en el mismo paquete. OpenMP da a cada
hilo la misma porción de trabajo, así que en cada barrera los núcleos rápidos esperan a los
lentos; y con los 16 hilos ocupados no queda ninguno libre para el resto del sistema, con lo que
la espera degenera en *spin waiting*.

`n_jobs = max(1, os.cpu_count() // 2)` deja margen y es portable. `make train` pasa de 4 min 43 s
a **11 s**, con cifras idénticas: LightGBM era lento, no incorrecto.

Emparentado con D-01: es el segundo problema de este proyecto que no está en el código sino
debajo de él.

### Consecuencias

- **La logística pasa a S4**: es la que se calibra, la que recibe el umbral de 0,25, la que se
  traduce a euros y la que se lee una vez sobre test.
- El árbol se queda en el repositorio, en `MODELS`, en la tabla y en este registro **con su
  número**. La frase que queda escrita es "el árbol ganó 0,0045 y no compensa", no "no probamos
  árboles".
- **SHAP deja de ser necesario en la fase 1.** Existía para explicar un modelo de árboles; con
  una logística, la explicación son sus coeficientes. Se mantiene la dependencia por la fase 2.
- No se ajustan hiperparámetros. D-12 lo dijo antes de empezar: si el modelo no gana, ninguna
  rejilla lo salva — y la curva de rondas muestra que el problema es exceso de capacidad, no
  falta de ella.
