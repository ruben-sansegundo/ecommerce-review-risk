# El problema

Planteamiento del proyecto: qué se predice, sobre qué población, con qué información y
contra qué criterio económico.

Este documento **presenta**. El registro de decisiones con su justificación está en
[`decisiones.md`](decisiones.md); cuando aquí se afirma algo, allí está el porqué.

Todos los números marcados como *medidos* se obtienen ejecutando `make data`.

---

## 1. La pregunta de negocio

En un marketplace, una reseña de 1-2 estrellas cuesta dinero por dos vías: el margen en
riesgo del propio pedido —devoluciones parciales, atención al cliente, reembolsos— y el daño
aguas abajo, que es el grueso: ese cliente compra menos, y el vendedor convierte peor con una
nota visible baja.

Intervenir antes de que la reseña ocurra —priorizar el envío, avisar al cliente, ofrecer un
cupón— cuesta bastante menos que el daño que evita. Pero solo si se interviene sobre los
pedidos correctos: hacerlo sobre todos convierte el ahorro en gasto.

> **¿Qué pedidos van a acabar en reseña negativa, con antelación suficiente para actuar?**

La segunda mitad de la pregunta importa tanto como la primera. Un modelo perfecto que acierta
cuando el paquete ya está entregado no sirve para nada.

---

## 2. El target

### Definición

```
y = 1  si  review_score <= 2
y = 0  si  review_score >= 3
```

Binario. Se agrupan 1 y 2 estrellas porque ambas señalan un cliente insatisfecho y ambas
disparan la misma intervención. La nota 3 se trata como no negativa: es tibia, no un fallo.

### Reglas de construcción

El paso de "una columna con notas" a "una columna `y` utilizable" exige tres reglas que el
dataset no resuelve solo:

**Unidad de análisis: un pedido, una fila.** La reseña es por pedido, no por artículo. Los
pedidos con varios vendedores toman el de mayor valor de artículo, más una feature
`n_sellers` que deja constancia del caso.

**Reseñas duplicadas: se conserva la primera** por `review_creation_date`. 551 pedidos tienen
más de una. Quedarse con la peor nota, o con la más reciente, usaría información posterior al
momento en que el modelo se aplicaría.

**Pedidos sin reseña: se excluyen.** Son 768, el 0,77%. Tratarlos como no negativos asumiría
que callarse no correlaciona con estar descontento, y es un supuesto más fuerte del necesario.

### Prevalencia medida

Sobre la población de análisis definida en `decisiones.md` D-07: pedidos que alcanzaron
ambos momentos de decisión, con reseña, dentro de la ventana temporal.

| Embudo | Pedidos |
|---|---|
| Pedidos en el dataset | 99.441 |
| Alcanzan t₀ y t₁ | 97.644 |
| Con reseña | 96.913 |
| Dentro de la ventana de análisis | **96.636** |
| **Prevalencia `y = 1`** | **13,42%** |

Reparto completo de notas:

| Nota | Pedidos | % |
|---|---|---|
| 1 ★ | 9.976 | 10,32% |
| 2 ★ | 2.997 | 3,10% |
| 3 ★ | 7.993 | 8,27% |
| 4 ★ | 18.926 | 19,58% |
| 5 ★ | 56.744 | 58,72% |

**Ventana temporal:** 2017-01-01 a 2018-08-31, 20 meses. Las colas del dataset —329 pedidos
repartidos por 2016 y 20 en septiembre y octubre de 2018— quedan fuera: son un piloto y un
corte de exportación, no volumen de negocio.

La prevalencia baja del 14,67% medido en S1 al 13,42% al exigir que el pedido alcance t₁:
los pedidos que nunca salieron son desproporcionadamente reseñas de 1★.

Un 13,42% de prevalencia es el número contra el que hay que leer cualquier métrica de este
proyecto. Un modelo que prediga "ninguna reseña será negativa", siempre, acierta el 86,58% de
las veces y no sirve absolutamente para nada. Por eso la accuracy no aparece como resultado.

---

## 3. Los dos momentos de decisión

El mismo pedido se puntúa en dos instantes donde el negocio podría actuar de verdad.
Compararlos es el núcleo del proyecto.

| | **t₀** — pedido aprobado | **t₁** — envío despachado |
|---|---|---|
| **Cuándo** | Al aprobarse el pago (mediana: 20 min tras la compra) | Al entregarse el paquete al transportista |
| **Se sabe** | Precio, categoría, vendedor, distancia comprador-vendedor, forma de pago, plazo prometido | Todo lo de t₀ + transportista, fecha real de despacho, desviación sobre el plazo |
| **Ventana de actuación** | Amplia: aún se puede reencaminar, avisar al vendedor, cambiar de transportista | Estrecha: el paquete ya se mueve, quedan comunicación y buena voluntad |
| **Señal esperada** | Más débil | Bastante mejor |

La tensión es evidente: en t₁ se sabe más, pero queda menos margen para hacer algo con ello.
El proyecto no supone cuál gana — lo mide.

**Ambos modelos se evalúan sobre la misma población**: los pedidos que llegaron a t₁.
Evaluar t₀ sobre todos los pedidos y t₁ solo sobre los despachados compararía sobre conjuntos
distintos, y la diferencia dejaría de ser atribuible a la información.

**Ninguna feature puede usar la fecha real de entrega.** No existe ni en t₀ ni en t₁, y es la
fuga más tentadora del dataset porque predice el target casi perfectamente.

---

## 4. La matriz de costes

> ⚠️ **Todo lo de esta sección son supuestos, no medidas.** El dataset no contiene costes:
> Olist no publica lo que le cuesta una reseña negativa, ni el margen por pedido, ni la
> eficacia de ninguna intervención. Están fijados de forma razonada, documentados como
> supuestos, y sometidos a análisis de sensibilidad. Presentarlos como datos observados sería
> el error grave.

### La intervención

Una sola acción compuesta, para mantener la decisión binaria: **priorización logística del
pedido, comunicación proactiva y cupón de fidelidad**.

### Parámetros

| Parámetro | Valor | De dónde sale |
|---|---|---|
| Coste de intervenir `c_int` | **3 €** | Cupón de 5 € con una tasa de canje del 60% |
| Coste de reseña negativa `C_neg` | **40 €** | 10 € de margen en riesgo + 30 € de efecto aguas abajo |
| Eficacia `e` | **0,30** | De cada 10 reseñas negativas evitables, la intervención evita 3 |

Moneda: euros, con un tipo fijo `BRL_PER_EUR = 3.6`. No se modela la evolución del cambio.

El `0,30` es deliberadamente conservador y es el parámetro con menos evidencia detrás: un
cupón no arregla un producto defectuoso, aunque sí mitiga un retraso. Si el caso de negocio
se sostiene al 30%, se sostiene mejor al 50%.

### La matriz

Costes **incrementales respecto a no hacer nada**, que es el statu quo:

| | Reseña OK | Reseña 1-2 ★ |
|---|---|---|
| **No intervenir** | 0 € | 40 € |
| **Intervenir** | 3 € | 31 € |

La celda de intervenir sobre un pedido que iba mal es `c_int + (1 - e) · C_neg`: pagas el
cupón y sigues comiéndote el 70% de las reseñas que no lograste evitar.

### El umbral

Intervenir compensa cuando el coste esperado de hacerlo es menor que el de no hacerlo:

```
c_int + (1-e)·p·C_neg  <  p·C_neg
  =>  p  >  c_int / (e · C_neg)
```

```
p* = 3 / (0,30 × 40) = 0,25
```

Dos lecturas importantes:

1. **El umbral solo depende del ratio `c_int / (e · C_neg)`**, no de los niveles absolutos.
   Equivocarse en si el daño reputacional son 40 € o 25 € mueve el umbral mucho menos de lo
   que parece.
2. **Un umbral de 0,25 frente a una prevalencia de 0,147** significa actuar sobre el segmento
   que carga con ~1,7 veces el riesgo base. Ni intervenir sobre todo el mundo ni no
   intervenir nunca: un punto operativamente razonable.

El umbral no se escribe a mano en ningún sitio. Se calcula en `src/config.py` a partir de los
tres parámetros, de modo que no puede quedar desincronizado cuando se varíen.

---

## 5. Cómo se valida

**Split temporal en tres bloques, nunca aleatorio.** Entrenamiento con los meses antiguos,
validación con los intermedios, test con los recientes. Un split aleatorio dejaría que el
modelo aprendiera de pedidos futuros para predecir pasados, que es lo contrario de la
situación real.

**El bloque de validación existe para calibrar.** La calibración y la elección del umbral se
hacen ahí, dejando el test intacto para el número final. Calibrar sobre el test sería fuga: el
modelo habría visto los datos con los que se le juzga.

**Agregados solo hacia atrás.** Cualquier estadística de vendedor o producto —nota media
histórica, volumen, tasa de retraso— se calcula únicamente con pedidos anteriores al que se
está puntuando. Es la fuga más silenciosa que existe en este tipo de problema.

**Baselines primero.** Una regla trivial, después una regresión logística. El modelo complejo
tiene que batirlos con un número explícito o no se justifica su coste.

**Métricas: PR-AUC y recall@k, siempre junto a la prevalencia.** Un clasificador aleatorio
tiene un PR-AUC igual a la prevalencia, así que un 0,30 no significa nada hasta saber que la
base es 0,147.

**Calibración obligatoria.** Curva de calibración y Brier score. Sin probabilidades
calibradas, comparar contra un umbral de 0,25 y traducirlo a euros es ficción: el modelo
podría estar ordenando bien y mintiendo en los valores absolutos.

---

## 6. Limitaciones conocidas

- **La matriz de costes es inventada**, con supuestos razonados. El análisis de sensibilidad
  barrerá `C_neg` entre 15 y 80 €, `e` entre 0,15 y 0,50, y `c_int` entre 1 y 8 €, reportando
  la frontera donde el proyecto deja de compensar.
- **La eficacia de la intervención no es medible con estos datos.** Requeriría un test A/B.
  Es el número que más condiciona el resultado, porque está en el denominador del umbral.
- **La prevalencia puede variar mes a mes.** Entrenar con meses antiguos y evaluar con
  recientes implica evaluar sobre una población que puede no ser idéntica. Se medirá y se
  reportará, porque parte de la degradación podría venir de ahí y no del modelo.
- **Los pedidos sin reseña quedan fuera.** Son el 0,77%, así que el sesgo potencial es
  pequeño, pero el supuesto de que su ausencia es independiente del target no se comprueba.
- **Un solo marketplace, un solo país, 2016-2018.** Nada garantiza que los patrones
  encontrados se trasladen a otro contexto.
