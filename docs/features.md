# Features y disponibilidad temporal

Tabla obligatoria por la regla 4 de `CLAUDE.md`: **cada feature declara en qué momento está
disponible**. Se actualiza en el mismo commit que añade la feature, no después.

> **Estado:** 27 features, construidas en S2 (`src/features.py`). Faltan los agregados de
> historial de vendedor y de cliente, que necesitan ventana hacia atrás y se abordan en S3.

**Momento** es uno de:

- **t₀** — disponible al aprobarse el pedido
- **t₁** — disponible al despacharse el envío (**incluye todo lo de t₀**)

La pertenencia no es una convención escrita solo aquí: vive en las listas `T0_FEATURES` y
`T1_FEATURES` de `src/features.py`, y `tests/test_features.py` rechaza cualquier columna
construida que no esté declarada en una de las dos.

**AUC** es el AUC univariante medido **solo sobre el bloque de entrenamiento** (64.360 pedidos,
prevalencia 14,70%). Se lee como la probabilidad de que un pedido con reseña negativa puntúe
más alto en esa variable que uno sin ella; 0,500 es no saber nada. Por debajo de 0,5 la
relación es inversa (más alto = mejor).

---

## t₀ · disponible al aprobarse el pedido

| Feature | AUC | Nulos | Origen | Notas |
|---|---|---|---|---|
| `freight_total` | 0,590 | 0% | `order_items` | La más informativa de todas. Es un compuesto de peso, volumen y distancia: el marketplace ya ha estimado la dificultad logística y la ha puesto en el precio del envío |
| `n_items` | 0,554 | 0% | `order_items` | Canal ajeno al retraso: predice la reseña y **no** predice el retraso (AUC 0,488 contra la etiqueta de retraso) |
| `payment_value` | 0,550 | 0% | `order_payments` | Muy correlacionada con `price_total`; ambas se mantienen y se revisará en S3 |
| `distance_km` | 0,544 | 0,52% | `customers`, `sellers`, `geolocation` | Distancia en línea recta comprador-vendedor. Nula cuando un prefijo postal no aparece en `geolocation` |
| `price_total` | 0,540 | 0% | `order_items` | |
| `weight_g` | 0,530 | 0,02% | `order_items`, `products` | Suma con `min_count=1`: un pedido sin ningún peso conocido queda nulo, no a cero |
| `promised_days` | 0,530 | 0% | `orders` | Plazo prometido, en días desde la compra. Satura a partir del tercer decil |
| `n_products` | 0,526 | 0% | `order_items` | |
| `volume_cm3` | 0,526 | 0,02% | `products` | |
| `installments` | 0,523 | 0% | `order_payments` | |
| `approval_latency_h` | 0,517 | 0% | `orders` | Horas entre la compra y la aprobación del pago |
| `freight_ratio` | 0,516 | 0% | `order_items` | `freight_total / price_total`, con el divisor cero convertido a nulo |
| `n_sellers` | 0,515 | 0% | `order_items` | Deja constancia del pedido multivendedor (D-06) |
| `max_item_price` | 0,514 | 0% | `order_items` | |
| `seller_margin_days` | 0,496 | 0% | `order_items`, `orders` | Días entre t₀ y el `shipping_limit_date` más temprano: el margen que se le concede al vendedor |
| `description_len` | 0,491 | 1,80% | `products` | |
| `photos` | 0,482 | 1,80% | `products` | Inversa: más fotos, menos reseñas negativas. Proxy de calidad del anuncio |
| `same_state` | 0,458 | 0% | `customers`, `sellers` | Inversa y de las más fuertes leída al revés: mismo estado 11,6% frente a 16,3% |
| `payment_type` | categórica | 0% | `order_payments` | **Resultado negativo documentado:** 4 niveles entre 13,9% y 15,2%. No aporta. Se conserva en el catálogo en vez de desaparecer sin dejar rastro |
| `product_category` | categórica | 1,83% | `products` | 21 niveles con 500+ pedidos, del 8,9% al 23,4%. Categoría del artículo más caro |
| `customer_state` | categórica | 0% | `customers` | 15 niveles, del 11,7% (SP) al 23,3% (MA). Más señal que casi toda la lista numérica |
| `seller_state` | categórica | 0% | `sellers` | 7 niveles, del 9,0% al 15,4% |

## t₁ · lo que añade la salida del envío

| Feature | AUC | Nulos | Origen | Notas |
|---|---|---|---|---|
| `handover_days` | 0,576 | 0% | espina | Días de t₀ a t₁. Plana en nueve deciles y salta al 25,6% en el décimo |
| `handover_vs_limit_days` | 0,570 | 0% | espina, `order_items` | Días de retraso del vendedor sobre su propio plazo. La de mayor AUC contra el retraso real (0,635) |
| `elapsed_share` | 0,564 | 0% | espina, `orders` | Fracción del plazo prometido ya consumida al salir el envío |
| `handover_late` | 0,543 | 0% | derivada | Booleana. Prevalencia 26,0% frente a 13,5% |
| `remaining_days_at_t1` | 0,488 | 0% | espina, `orders` | Días de colchón que quedan. Solo informa su decil más bajo: 17,2% frente a ~14,5% en el resto |

---

## Reglas al añadir una feature

**Nada posterior al momento de predicción.** La comprobación es preguntarse: *si estuviera
puntuando este pedido en t₀, ¿existiría ya este dato?* La fecha real de entrega
(`order_delivered_customer_date`) no existe ni en t₀ ni en t₁, y predice el target casi
perfectamente: es la fuga más tentadora del dataset.

**Los agregados miran solo hacia atrás.** Cualquier estadística de vendedor, producto o
cliente —nota media histórica, volumen, tasa de retraso— se calcula únicamente con pedidos
**anteriores** al que se está puntuando. Calcularla sobre el conjunto entero mete el futuro
en el pasado sin que salte ningún error.

**Cada feature se prueba.** `tests/test_features.py` debe contener al menos un caso que
falle si la feature empieza a usar información futura.

---

## Avisos sobre el esquema

Descubiertos en S1 al explorar las tablas. Ver `decisiones.md` D-06.

### `customer_id` no identifica a un cliente

Hay **99.441 `customer_id` distintos para 99.441 pedidos**: uno por pedido. El identificador
real de persona es **`customer_unique_id`** (96.096 valores distintos).

Cualquier feature de historial de cliente construida sobre `customer_id` devolvería siempre
"cliente nuevo, cero pedidos previos" y parecería funcionar. Usar siempre `customer_unique_id`.

### `geolocation` no se une por nombre de columna

El prefijo de código postal se llama distinto en cada tabla:

| Tabla | Columna |
|---|---|
| `customers` | `customer_zip_code_prefix` |
| `sellers` | `seller_zip_code_prefix` |
| `geolocation` | `geolocation_zip_code_prefix` |

La feature de distancia comprador-vendedor de t₀ tendrá que unir explícitamente por columnas
con nombres diferentes. El grafo de joins que imprime `make data` se deriva de nombres
coincidentes, así que `geolocation` no aparece en él.

### Un pedido puede tener varios vendedores

`order_items` tiene 112.650 filas para 99.441 pedidos. Las features de vendedor toman el de
mayor valor de artículo, y `n_sellers` deja constancia del caso.


---

## Lo que todavía no hay

**Historial de vendedor y de cliente.** Ninguna feature actual describe la trayectoria del
vendedor, y un vendedor que llegó tarde en sus últimos veinte pedidos es el predictor obvio.
Requiere agregados con ventana hacia atrás, que es el punto del proyecto donde es más fácil
meter una fuga sin enterarse, y por eso tiene sesión propia (S3).

Hay además una sutileza que la regla 3 de `CLAUDE.md` no cubre del todo: no basta con usar
**pedidos anteriores**, hay que usar **etiquetas ya conocidas**. La reseña de un pedido llega
una mediana de 10 días después de la compra, así que al puntuar un pedido hay pedidos
anteriores cuya reseña todavía no existe. Por eso la espina guarda `review_creation_date`.

**Texto de las reseñas.** Reservado a la fase 2 (NLP). Se carga íntegro pero no se toca.

---

## Dos hallazgos que condicionan el modelado

**No hay ninguna variable fuerte.** El mejor AUC univariante es 0,590. El valor tendrá que
salir de combinar señales débiles, y cualquier predictor individual espectacular en este
dataset debe tratarse como sospecha de fuga antes que como buena noticia.

**Hay dos canales, no uno.** El retraso de entrega explica muchísimo (54,0% de prevalencia en
los pedidos que llegan tarde frente a 9,2% en los que llegan a tiempo), pero no todo:
`n_items` predice la reseña sin predecir el retraso. Más artículos es más superficie para que
algo salga mal —talla equivocada, un artículo que falta— y nada de eso es logística. Un modelo
reducido a predictor de retraso dejaría ese canal fuera.