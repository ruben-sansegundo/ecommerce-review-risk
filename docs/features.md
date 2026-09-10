# Features y disponibilidad temporal

Tabla obligatoria por la regla 4 de `CLAUDE.md`: **cada feature declara en qué momento está
disponible**. Se actualiza en el mismo commit que añade la feature, no después.

> **Estado:** vacía. Las features se construyen en S2. Este documento existe desde S1 para
> que la primera feature ya nazca documentada.

---

## La tabla

| Feature | Momento | Tablas de origen | Notas |
|---|---|---|---|
| _(pendiente, S2)_ | | | |

**Momento** es uno de:

- **t₀** — disponible al aprobarse el pedido
- **t₁** — disponible al despacharse el envío (incluye todo lo de t₀)

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
