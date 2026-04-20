# Semana 1 — Fundamentos LLM + Python AI Stack

> "Entender cómo funcionan los LLMs antes de orquestarlos"

## Días

| Día | Tema | Estado |
|-----|------|--------|
| 1 | LLMs: tokens, contexto, temperatura, embeddings | ⬜ |
| 2 | OpenAI API / Azure OpenAI: llamadas, roles, system prompt, function calling | ⬜ |
| 3 | Prompting avanzado: few-shot, chain-of-thought, RAG conceptual | ⬜ |
| 4 | LangChain core: chains, prompts, parsers, memory | ⬜ |
| 5 | LangChain avanzado: retrievers, vectorstores — ChromaDB, FAISS y Milvus | ⬜ |
| 6 | Proyecto: Chatbot RAG con memoria y búsqueda en documentos | ⬜ |
| 7 | Test semana 1 + repaso | ⬜ |

**Checkpoint:** Chatbot RAG que responde preguntas sobre documentos usando LangChain + ChromaDB, con embeddings de Azure OpenAI y trazabilidad en LangSmith.

---

## Día 1 — Cómo funcionan los LLMs, Embeddings y la base de RAG

### Conceptos clave

---

### ¿Qué es un LLM?

Un LLM (Large Language Model) es un modelo de lenguaje entrenado con enormes cantidades de texto para predecir qué token viene a continuación. Durante el entrenamiento aprende patrones estadísticos del lenguaje humano. En inferencia, dada una entrada (prompt), genera una respuesta token a token.

Lo que parece "razonamiento" es interpolación sofisticada sobre patrones aprendidos. No almacena nada entre llamadas — cada llamada empieza desde cero.

**El LLM tiene dos capas de conocimiento:**
```
Capa 1 — Entrenamiento previo (de OpenAI, no modificable)
─────────────────────────────────────────────────────────
Sabe qué es un impago, qué es un contrato, cómo redactar
una respuesta coherente en español, gramática, lógica...
Todo el conocimiento general del mundo.

Capa 2 — Contexto que tú le mandas (tus chunks)
─────────────────────────────────────────────────
Los datos específicos de TU negocio: contratos, clientes,
normativa interna, historiales...
```

---

### Token

Unidad mínima que procesa el LLM. No es una palabra — es un fragmento. "MLOps" puede ser 1-3 tokens. Importante porque el coste de la API se mide en tokens.

- GPT-4o-mini: 0.15$ por millón de tokens de entrada, 0.60$ por millón de salida
- Una llamada típica de práctica (~90 tokens) cuesta ~0.00004$
- Con 5$ tienes ~125.000 llamadas de práctica

---

### Contexto (context window)

Cuántos tokens puede procesar el modelo en una llamada. GPT-4o tiene 128K tokens (~300 páginas de texto). Si superas el límite, la información más antigua se pierde.

---

### Temperatura

Controla la aleatoriedad de las respuestas:
- `0.0` → casi determinista, respuestas muy similares entre ejecuciones (producción, código, respuestas factuales)
- `0.7` → balance creatividad/coherencia (chatbots)
- `1.0+` → muy creativo, puede ser incoherente (brainstorming)

> Nota: temperatura 0 no es 100% determinista por el paralelismo en GPU, pero la variación es mínima.

---

### Roles en la API

```python
messages=[
    {"role": "system", "content": "Eres un asistente experto en MLOps."},  # personalidad del modelo
    {"role": "user",   "content": "¿Qué es un embedding en 2 frases?"}      # pregunta del usuario
]
```

- `system` → define el comportamiento y personalidad del modelo
- `user` → mensaje del usuario
- `assistant` → respuesta del modelo (se usa para pasar historial de conversación)

---

### ¿Qué es un Embedding?

Un embedding es la representación numérica de un texto como vector en un espacio de alta dimensión.

- `text-embedding-ada-002` genera vectores de **1536 dimensiones**
- Textos con significado similar tienen vectores cercanos (apuntan en la misma dirección)
- Textos sin relación tienen vectores lejanos (apuntan en direcciones distintas)

```python
embedding_respuesta = client.embeddings.create(
    model="text-embedding-ada-002",
    input="El modelo de scoring de crédito predice el riesgo de impago"
)
vector = embedding_respuesta.data[0].embedding  # lista de 1536 números flotantes
# [-0.026, -0.018, 0.030, -0.019, -0.005, ...]
```

**Resultado en práctica:**
```
Similitud 'impago' vs 'riesgo crédito': 0.8094  ← semánticamente relacionados
Similitud 'impago' vs 'receta de cocina': 0.7642 ← sin relación
```

**¿Cómo se calcula la similitud? → Similitud coseno**

No compara número a número. Trata cada vector como una flecha en espacio de 1536 dimensiones y mide el ángulo entre ellas:

```python
import numpy as np

def similitud(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
```

- Flechas apuntando en la misma dirección → significado similar → score cercano a 1
- Flechas apuntando en direcciones opuestas → significado diferente → score cercano a 0

---

### LLM vs Modelo de Embeddings — son cosas distintas

```
text-embedding-ada-002          gpt-4o-mini
(modelo de embeddings)          (LLM)
        |                           |
Vectoriza textos            Entiende lenguaje natural
Genera vectores             y genera respuestas en texto
No "habla"                  No genera vectores
        |                           |
        └──────── client ───────────┘
                    |
            Conector Python que
            accede a ambos modelos
```

El `client = OpenAI(api_key=...)` es solo el conector Python. Los modelos viven en los servidores de OpenAI.

---

### ¿Qué modelo de embeddings usar?

| Situación | Modelo recomendado |
|-----------|-------------------|
| Cliente Azure / OpenAI, caso general | `text-embedding-ada-002` |
| Más calidad, presupuesto mayor | `text-embedding-3-large` (3072 dims) |
| Open source, sin coste por llamada | `sentence-transformers` (384-768 dims) |
| Documentos en español exclusivamente | `paraphrase-multilingual-mpnet-base-v2` |

**Regla crítica:** el modelo que usas para indexar y el que usas para buscar deben ser siempre el mismo. Si mezclas modelos los vectores son incomparables.

---

### Chunk, Embedding y Vectorstore — diferencias

```
Chunk      → fragmento de texto original de un documento (~500 palabras)
Embedding  → vector de 1536 números que representa ese chunk
Vectorstore → base de datos que almacena pares (chunk, embedding)
```

El vectorstore NO es un vector más grande. Es una base de datos especializada en búsqueda por similitud:

```
| CHUNK (texto)                              | EMBEDDING (vector)          |
|--------------------------------------------|------------------------------|
| "García tiene 3 impagos en 2023"           | [-0.026, -0.018, 0.030, ...] |
| "La política de riesgo exige garantías"    | [-0.041, 0.012, -0.009, ...] |
| "El contrato vence en diciembre de 2025"   | [0.018, -0.033, 0.021, ...]  |
```

La búsqueda no es por palabras exactas como SQL — es por similitud entre vectores.

---

### La base de RAG — flujo completo

RAG (Retrieval-Augmented Generation) combina el vectorstore con el LLM:

```
FASE 1 — Indexación (se hace una vez, tú la construyes)
────────────────────────────────────────────────────────
Tienes PDFs / documentos
        ↓
Tu código los lee y trocea en chunks (~500 palabras)
        ↓
Tu código pasa cada chunk por text-embedding-ada-002
        ↓
Tu código guarda (chunk + vector) en el vectorstore
El vectorstore empieza vacío — tú lo rellenas

FASE 2 — Consulta (ocurre en tiempo real)
──────────────────────────────────────────
Usuario hace una pregunta
        ↓
Tu código vectoriza la pregunta con text-embedding-ada-002
        ↓
Tu código busca en el vectorstore los 3-5 chunks más similares
(similitud coseno entre vector_pregunta y todos los embeddings)
        ↓
Tu código construye el prompt:
  "Contexto: [chunk1] [chunk2] [chunk3]
   Pregunta: ¿cuál es el riesgo de impago de García?
   Responde SOLO con el contexto anterior."
        ↓
gpt-4o-mini recibe ese prompt y responde
basándose en los chunks, NO en su entrenamiento
        ↓
Respuesta al usuario
```

**¿Por qué RAG y no meter todos los documentos en el contexto?**
- Coste: 500 PDFs en cada llamada sería carísimo en tokens
- Calidad: los LLMs rinden peor con contextos muy largos (lost-in-the-middle problem)
- RAG selecciona solo los 3-5 fragmentos más relevantes para cada pregunta

**¿Dónde está la "memoria" del sistema?**
```
Conocimiento del mundo    →  en el entrenamiento del LLM (no modificable)
Tus documentos            →  en el vectorstore (tú lo construyes)
Contexto de la llamada    →  en el prompt (se manda en cada llamada)
Historial conversación    →  en tu código (mandas mensajes anteriores)
```

---

### Código del Día 1

Ver [dia1/dia1_llm_basico.py](dia1/dia1_llm_basico.py)

---

### Preguntas y dudas del Día 1 — resueltas

---

**¿El LLM es básicamente el cliente que nos da la respuesta?**

El LLM no es el cliente. El cliente es el objeto Python `client = OpenAI(api_key=...)` — es solo el conector a la API. El LLM es el modelo que vive en los servidores de OpenAI (`gpt-4o-mini`). El cliente es la herramienta para llegar a él, no el modelo en sí.

---

**¿En qué momento hemos definido el nombre del contenedor "dia3-api"?**

No se define explícitamente. Docker Compose lo genera automáticamente con el patrón: `{carpeta}-{servicio}-{réplica}`. En nuestro caso: carpeta `dia3` + servicio `api` + réplica `1` = `dia3-api-1`. Si quisieras un nombre fijo usarías `container_name:` en el compose.

---

**¿La pregunta del usuario es el chunk?**

No. Son roles distintos:
- **Chunk** — fragmento de tus documentos indexado previamente en el vectorstore
- **Pregunta** — lo que escribe el usuario en tiempo real, no se guarda en el vectorstore

Lo que tienen en común: ambos pasan por `text-embedding-ada-002` para convertirse en vector. Pero su rol es opuesto — los chunks se indexan, la pregunta se usa para buscar.

---

**¿El vectorstore es un vector más grande que agrupa todos los embeddings?**

No. El vectorstore es una base de datos — como MySQL pero especializada en vectores. No es un vector más grande. Almacena pares (chunk de texto + su embedding) y su función especial es buscar por similitud entre vectores, no por palabras exactas.

---

**¿La búsqueda en el vectorstore se hace mirando los embeddings posición por posición?**

No compara número por número. Usa similitud coseno — trata cada vector como una flecha en espacio de 1536 dimensiones y mide el ángulo entre la flecha de la pregunta y las flechas de todos los chunks. Las flechas que apuntan en la misma dirección tienen significado similar. No importan los valores individuales de cada posición, importa la dirección del vector completo.

---

**¿Siempre hay 1536 números por vector?**

Depende del modelo, no siempre. Cada modelo tiene su dimensión fija:
- `text-embedding-ada-002` → 1536
- `text-embedding-3-large` → 3072
- `sentence-transformers` → 384 o 768

Lo importante: todos los vectores del mismo modelo tienen exactamente las mismas dimensiones. Por eso no puedes mezclar modelos en el mismo vectorstore — los vectores serían incomparables.

---

**¿Los chunks los crea el vectorstore automáticamente o los creas tú?**

Los creas tú. El vectorstore empieza vacío. Tú escribes el código que lee los PDFs, los trocea en chunks, genera los embeddings y los guarda en el vectorstore. Lo haremos en el Día 5 del curso.

---

**¿El LLM responde con información de su entrenamiento o de los chunks?**

En RAG responde basándose en los chunks que tú le mandas, NO en su entrenamiento. Le dices explícitamente: "Responde SOLO con la información de este contexto. Si no está aquí, di que no lo sabes." Así evitas que invente cosas de su entrenamiento que pueden ser incorrectas o desactualizadas.

El LLM usa su entrenamiento para saber cómo leer, entender y redactar la respuesta — pero el contenido factual viene de tus chunks.

---

**¿Entonces el LLM no almacena nada de mis documentos?**

Correcto. El LLM es como una persona con amnesia total — cada llamada empieza desde cero. No recuerda nada de llamadas anteriores ni tiene base de datos propia de tus documentos. Todo lo que "sabe" en ese momento es lo que tú le mandas en el prompt de esa llamada.

La memoria del sistema vive en el vectorstore, no en el LLM.

---

**¿El LLM ha sido entrenado para responder solo con el contexto que le das?**

Sí. El LLM fue entrenado por OpenAI para seguir instrucciones. Cuando le dices "responde solo con este contexto", lo cumple porque durante su entrenamiento aprendió a seguir instrucciones del system prompt. Su entrenamiento le da la capacidad de entender y responder — tus chunks le dan los datos específicos para responder sobre tu negocio. RAG combina las dos cosas.

---

## Día 5 — Vectorstores: ChromaDB, FAISS y Milvus

### ¿Qué es un vectorstore?

Una base de datos especializada en almacenar embeddings (vectores) y buscar los más similares a un vector query. Es el componente central de cualquier sistema RAG.

**Flujo RAG completo:**
```
Documentos (PDFs, textos...)
        ↓
Dividir en chunks (trozos)
        ↓
Generar embedding de cada chunk (Azure OpenAI)
        ↓
Guardar embeddings en vectorstore
        ↓
[Cuando llega una pregunta]
        ↓
Generar embedding de la pregunta
        ↓
Buscar los N chunks más similares en el vectorstore
        ↓
Enviar pregunta + chunks relevantes al LLM
        ↓
LLM genera respuesta basada en el contexto
```

---

### ChromaDB — para desarrollo

**Cuándo usarlo:** proyectos locales, prototipos, desarrollo. Fácil de arrancar, sin infraestructura.

```python
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(model="text-embedding-ada-002")

# Crear vectorstore desde documentos
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"   # persiste en disco
)

# Buscar documentos relevantes
resultados = vectorstore.similarity_search("¿Cuál es el riesgo de impago?", k=3)
```

---

### FAISS — para velocidad en memoria

**Qué es:** librería de Meta para búsqueda de similitud extremadamente rápida en memoria. Ideal cuando tienes millones de vectores y necesitas respuesta en milisegundos.

**Cuándo usarlo:** cuando ChromaDB es demasiado lento pero no necesitas infraestructura distribuida.

```python
from langchain_community.vectorstores import FAISS

# Crear y guardar
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")

# Cargar
vectorstore = FAISS.load_local("faiss_index", embeddings)
resultados = vectorstore.similarity_search("consulta", k=5)
```

**Diferencia clave con ChromaDB:** FAISS es solo búsqueda en memoria — no tiene UI, no persiste fácilmente en cloud. ChromaDB es más completo como base de datos.

---

### Milvus — para producción a escala

**Qué es:** base de datos vectorial distribuida open source. Diseñada para producción con millones o miles de millones de vectores, alta disponibilidad y escalabilidad horizontal.

**Cuándo usarlo vs ChromaDB/FAISS:**
| | ChromaDB | FAISS | Milvus |
|---|---|---|---|
| Escala | Miles de docs | Millones (memoria) | Miles de millones |
| Despliegue | Local/simple | En memoria | Cluster distribuido |
| HA / failover | No | No | Sí |
| Caso de uso | Dev / prototipo | Velocidad memoria | Producción enterprise |
| Kyndryl | Prototipado | Tests de velocidad | Clientes grandes |

**Arquitectura Milvus en producción:**
```
App (LangChain) → Milvus Proxy
                       ↓
            Query Nodes + Data Nodes
                       ↓
              MinIO (almacenamiento)
              etcd (metadatos)
```

**Uso con LangChain:**
```python
from langchain_community.vectorstores import Milvus

vectorstore = Milvus.from_documents(
    documents=chunks,
    embedding=embeddings,
    connection_args={
        "host": "milvus-server",
        "port": 19530
    },
    collection_name="documentos_cliente"
)

resultados = vectorstore.similarity_search("consulta", k=5)
```

**Despliegue local para práctica (Docker Compose):**
```yaml
version: '3.5'
services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
  minio:
    image: minio/minio:RELEASE.2023-03-13T19-46-17Z
  standalone:
    image: milvusdb/milvus:v2.3.0
    ports:
      - "19530:19530"
```

---

### Cuándo usar cada uno — regla práctica

```
¿Estás prototipando?
    → ChromaDB

¿Necesitas máxima velocidad y los datos caben en RAM?
    → FAISS

¿Es producción con muchos documentos y clientes enterprise?
    → Milvus

¿Usas Azure y quieres managed service?
    → Azure AI Search (vectorstore gestionado de Microsoft)
```

---

---

## Día 2 — OpenAI API en profundidad

### Temas: historial de conversación, system prompt avanzado, function calling

---

### Parte 1 — Conversación con historial

El LLM no recuerda nada entre llamadas. Cada llamada es independiente. Para que "recuerde" el contexto anterior tienes que mandarle el historial completo en cada llamada.

**¿Cómo funciona?**

Mantienes una lista `historial` en tu código. Cada turno añades el mensaje del usuario y la respuesta del modelo. En cada llamada mandas la lista completa:

```python
historial = [
    {"role": "system", "content": "Eres un asistente experto en riesgo financiero."}
]

def chat(mensaje):
    historial.append({"role": "user", "content": mensaje})
    respuesta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=historial,   # ← mandas TODO el historial
        temperature=0.7
    )
    contenido = respuesta.choices[0].message.content
    historial.append({"role": "assistant", "content": contenido})
    return contenido
```

**Por qué importa:** preguntas ambiguas como "¿Y cómo se mide?" o "¿Cuál de esas métricas...?" funcionan porque el modelo tiene el contexto anterior. Sin historial, no sabría a qué se refiere "esas métricas".

**Límite importante:** el historial consume tokens en cada llamada. Una conversación larga puede volverse cara. En producción se gestionan estrategias como truncar el historial antiguo o resumirlo.

---

### Parte 2 — System prompt avanzado

El system prompt no solo da personalidad — controla formato, idioma, tono y límites del modelo.

```python
{"role": "system", "content": """Eres un asistente de riesgo financiero para un banco español.
Reglas estrictas:
- Responde SIEMPRE en español
- Responde SOLO sobre temas de riesgo financiero y banca
- Si te preguntan algo fuera de ese ámbito, di exactamente: "Solo puedo ayudarte con temas de riesgo financiero."
- Máximo 3 frases por respuesta
- Nunca uses bullet points, solo párrafos"""}
```

**Resultado:** al preguntar "¿Cuál es la capital de Francia?" responde exactamente: *"Solo puedo ayudarte con temas de riesgo financiero."*

**Para qué sirve en producción:**
- Chatbots internos de empresa: limitar el ámbito de respuestas
- Definir el formato exacto de salida (JSON, markdown, texto plano)
- Forzar un idioma o tono específico
- Instrucciones de seguridad y compliance

---

### Parte 3 — Function calling

Function calling permite que el LLM decida cuándo necesita datos externos y qué función de tu código debe llamar. El LLM **no ejecuta** la función — te dice qué ejecutar y con qué parámetros. Tú la ejecutas y le devuelves el resultado.

**¿Por qué es la base de los agentes?** Porque un agente es exactamente esto pero con múltiples herramientas en bucle — el LLM decide qué herramienta usar, tú la ejecutas, el LLM ve el resultado y decide si necesita otra herramienta o ya puede responder.

**Flujo completo:**
```
Usuario: "¿riesgo del cliente C001?"
        ↓
1ª llamada al LLM → "necesito llamar a obtener_riesgo_cliente(C001)"
        ↓
Tú ejecutas la función → {"nombre": "García", "pd": 0.15, "rating": "B"}
        ↓
2ª llamada al LLM con el resultado → respuesta en lenguaje natural
        ↓
"El cliente García tiene PD 15% y rating B — riesgo moderado"
```

**Código:**
```python
import json

# 1. Función real de tu código (el LLM nunca la toca directamente)
def obtener_riesgo_cliente(cliente_id: str) -> dict:
    datos = {
        "C001": {"nombre": "García", "pd": 0.15, "rating": "B"},
        "C002": {"nombre": "Martínez", "pd": 0.03, "rating": "AA"},
    }
    return datos.get(cliente_id, {"error": "cliente no encontrado"})

# 2. Descripción de la función para el LLM (en JSON — el LLM no lee Python)
herramientas = [{
    "type": "function",
    "function": {
        "name": "obtener_riesgo_cliente",
        "description": "Obtiene el perfil de riesgo de un cliente dado su ID",
        "parameters": {
            "type": "object",
            "properties": {
                "cliente_id": {"type": "string", "description": "El ID del cliente, ej: C001"}
            },
            "required": ["cliente_id"]
        }
    }
}]

# 3. Primera llamada — el LLM decide si necesita llamar a una función
respuesta_fc = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    tools=herramientas,
    tool_choice="auto"   # el LLM decide solo
)

# 4. Tú ejecutas la función con los argumentos que el LLM eligió
if mensaje.tool_calls:
    argumentos = json.loads(tool_call.function.arguments)
    resultado = obtener_riesgo_cliente(**argumentos)

# 5. Segunda llamada — el LLM genera respuesta con los datos reales
respuesta_final = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        ...,
        mensaje,                                                    # respuesta del paso 3
        {"role": "tool", "tool_call_id": tool_call.id,
         "content": json.dumps(resultado)}                          # resultado de tu función
    ]
)
```

**Puntos clave:**
- El LLM describe qué quiere ejecutar, no lo ejecuta
- Tú tienes control total sobre qué se ejecuta y cuándo
- `tool_choice="auto"` → el LLM decide; `tool_choice="none"` → nunca llama funciones
- El LLM no inventa datos — espera el resultado real de tu función

---

### Código del Día 2

Ver [dia2/dia2_openai_api.py](dia2/dia2_openai_api.py)

---

### Preguntas y dudas del Día 2 — resueltas

**¿Por qué el LLM entiende preguntas ambiguas como "¿Y cómo se mide?"?**

Porque en cada llamada le mandas el historial completo de la conversación. El LLM ve todos los mensajes anteriores y puede inferir el contexto. Sin el historial, esa pregunta no tendría sentido para él.

**¿El LLM ejecuta la función directamente?**

No. El LLM solo decide qué función llamar y con qué argumentos. Tú eres quien ejecuta la función en tu código y le devuelves el resultado. El LLM nunca tiene acceso directo a tu sistema — todo pasa por ti.

**¿Qué diferencia hay entre function calling y un agente?**

Function calling es una llamada única donde el LLM decide usar una herramienta. Un agente es function calling en bucle — el LLM puede usar múltiples herramientas en secuencia, ver los resultados intermedios y decidir el siguiente paso hasta completar la tarea.

---

## Día 3 — Prompting avanzado + RAG conceptual

### Temas: few-shot prompting, chain-of-thought, mini RAG sin vectorstore

---

### Parte 1 — Few-shot prompting

En vez de explicarle al LLM el formato que quieres con instrucciones, le das ejemplos directamente en el historial de mensajes. El modelo aprende el patrón de los ejemplos y lo aplica a casos nuevos.

```python
messages=[
    {"role": "system",    "content": "Clasifica el riesgo de clientes bancarios."},
    {"role": "user",      "content": "Cliente: Juan, PD: 0.02, deuda: 5000€"},
    {"role": "assistant", "content": "RIESGO: BAJO | ACCIÓN: Aprobar automáticamente"},
    {"role": "user",      "content": "Cliente: María, PD: 0.25, deuda: 50000€"},
    {"role": "assistant", "content": "RIESGO: ALTO | ACCIÓN: Revisión manual urgente"},
    {"role": "user",      "content": "Cliente: Pedro, PD: 0.12, deuda: 20000€"},  # ← nuevo caso
]
```

**Resultado:** el modelo clasifica a Pedro como `RIESGO: MEDIO | ACCIÓN: Evaluar condiciones adicionales` — aprendió el formato y la escala de riesgo solo con 2 ejemplos.

**Cuándo usarlo:** cuando necesitas un formato de salida muy específico y es más fácil mostrar ejemplos que describir el formato con palabras. También cuando quieres que el modelo aprenda el tono o estilo de respuesta de tu empresa.

---

### Parte 2 — Chain-of-thought

Fuerza al LLM a razonar paso a paso antes de dar la respuesta final. Mejora la precisión en problemas que requieren lógica, cálculos o toma de decisiones.

```python
{"role": "system", "content": """Eres un analista de riesgo.
Antes de responder, razona paso a paso:
1. Calcula la pérdida esperada (PE = PD × LGD × EAD)
2. Evalúa si PE es aceptable para el banco (umbral: 5000€)
3. Da tu recomendación final"""}
```

**Comparativa:**

| | Sin CoT | Con CoT |
|---|---|---|
| Cálculo | Correcto (3600€) | Correcto (3600€) |
| Decisión | "Depende de varios factores..." | "Aprobar — PE por debajo del umbral" |

Chain-of-thought no mejora el cálculo matemático — mejora la **toma de decisión final**. Al forzar el razonamiento paso a paso el modelo llega a conclusiones concretas en vez de respuestas ambiguas.

**Para producción bancaria esto es crítico** — necesitas recomendaciones accionables, no "podría ser razonable considerar...".

---

### Parte 3 — Mini RAG sin vectorstore

RAG implementado a mano sin ninguna librería, para entender exactamente qué ocurre por dentro.

**Lo que hace:**
1. Genera embeddings de todos los documentos (fase indexación)
2. Genera el embedding de la pregunta
3. Calcula similitud coseno entre la pregunta y cada documento
4. Recupera los N chunks más similares
5. Los manda al LLM como contexto

```python
# Calcular similitudes
similitudes = [similitud_coseno(embedding_pregunta, emb) for emb in embeddings_docs]

# Recuperar top N
indices_top = np.argsort(similitudes)[-N:][::-1]
chunks_relevantes = [documentos[i] for i in indices_top]
```

**`np.argsort` explicado:**
```
similitudes = [0.83, 0.75, 0.79, 0.75, 0.75]
argsort     = [1,    2,    3,    4,    0   ]  ← índices ordenados menor→mayor
[-3:]       = [4,    0   ]                    ← los 3 índices con mayor similitud
[::-1]      = [0,    4   ]                    ← el más similar primero
```
Devuelve índices, no valores. El índice te dice qué documento recuperar.

---

### Problema real descubierto: fallo de retrieval

Con la pregunta "¿Qué debo hacer con el cliente García?" las similitudes reales fueron:

```
[0] 0.8383 — García (3 impagos, PD 0.15)           ← recuperado ✅
[1] 0.7534 — Política banco (revisión si PD > 0.10) ← NO recuperado ❌ (similitud más baja)
[2] 0.7957 — Martínez (historial impecable)         ← recuperado ✅ (pero no útil)
[3] 0.7548 — Préstamos hipotecarios                 ← recuperado con N=4
[4] 0.7568 — Límite exposición                      ← recuperado con N=4
```

**El problema:** "¿Qué debo hacer con García?" es semánticamente más cercana a otros clientes que a una política interna. La búsqueda semántica no siempre recupera lo que lógicamente necesitas.

Con N=2 el LLM respondió "No tengo esa información". Con N=5 respondió correctamente: "Debes realizar una revisión manual ya que PD 0.15 supera el umbral de 0.10."

**Cómo se resuelve en producción:**

```
1. Re-ranking — recuperar top-10, luego un segundo modelo reordena 
   por relevancia real

2. Mejor chunking — separar datos de clientes y políticas en 
   colecciones distintas del vectorstore

3. HyDE (Hypothetical Document Embeddings) — generar una respuesta 
   hipotética y buscar chunks similares a ella, no a la pregunta

4. Query expansion — reformular la pregunta antes de buscar:
   "¿Qué debo hacer con García?" → 
   "política revisión manual cliente PD alto impagos"
```

Esto es lo que diferencia un RAG de prototipo de uno de producción.

---

### Código del Día 3

Ver [dia3/dia3_prompting_rag.py](dia3/dia3_prompting_rag.py)

---

### Preguntas y dudas del Día 3 — resueltas

**¿Qué hace `np.argsort(similitudes)[-2:][::-1]`?**

- `np.argsort(similitudes)` → ordena las similitudes de menor a mayor y devuelve los **índices** de los documentos en ese orden (no los valores)
- `[-2:]` → coge los últimos 2 elementos, es decir, los 2 índices con mayor similitud
- `[::-1]` → invierte el orden para que el más similar vaya primero

Ejemplo: si las similitudes son `[0.83, 0.75, 0.79, 0.75, 0.76]`, argsort devuelve `[1, 3, 4, 2, 0]`. Los últimos 2 son `[2, 0]` — el documento 2 y el documento 0 son los más similares.

**¿Por qué el chunk de la política del banco tenía la similitud más baja si era el más relevante?**

Porque la similitud coseno mide cercanía semántica, no relevancia lógica. "¿Qué debo hacer con García?" es semánticamente más parecido a frases sobre clientes bancarios que a frases sobre políticas internas. La búsqueda semántica no entiende que para responder esa pregunta necesitas la política — solo ve qué textos se parecen en significado.

---

## Día 4 — LangChain core

### Temas: chains, prompts, parsers, memory + comparativa con código manual

---

### ¿Qué es LangChain?

LangChain es una librería que abstrae todo lo que hiciste a mano los días 1-3. En vez de construir diccionarios de mensajes, calcular similitud coseno y gestionar historiales manualmente, LangChain lo hace por ti con menos código.

**Lo importante:** los días 1-3 a mano te permiten entender exactamente qué hace LangChain por debajo. Muchos desarrolladores usan LangChain sin entender qué ocurre — si algo falla no saben debuggearlo. Tú sí.

---

### El operador `|` — cómo funciona LangChain

LangChain encadena pasos con `|`. Cada paso recibe la salida del anterior:

```python
chain = prompt | llm | parser
# prompt rellena la plantilla → llm genera respuesta → parser extrae el texto
```

---

### Parte 1 — ChatModel + PromptTemplate

Plantillas reutilizables con variables — en vez de construir el diccionario de mensajes a mano:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un analista de riesgo financiero de {banco}."),
    ("user", "Analiza el riesgo del cliente con PD={pd} y deuda de {deuda}€")
])

chain = prompt | llm
respuesta = chain.invoke({"banco": "Santander", "pd": 0.15, "deuda": 50000})
print(respuesta.content)
```

**vs Manual (Día 1/2):**
```python
# Manual
messages=[
    {"role": "system", "content": f"Eres analista de {banco}"},
    {"role": "user", "content": f"PD={pd}, deuda={deuda}€"}
]
client.chat.completions.create(model="gpt-4o-mini", messages=messages)
```

---

### Parte 2 — Chains encadenadas

Salida de un LLM → entrada del siguiente. Dos llamadas en secuencia:

```python
chain_completa = (
    prompt_analisis          # plantilla 1
    | llm                    # primera llamada al LLM → análisis
    | StrOutputParser()      # extrae el texto del AIMessage
    | (lambda x: {"analisis": x})  # convierte string a dict para el siguiente prompt
    | prompt_accion          # plantilla 2 usa {analisis}
    | llm                    # segunda llamada → recomendación
    | StrOutputParser()      # extrae texto final
)
```

**El lambda explicado:**
- `StrOutputParser()` devuelve un string
- El siguiente prompt espera un diccionario con clave `{analisis}`
- El lambda convierte: `"texto..."` → `{"analisis": "texto..."}`
- Sin el lambda daría error porque el prompt no acepta strings directamente

---

### Parte 3 — Output parsers

Forzar al LLM a devolver JSON estructurado directamente como diccionario Python:

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

chain_json = prompt_json | llm | parser

resultado = chain_json.invoke({"pd": 0.15, "deuda": 50000})
# resultado ya es un dict Python:
# {'riesgo': 'bajo', 'pd_porcentaje': 15, 'accion': '...', 'aprobar': True}

print(resultado['aprobar'])  # True — acceso directo, no parseo manual
```

**vs Manual:**
```python
# Sin parser: tienes que parsear tú
import json
texto = respuesta.choices[0].message.content
datos = json.loads(texto)
```

**Cuándo usar `JsonOutputParser`:** cuando necesitas procesar la respuesta en código — tomar decisiones, actualizar BD, enviar alertas. Para texto libre no hace falta.

---

### Parte 4 — Memory

LangChain gestiona el historial de conversación automáticamente por `session_id`:

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

sesiones = {}
def get_session(session_id):
    if session_id not in sesiones:
        sesiones[session_id] = ChatMessageHistory()
    return sesiones[session_id]

chain_con_memoria = RunnableWithMessageHistory(
    chain,
    get_session,
    input_messages_key="input",
    history_messages_key="historial"
)

config = {"configurable": {"session_id": "sesion_001"}}
chain_con_memoria.invoke({"input": "¿Qué es la PD?"}, config=config)
chain_con_memoria.invoke({"input": "¿Y cómo se calcula?"}, config=config)
# La segunda pregunta entiende el contexto automáticamente
```

**vs Manual (Día 2):** tú mantenías y pasabas la lista `historial` en cada llamada. Con `RunnableWithMessageHistory` cada `session_id` tiene su propio historial — útil en producción con múltiples usuarios simultáneos.

---

### ¿Puedo mezclar LangChain con código manual?

Técnicamente sí, pero no es recomendable. LangChain usa sus propios objetos internos (`AIMessage`, etc.) que son incompatibles con los dicts de OpenAI sin conversión manual. 

**Regla práctica:**
- ¿Usas LangChain? → úsalo para todo en ese proyecto
- ¿Estás en manual? → quédate en manual

Lo que sí puedes mezclar sin problemas: librería `openai` para embeddings (devuelve arrays numpy) con LangChain para chains.

---

### Cuándo usar LangChain vs manual

| | LangChain | Manual |
|---|---|---|
| Historial multi-sesión | ✅ Mucho mejor | Tienes que gestionar tú |
| RAG con vectorstore | ✅ Mucho mejor | ~20 líneas más |
| Few-shot con muchos ejemplos | ✅ Más limpio | Igual de válido |
| Agentes con múltiples herramientas | ✅ Diseñado para eso | Muy complejo |
| System prompt / CoT | ⚠️ Igual | Igual |
| Embeddings simples | ⚠️ Igual | Igual |
| Function calling simple | ❌ Añade complejidad | Más claro |
| Sistemas críticos / debug | ❌ Demasiada abstracción | Más control |

---

### Lección importante del Mini RAG comparativo

El Mini RAG con LangChain también falló el retrieval (respondió "No tengo esa información") igual que en el Día 3 con código manual. **LangChain no resuelve los problemas de diseño de RAG, solo abstrae el código.** El problema de retrieval — que la búsqueda semántica no siempre recupera lo que lógicamente necesitas — existe con o sin LangChain.

---

### Código del Día 4

- [dia4/dia4_langchain_core.py](dia4/dia4_langchain_core.py) — ejercicios principales
- [dia4/dia4_comparativa_langchain_vs_manual.py](dia4/dia4_comparativa_langchain_vs_manual.py) — comparativa completa días 1-3 con y sin LangChain

---

## Día 5 — Vectorstores: ChromaDB y FAISS

### Temas: indexar documentos, persistencia, RAG completo, comparativa vectorstores

---

### ¿Qué aporta ChromaDB sobre el InMemoryVectorStore del Día 4?

`InMemoryVectorStore` desaparece al cerrar el programa. ChromaDB persiste en disco — los embeddings se guardan y no hay que recalcularlos en cada ejecución. Es una base de datos vectorial real con colecciones, metadatos y filtros.

---

### Metadatos

Cada documento puede tener metadatos asociados — información adicional que no forma parte del texto pero que puedes usar para filtrar:

```python
metadatos = [
    {"tipo": "cliente", "nombre": "García"},
    {"tipo": "politica", "categoria": "riesgo"},
    {"tipo": "normativa", "fuente": "Basel III"},
]

vectorstore = Chroma.from_texts(
    texts=documentos,
    embedding=embeddings,
    metadatas=metadatos,
    collection_name="banco_docs"
)
```

En producción los metadatos permiten filtrar antes de buscar: "busca solo en documentos de tipo política" o "solo documentos de 2024".

---

### Búsqueda simple vs búsqueda con score

```python
# Búsqueda simple — devuelve documentos
resultados = vectorstore.similarity_search("alto riesgo", k=3)

# Búsqueda con score — devuelve (documento, distancia)
resultados_score = vectorstore.similarity_search_with_score("normativa", k=3)
for doc, score in resultados_score:
    print(f"Score: {score:.4f} | {doc.page_content}")
```

**Importante:** en ChromaDB el score es **distancia**, no similitud.
- `0` = idéntico
- `2` = completamente opuesto

Al contrario que la similitud coseno manual donde `1` = idéntico y `0` = sin relación.

---

### Persistencia en disco

```python
# Crear e indexar — genera embeddings y los guarda en ./chroma_db
vectorstore = Chroma.from_texts(
    texts=documentos,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Cargar desde disco — NO recalcula embeddings, los lee del disco
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
```

---

### Problema de duplicados en producción

Si ejecutas `Chroma.from_texts()` dos veces con la misma `persist_directory`, los documentos se duplican — ChromaDB añade sin comprobar si ya existen.

**Soluciones según el caso:**

```python
# Opción A — Borrar y recrear (batch completo)
vectorstore.delete_collection()
vectorstore = Chroma.from_texts(...)

# Opción B — Separar indexación de consulta (producción)
# indexar.py  → se ejecuta una vez o cuando cambian los docs
# consultar.py → carga desde disco, no indexa

# Opción C — IDs únicos (actualizaciones incrementales)
Chroma.from_texts(texts=documentos, ids=["doc_1", "doc_2", ...])
# Si el ID ya existe, actualiza en vez de duplicar
```

**Regla:**
- Dev/prototipo → Opción A
- Producción → Opción B
- Actualizaciones incrementales → Opción C

---

### RAG completo con ChromaDB

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def formatear_docs(docs):
    return "\n".join(f"- {doc.page_content}" for doc in docs)

chain_rag = (
    {"contexto": retriever | formatear_docs, "pregunta": RunnablePassthrough()}
    | prompt_rag
    | llm
    | StrOutputParser()
)

chain_rag.invoke("¿Qué normativa regula las provisiones bancarias?")
# → "La normativa Basel III exige provisiones del 8%..."

chain_rag.invoke("¿Cuál es el tipo de cambio euro/dólar hoy?")
# → "No tengo esa información en la base de conocimiento."
```

El sistema responde solo con lo que está en los documentos. Si la información no existe, lo dice explícitamente en vez de inventar.

---

### FAISS — vectorstore en memoria

```python
from langchain_community.vectorstores import FAISS

# Crear en memoria
vectorstore_faiss = FAISS.from_texts(texts=documentos, embedding=embeddings)

# Guardar a disco
vectorstore_faiss.save_local("./faiss_index")

# Cargar desde disco
vectorstore_faiss = FAISS.load_local(
    "./faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
```

---

### Comparativa ChromaDB vs FAISS

| | ChromaDB | FAISS |
|---|---|---|
| Persistencia | Base de datos en disco | Fichero binario |
| Velocidad búsqueda | Media | Muy alta |
| Filtros por metadatos | Sí | No |
| Escalabilidad | Miles de docs | Millones (en RAM) |
| Caso de uso | RAG en producción | Búsqueda de alta velocidad |
| Cuándo usarlo | La mayoría de proyectos RAG | Cuando ChromaDB es lento |

---

### Problema persistente de retrieval — García

A lo largo de los días 3, 4 y 5 hemos visto que la pregunta "¿Qué debo hacer con el cliente García?" no recupera el chunk de la política de revisión manual (PD > 0.10), que es el más relevante lógicamente.

**Por qué ocurre:** "¿qué debo hacer?" es semánticamente más similar a otros clientes que a una política interna. La búsqueda semántica mide similitud de significado, no relevancia lógica.

**La solución** — retriever más inteligente con reranking y query expansion — se implementa en el Día 6 (proyecto RAG completo).

---

### Código del Día 5

Ver [dia5/dia5_vectorstores.py](dia5/dia5_vectorstores.py)

---

## Día 6 — Proyecto RAG completo

### Pipeline completo: carga documentos → chunking → indexación → RAG con historial

---

### Arquitectura completa del proyecto

```
normativa_riesgo.txt
        ↓  TextLoader
Texto crudo (1 documento, 1305 caracteres)
        ↓  RecursiveCharacterTextSplitter (chunk_size=300, overlap=50)
5 chunks (uno por sección del documento)
        ↓  OpenAIEmbeddings (text-embedding-ada-002)
Vectores de 1536 dimensiones
        ↓  Chroma.from_documents → persist_directory="./chroma_normativa"
ChromaDB en disco
        ↓  as_retriever(k=5)
Chunks relevantes recuperados
        ↓  LLM (gpt-4o-mini) + prompt RAG
Respuesta fundamentada en la normativa
```

---

### Parte 1 — Cargar y trocear documentos

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("./docs/normativa_riesgo.txt", encoding="utf-8")
documentos_raw = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,     # máximo 300 caracteres por chunk
    chunk_overlap=50,   # 50 chars de solapamiento entre chunks consecutivos
    separators=["\n\n", "\n", ". ", " "]  # orden de preferencia para cortar
)
chunks = splitter.split_documents(documentos_raw)
```

**¿Cómo decide dónde cortar `RecursiveCharacterTextSplitter`?**
Aplica los separadores en orden de prioridad:
1. Primero intenta cortar en párrafos (`\n\n`)
2. Si el chunk sigue siendo muy grande, corta en líneas (`\n`)
3. Si sigue siendo grande, corta en frases (`. `)
4. Si sigue siendo grande, corta en palabras (` `)

**¿Cómo saber si el chunking es correcto?**
- Visual: imprimes los chunks y compruebas que cada uno tiene una idea completa
- Prueba de preguntas: haces preguntas cuya respuesta conoces y ves si el RAG responde bien
- Evaluación con RAGAS (framework específico, lo veremos en Semana 3)

**¿Cuántos caracteres por chunk?**

| Tipo de documento | Chunk size recomendado |
|-------------------|----------------------|
| Normativa, contratos densos | 300-500 chars |
| Artículos, documentación | 500-1000 chars |
| FAQs, fichas cortas | 150-300 chars |

El `chunk_overlap` típico es el 10-20% del `chunk_size`.

---

### Parte 2 — Indexar en ChromaDB

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="normativa_banco",
    persist_directory="./chroma_normativa"
)
```

**Problema de duplicados:** si ejecutas `from_documents` dos veces con la misma `persist_directory`, los documentos se duplican. En producción:
- Dev: borra la carpeta antes de cada ejecución
- Producción: separa el script de indexación del de consulta
- Actualizaciones: usa IDs únicos por documento

---

### Parte 3 — RAG con retrieval mejorado

**El problema de García (resuelto):** la pregunta "¿Qué hago con un cliente con PD 0.15?" no es semánticamente similar a "revisión del comité de riesgos". La solución fue:

1. Usar `k=5` para recuperar todos los chunks (con documentos pequeños es lo más fiable)
2. Cambiar el prompt para permitir razonamiento e inferencia, no solo búsqueda literal

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

prompt_rag = ChatPromptTemplate.from_messages([
    ("system", """Eres un asistente experto en normativa bancaria.
Usa el contexto proporcionado para responder. Puedes razonar e inferir conclusiones
a partir de la información del contexto aunque la respuesta no esté escrita explícitamente.
Si la información no está en el contexto ni se puede inferir, di: 'No encuentro esa información en la normativa.'
Cita siempre de qué sección proviene la información."""),
    ("human", "Contexto:\n{contexto}\n\nPregunta: {pregunta}")
])
```

**Resultado:**
```
P: ¿Qué hago con un cliente con PD de 0.15?
R: PD 0.15 > 10% → riesgo alto (sección 1) → revisión del comité de riesgos (sección 2)

P: ¿Cuál es el tipo de cambio euro/dólar?
R: No encuentro esa información en la normativa.
```

---

### Parte 4 — Chatbot RAG con historial

Combina RAG + memoria de conversación. El sistema recuerda el contexto de preguntas anteriores:

```python
def preguntar(pregunta, session_id="default"):
    contexto = formatear_docs(retriever.invoke(pregunta))
    return chain_con_historial.invoke(
        {"contexto": contexto, "pregunta": pregunta},
        config={"configurable": {"session_id": session_id}}
    )

preguntar("¿Qué hago con un cliente que tiene PD de 0.15?")
# → riesgo alto, revisión del comité

preguntar("¿Y si además tiene 3 impagos este año?")
# → entiende que "además" se refiere al cliente anterior
# → PD 0.15 + 3 impagos → vigilancia especial → no puede pedir préstamos

preguntar("¿Puede ese cliente solicitar un nuevo préstamo?")
# → recuerda todo el contexto → no, está en vigilancia especial
```

---

### Tipos de retriever — cuándo usar cada uno

| Retriever | Cómo funciona | Cuándo usarlo |
|-----------|--------------|---------------|
| `as_retriever(k=N)` | Similitud coseno, top N chunks | Caso base |
| `MultiQueryRetriever` | Genera N versiones de la pregunta | Preguntas ambiguas |
| `MMR` | Chunks relevantes y diversos | Chunks muy similares entre sí |
| `Contextual Compression` | Extrae solo la parte relevante del chunk | Chunks con mucho ruido |
| `Parent-Child` | Indexa chunks pequeños, devuelve el padre | Precisión en búsqueda + contexto amplio |
| `Ensemble` | Semántica + palabras clave combinadas | Documentos con términos muy específicos |

---

### Vectorstores disponibles

| Vectorstore | Cuándo usarlo |
|------------|---------------|
| **ChromaDB** | Desarrollo, prototipos, proyectos medianos |
| **FAISS** | Velocidad máxima en memoria, millones de vectores |
| **Milvus** | Producción enterprise, miles de millones de vectores |
| **Azure AI Search** | Proyectos en Azure, managed service |
| **Pinecone** | Cloud managed, fácil de escalar |
| **pgvector** | Si ya usas PostgreSQL |

---

### Código del Día 6

- [dia6/dia6_proyecto_rag.py](dia6/dia6_proyecto_rag.py)
- [dia6/docs/normativa_riesgo.txt](dia6/docs/normativa_riesgo.txt)

<!-- El contenido de cada día se añade aquí conforme se completa -->
