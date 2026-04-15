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

<!-- El contenido de cada día se añade aquí conforme se completa -->
