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

## Día 1 — Cómo funcionan los LLMs

### Conceptos clave

**Token** — unidad mínima que procesa el LLM. No es una palabra, es un fragmento. "MLOps" puede ser 1-3 tokens. Importante porque el coste de la API se mide en tokens.

**Contexto (context window)** — cuántos tokens puede "recordar" el modelo en una conversación. GPT-4o tiene 128K tokens (~300 páginas de texto). Si superas el límite, la información más antigua se pierde.

**Temperatura** — controla la aleatoriedad de las respuestas:
- `0.0` → determinista, siempre la misma respuesta (producción, código)
- `0.7` → balance creatividad/coherencia (chatbots)
- `1.0+` → muy creativo, puede ser incoherente (brainstorming)

**Embedding** — representación numérica de un texto como vector en un espacio de alta dimensión. Textos con significado similar tienen vectores cercanos. Ejemplo: `"perro"` y `"can"` tendrán vectores más parecidos entre sí que `"perro"` y `"avión"`.

```python
from openai import AzureOpenAI

client = AzureOpenAI(...)

# Generar embedding de un texto
respuesta = client.embeddings.create(
    model="text-embedding-ada-002",
    input="El modelo de scoring de crédito predice el riesgo de impago"
)
vector = respuesta.data[0].embedding  # lista de 1536 números flotantes
```

**¿Para qué sirven los embeddings?**
- RAG: buscas los documentos más relevantes para una pregunta comparando vectores
- Búsqueda semántica: "encontrar documentos similares" aunque usen palabras distintas
- Clasificación: agrupar textos por significado

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
