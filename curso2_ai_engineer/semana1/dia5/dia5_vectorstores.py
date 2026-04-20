# Día 5 — Vectorstores: ChromaDB, FAISS y comparativa
# Curso 2: AI Engineer
# Temas: indexar documentos, persistencia, retrieval, comparativa vectorstores

from dotenv import load_dotenv
import os

load_dotenv(r"C:\Users\50051676\Desktop\Curso_MLOps\curso2_ai_engineer\.env")

# ─────────────────────────────────────────
# PARTE 1 — ChromaDB: indexar y buscar
# ─────────────────────────────────────────

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Documentos — simulan fichas de clientes y políticas del banco
documentos = [
    "El cliente García tiene 3 impagos en 2023 y una PD de 0.15.",
    "La política del banco exige revisión manual para PD superiores a 0.10.",
    "El cliente Martínez tiene historial impecable y PD de 0.02.",
    "Los préstamos hipotecarios requieren garantía real como colateral.",
    "El límite de exposición por cliente es de 500.000€ según normativa interna.",
    "El cliente Rodríguez solicitó un préstamo de 80.000€ para compra de vivienda.",
    "La normativa Basel III exige provisiones del 8% sobre activos ponderados por riesgo.",
    "Los clientes con rating AA tienen acceso a tipos de interés preferenciales.",
]

# Metadatos — información adicional por documento
metadatos = [
    {"tipo": "cliente", "nombre": "García"},
    {"tipo": "politica", "categoria": "riesgo"},
    {"tipo": "cliente", "nombre": "Martínez"},
    {"tipo": "politica", "categoria": "garantias"},
    {"tipo": "politica", "categoria": "limites"},
    {"tipo": "cliente", "nombre": "Rodríguez"},
    {"tipo": "normativa", "fuente": "Basel III"},
    {"tipo": "politica", "categoria": "tipos"},
]

# Crear vectorstore en memoria (sin persistencia todavía)
vectorstore = Chroma.from_texts(
    texts=documentos,
    embedding=embeddings,
    metadatas=metadatos,
    collection_name="banco_docs"
)

print("--- Búsqueda semántica ---")
# Búsqueda simple
resultados = vectorstore.similarity_search(
    "¿Qué clientes tienen alto riesgo?", k=3
)
for r in resultados:
    print(f"[{r.metadata['tipo']}] {r.page_content}")

print("\n--- Búsqueda con score ---")
# Búsqueda con puntuación de similitud
resultados_score = vectorstore.similarity_search_with_score(
    "normativa regulatoria bancaria", k=3
)
for doc, score in resultados_score:
    print(f"Score: {score:.4f} | {doc.page_content[:60]}...")

# ─────────────────────────────────────────
# PARTE 2 — Persistencia en disco
# ─────────────────────────────────────────

# Persistir en disco — los embeddings se guardan y no hay que recalcularlos
vectorstore_persistido = Chroma.from_texts(
    texts=documentos,
    embedding=embeddings,
    metadatas=metadatos,
    collection_name="banco_docs_persistido",
    persist_directory="./chroma_db"  # ← carpeta donde se guarda
)

print("\n--- Vectorstore persistido en ./chroma_db ---")

# Cargar desde disco — sin recalcular embeddings
vectorstore_cargado = Chroma(
    collection_name="banco_docs_persistido",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

resultados = vectorstore_cargado.similarity_search("cliente con impagos", k=2)
print("Cargado desde disco:")
for r in resultados:
    print(f"  [{r.metadata['tipo']}] {r.page_content}")

# ─────────────────────────────────────────
# PARTE 3 — RAG completo con ChromaDB
# ─────────────────────────────────────────

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Retriever — busca los 3 chunks más relevantes
retriever = vectorstore_persistido.as_retriever(search_kwargs={"k": 3})

# Prompt RAG
prompt_rag = ChatPromptTemplate.from_messages([
    ("system", """Eres un asistente de riesgo bancario.
Responde SOLO con la información del contexto proporcionado.
Si la información no está en el contexto, di exactamente: 'No tengo esa información en la base de conocimiento.'"""),
    ("human", "Contexto:\n{contexto}\n\nPregunta: {pregunta}")
])

def formatear_docs(docs):
    return "\n".join(f"- {doc.page_content}" for doc in docs)

# Chain RAG completa
chain_rag = (
    {"contexto": retriever | formatear_docs, "pregunta": RunnablePassthrough()}
    | prompt_rag
    | llm
    | StrOutputParser()
)

print("\n--- RAG completo con ChromaDB ---")
preguntas = [
    "¿Qué debo hacer con el cliente García?",
    "¿Qué normativa regula las provisiones bancarias?",
    "¿Cuánto puede pedir prestado un cliente como máximo?",
    "¿Cuál es el tipo de cambio euro/dólar hoy?"
]

for pregunta in preguntas:
    print(f"\nP: {pregunta}")
    print(f"R: {chain_rag.invoke(pregunta)}")

# ─────────────────────────────────────────
# PARTE 4 — FAISS: velocidad en memoria
# ─────────────────────────────────────────

from langchain_community.vectorstores import FAISS

# Crear vectorstore FAISS — todo en memoria, no persiste
vectorstore_faiss = FAISS.from_texts(
    texts=documentos,
    embedding=embeddings,
    metadatas=metadatos
)

print("\n--- FAISS ---")
resultados_faiss = vectorstore_faiss.similarity_search(
    "¿Qué debo hacer con el cliente García?", k=3
)
for r in resultados_faiss:
    print(f"[{r.metadata['tipo']}] {r.page_content}")

# Guardar y cargar FAISS
vectorstore_faiss.save_local("./faiss_index")
vectorstore_cargado_faiss = FAISS.load_local(
    "./faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("\nFAISS cargado desde disco:")
resultados = vectorstore_cargado_faiss.similarity_search("normativa bancaria", k=2)
for r in resultados:
    print(f"  [{r.metadata['tipo']}] {r.page_content}")

# Comparativa final
print("\n--- Comparativa ChromaDB vs FAISS ---")
print("ChromaDB: base de datos completa, persiste en disco, filtros por metadatos, más lento")
print("FAISS:    índice en memoria, muy rápido, guardado simple, sin filtros avanzados")
