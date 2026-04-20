# Día 6 — Proyecto RAG completo
# Curso 2: AI Engineer
# Pipeline completo: carga documentos → chunking → indexación → RAG con retrieval mejorado

from dotenv import load_dotenv
import os

load_dotenv(r"C:\Users\50051676\Desktop\Curso_MLOps\curso2_ai_engineer\.env")

# ─────────────────────────────────────────
# PARTE 1 — Cargar y trocear documentos
# ─────────────────────────────────────────

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Cargar documento
loader = TextLoader("./docs/normativa_riesgo.txt", encoding="utf-8")
documentos_raw = loader.load()

print(f"Documento cargado: {len(documentos_raw)} página(s)")
print(f"Total caracteres: {len(documentos_raw[0].page_content)}")

# Trocear en chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,        # máximo 300 caracteres por chunk
    chunk_overlap=50,      # 50 caracteres de solapamiento entre chunks
    separators=["\n\n", "\n", ". ", " "]  # orden de preferencia para cortar
)

chunks = splitter.split_documents(documentos_raw)

print(f"\nChunks generados: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\n[Chunk {i+1}] ({len(chunk.page_content)} chars)")
    print(chunk.page_content)

# ─────────────────────────────────────────
# PARTE 2 — Indexar en ChromaDB
# ─────────────────────────────────────────

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Borrar colección anterior si existe para evitar duplicados
persist_dir = "./chroma_normativa"

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="normativa_banco",
    persist_directory=persist_dir
)

print(f"\n--- Indexación completada ---")
print(f"Chunks indexados: {vectorstore._collection.count()}")
print(f"Persistido en: {persist_dir}")

# ─────────────────────────────────────────
# PARTE 3 — RAG con retrieval mejorado
# ─────────────────────────────────────────

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Retriever con MultiQueryRetriever — genera 3 versiones de la pregunta
# para aumentar la probabilidad de recuperar los chunks correctos
# k=5 recupera todos los chunks — con solo 5 chunks en el vectorstore
# es más fiable que MultiQueryRetriever para este caso
retriever_mejorado = vectorstore.as_retriever(search_kwargs={"k": 5})

prompt_rag = ChatPromptTemplate.from_messages([
    ("system", """Eres un asistente experto en normativa bancaria.
Usa el contexto proporcionado para responder. Puedes razonar e inferir conclusiones
a partir de la información del contexto aunque la respuesta no esté escrita explícitamente.
Si la información no está en el contexto ni se puede inferir, di: 'No encuentro esa información en la normativa.'
Cita siempre de qué sección proviene la información."""),
    ("human", "Contexto:\n{contexto}\n\nPregunta: {pregunta}")
])

def formatear_docs(docs):
    # Eliminar duplicados manteniendo orden
    vistos = set()
    unicos = []
    for doc in docs:
        if doc.page_content not in vistos:
            vistos.add(doc.page_content)
            unicos.append(doc)
    return "\n\n".join(f"[Sección]\n{doc.page_content}" for doc in unicos)

chain_rag = (
    {"contexto": retriever_mejorado | formatear_docs, "pregunta": RunnablePassthrough()}
    | prompt_rag
    | llm
    | StrOutputParser()
)

print("\n--- RAG con retrieval mejorado ---")
preguntas = [
    "¿Qué hago con un cliente que tiene PD de 0.15?",
    "¿Cuánto puede pedir prestado un cliente con rating B?",
    "¿Cuándo pasa un cliente a vigilancia especial?",
    "¿Cuál es el tipo de cambio euro/dólar?"
]

for pregunta in preguntas:
    print(f"\nP: {pregunta}")
    print(f"R: {chain_rag.invoke(pregunta)}")

# ─────────────────────────────────────────
# PARTE 4 — Chatbot RAG con historial
# ─────────────────────────────────────────

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder

prompt_chat = ChatPromptTemplate.from_messages([
    ("system", """Eres un asistente experto en normativa bancaria.
Usa el contexto proporcionado para responder. Puedes razonar e inferir conclusiones.
Si la información no está en el contexto, di: 'No encuentro esa información en la normativa.'"""),
    MessagesPlaceholder(variable_name="historial"),
    ("human", "Contexto:\n{contexto}\n\nPregunta: {pregunta}")
])

chain_chat = prompt_chat | llm | StrOutputParser()

sesiones = {}
def get_session(session_id):
    if session_id not in sesiones:
        sesiones[session_id] = ChatMessageHistory()
    return sesiones[session_id]

chain_con_historial = RunnableWithMessageHistory(
    chain_chat,
    get_session,
    input_messages_key="pregunta",
    history_messages_key="historial"
)

def preguntar(pregunta, session_id="default"):
    contexto = formatear_docs(retriever_mejorado.invoke(pregunta))
    return chain_con_historial.invoke(
        {"contexto": contexto, "pregunta": pregunta},
        config={"configurable": {"session_id": session_id}}
    )

print("\n--- Chatbot RAG con historial ---")
print(preguntar("¿Qué hago con un cliente que tiene PD de 0.15?"))
print(preguntar("¿Y si además tiene 3 impagos este año?"))
print(preguntar("¿Puede ese cliente solicitar un nuevo préstamo?"))
