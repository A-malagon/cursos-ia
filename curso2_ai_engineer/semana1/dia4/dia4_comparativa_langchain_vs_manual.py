# Día 4 — Comparativa: LangChain vs código manual
# Curso 2: AI Engineer
#
# Este script reimplementa los ejercicios de los días 1, 2 y 3
# usando LangChain donde es posible, y explica cuándo NO usar LangChain.

from dotenv import load_dotenv
import os

load_dotenv(r"C:\Users\50051676\Desktop\Curso_MLOps\curso2_ai_engineer\.env")


# ══════════════════════════════════════════════════════════════════
# DÍA 1 — Embeddings y similitud coseno
# ══════════════════════════════════════════════════════════════════
# ✅ LangChain SÍ puede generar embeddings
# ❌ LangChain NO tiene utilidad para similitud coseno — se hace igual con numpy

print("=" * 60)
print("DÍA 1 — Embeddings con LangChain")
print("=" * 60)

from langchain_openai import OpenAIEmbeddings
import numpy as np

# MANUAL (Día 1):
# client.embeddings.create(model="text-embedding-ada-002", input=texto)

# CON LANGCHAIN:
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")

vector1 = embeddings_model.embed_query("impago de cliente")
vector2 = embeddings_model.embed_query("riesgo de crédito")
vector3 = embeddings_model.embed_query("receta de cocina")

def similitud_coseno(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print(f"Dimensiones del vector: {len(vector1)}")
print(f"Similitud 'impago' vs 'riesgo crédito': {similitud_coseno(vector1, vector2):.4f}")
print(f"Similitud 'impago' vs 'receta cocina':  {similitud_coseno(vector1, vector3):.4f}")

# DIFERENCIA:
# Manual:    client.embeddings.create(...).data[0].embedding
# LangChain: embeddings_model.embed_query(texto)
# Resultado idéntico. LangChain es algo más limpio pero no aporta nada extra aquí.
# Para similitud coseno: igual en los dos casos — numpy siempre.


# ══════════════════════════════════════════════════════════════════
# DÍA 2 — Historial de conversación
# ══════════════════════════════════════════════════════════════════
# ✅ LangChain SÍ mejora esto — gestiona el historial automáticamente

print("\n" + "=" * 60)
print("DÍA 2 — Historial de conversación con LangChain")
print("=" * 60)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# MANUAL (Día 2):
# historial = [{"role": "system", ...}]
# historial.append({"role": "user", ...})
# historial.append({"role": "assistant", ...})
# → Tú gestionas la lista manualmente

# CON LANGCHAIN:
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

prompt_historial = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente experto en riesgo financiero. Responde de forma concisa."),
    MessagesPlaceholder(variable_name="historial"),
    ("human", "{input}")
])

chain_historial = prompt_historial | llm | StrOutputParser()

sesiones = {}
def get_session(session_id):
    if session_id not in sesiones:
        sesiones[session_id] = ChatMessageHistory()
    return sesiones[session_id]

chain_con_memoria = RunnableWithMessageHistory(
    chain_historial,
    get_session,
    input_messages_key="input",
    history_messages_key="historial"
)

config = {"configurable": {"session_id": "dia2_test"}}

print(chain_con_memoria.invoke({"input": "¿Qué es el riesgo de crédito?"}, config=config))
print(chain_con_memoria.invoke({"input": "¿Y cómo se mide?"}, config=config))
print(chain_con_memoria.invoke({"input": "¿Cuál es la métrica más usada en banca?"}, config=config))

# DIFERENCIA:
# Manual:    tú mantienes y pasas la lista historial en cada llamada
# LangChain: RunnableWithMessageHistory lo hace automático por session_id
# VENTAJA LANGCHAIN: en producción con múltiples usuarios, cada uno tiene su sesión
# sin que tú gestiones nada. En manual tendrías que gestionar un dict de historiales.


# ══════════════════════════════════════════════════════════════════
# DÍA 2 — System prompt avanzado
# ══════════════════════════════════════════════════════════════════
# ⚠️ LangChain NO aporta nada especial aquí — es lo mismo

print("\n" + "=" * 60)
print("DÍA 2 — System prompt avanzado (igual en LangChain)")
print("=" * 60)

prompt_estricto = ChatPromptTemplate.from_messages([
    ("system", """Eres un asistente de riesgo financiero para un banco español.
Reglas estrictas:
- Responde SIEMPRE en español
- Responde SOLO sobre temas de riesgo financiero y banca
- Si te preguntan algo fuera de ese ámbito, di exactamente: "Solo puedo ayudarte con temas de riesgo financiero."
- Máximo 3 frases por respuesta
- Nunca uses bullet points, solo párrafos"""),
    ("human", "{input}")
])

llm_estricto = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain_estricto = prompt_estricto | llm_estricto | StrOutputParser()

print(chain_estricto.invoke({"input": "¿Cuál es la capital de Francia?"}))

# DIFERENCIA:
# Manual:    messages=[{"role": "system", "content": "..."}, {"role": "user", ...}]
# LangChain: ChatPromptTemplate.from_messages([("system", "..."), ("human", "...")])
# Resultado idéntico. Solo cambia la sintaxis.


# ══════════════════════════════════════════════════════════════════
# DÍA 2 — Function calling
# ══════════════════════════════════════════════════════════════════
# ⚠️ LangChain puede hacerlo pero añade complejidad sin aportar claridad
# Para function calling simple, el código manual del Día 2 es más claro

print("\n" + "=" * 60)
print("DÍA 2 — Function calling (mejor hacerlo manual)")
print("=" * 60)

# LangChain tiene .bind_tools() para esto, pero para 1-2 funciones
# el código manual es más legible y directo.
# Lo veremos en agentes (Semana 2) donde LangChain SÍ aporta mucho.

print("→ Function calling simple: mejor manual (ver dia2_openai_api.py)")
print("→ Function calling con múltiples herramientas: mejor LangChain Agents (Semana 2)")


# ══════════════════════════════════════════════════════════════════
# DÍA 3 — Few-shot prompting
# ══════════════════════════════════════════════════════════════════
# ✅ LangChain tiene FewShotChatMessagePromptTemplate — más limpio

print("\n" + "=" * 60)
print("DÍA 3 — Few-shot prompting con LangChain")
print("=" * 60)

from langchain_core.prompts import FewShotChatMessagePromptTemplate

ejemplos = [
    {"input": "Cliente: Juan, PD: 0.02, deuda: 5000€",
     "output": "RIESGO: BAJO | ACCIÓN: Aprobar automáticamente"},
    {"input": "Cliente: María, PD: 0.25, deuda: 50000€",
     "output": "RIESGO: ALTO | ACCIÓN: Revisión manual urgente"},
]

ejemplo_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=ejemplo_prompt,
    examples=ejemplos,
)

prompt_final = ChatPromptTemplate.from_messages([
    ("system", "Clasifica el riesgo de clientes bancarios."),
    few_shot_prompt,
    ("human", "{input}"),
])

llm_fs = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain_fs = prompt_final | llm_fs | StrOutputParser()

print(chain_fs.invoke({"input": "Cliente: Pedro, PD: 0.12, deuda: 20000€"}))

# DIFERENCIA:
# Manual:    añades los ejemplos como mensajes en la lista directamente
# LangChain: FewShotChatMessagePromptTemplate — más limpio cuando tienes
#            muchos ejemplos o los cargas dinámicamente desde una BD


# ══════════════════════════════════════════════════════════════════
# DÍA 3 — Chain-of-thought
# ══════════════════════════════════════════════════════════════════
# ⚠️ LangChain NO aporta nada especial — es solo un system prompt

print("\n" + "=" * 60)
print("DÍA 3 — Chain-of-thought (igual en LangChain)")
print("=" * 60)

prompt_cot = ChatPromptTemplate.from_messages([
    ("system", """Eres un analista de riesgo.
Antes de responder, razona paso a paso:
1. Calcula la pérdida esperada (PE = PD × LGD × EAD)
2. Evalúa si PE es aceptable para el banco (umbral: 5000€)
3. Da tu recomendación final"""),
    ("human", "{input}")
])

llm_cot = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain_cot = prompt_cot | llm_cot | StrOutputParser()

print(chain_cot.invoke({"input": "PD: 0.08, LGD: 0.45, EAD: 100000€. ¿Aprobar?"}))

# DIFERENCIA: ninguna relevante. CoT es una técnica de prompting,
# no una funcionalidad de LangChain. La sintaxis cambia, el resultado es igual.


# ══════════════════════════════════════════════════════════════════
# DÍA 3 — Mini RAG
# ══════════════════════════════════════════════════════════════════
# ✅ LangChain mejora MUCHO esto — abstrae embeddings, similitud y prompt

print("\n" + "=" * 60)
print("DÍA 3 — Mini RAG con LangChain")
print("=" * 60)

from langchain_core.runnables import RunnablePassthrough

documentos_texto = [
    "El cliente García tiene 3 impagos en 2023 y una PD de 0.15.",
    "La política del banco exige revisión manual para PD superiores a 0.10.",
    "El cliente Martínez tiene historial impecable y PD de 0.02.",
    "Los préstamos hipotecarios requieren garantía real como colateral.",
    "El límite de exposición por cliente es de 500.000€ según normativa interna."
]

# MANUAL (Día 3): calcular embeddings uno a uno, similitud coseno, argsort...
# CON LANGCHAIN: el retriever hace todo eso internamente

from langchain_core.vectorstores import InMemoryVectorStore

vectorstore = InMemoryVectorStore.from_texts(
    documentos_texto,
    embedding=embeddings_model
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_rag = ChatPromptTemplate.from_messages([
    ("system", "Responde SOLO con la información del contexto. Si no está, di que no tienes esa información."),
    ("human", "Contexto:\n{contexto}\n\nPregunta: {pregunta}")
])

llm_rag = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain_rag = (
    {"contexto": retriever | (lambda docs: "\n".join(d.page_content for d in docs)),
     "pregunta": RunnablePassthrough()}
    | prompt_rag
    | llm_rag
    | StrOutputParser()
)

print(chain_rag.invoke("¿Qué debo hacer con el cliente García?"))

# DIFERENCIA:
# Manual:    ~20 líneas (embeddings, similitud coseno, argsort, construir prompt)
# LangChain: ~5 líneas (vectorstore + retriever + chain)
# AQUÍ LANGCHAIN SÍ APORTA MUCHO — abstrae toda la lógica de retrieval


# ══════════════════════════════════════════════════════════════════
# RESUMEN: cuándo usar LangChain y cuándo no
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RESUMEN — LangChain vs Manual")
print("=" * 60)
print("""
✅ USA LANGCHAIN para:
   - Historial de conversación con múltiples sesiones
   - RAG con vectorstore (abstrae mucho código)
   - Few-shot con muchos ejemplos dinámicos
   - Agentes con múltiples herramientas (Semana 2)
   - Chains complejas con muchos pasos

⚠️ DA IGUAL (misma complejidad):
   - System prompt avanzado
   - Chain-of-thought
   - Embeddings simples

❌ MEJOR MANUAL para:
   - Function calling simple (1-2 funciones)
   - Sistemas críticos donde necesitas control total
   - Cuando el equipo no conoce LangChain y añade complejidad
   - Debug y producción donde LangChain añade capas de abstracción
""")
