# Día 4 — LangChain core
# Curso 2: AI Engineer
# Temas: chains, prompts, parsers, memory

from dotenv import load_dotenv
import os

load_dotenv(r"C:\Users\50051676\Desktop\Curso_MLOps\curso2_ai_engineer\.env")

# ─────────────────────────────────────────
# PARTE 1 — ChatModel + PromptTemplate
# ─────────────────────────────────────────

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# El modelo — equivale a tu client.chat.completions.create
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Plantilla reutilizable con variables
prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un analista de riesgo financiero de {banco}."),
    ("user", "Analiza el riesgo del cliente con PD={pd} y deuda de {deuda}€")
])

# Rellenar la plantilla y llamar al modelo
chain = prompt | llm
respuesta = chain.invoke({
    "banco": "Santander",
    "pd": 0.15,
    "deuda": 50000
})

print("--- ChatModel + PromptTemplate ---")
print(respuesta.content)
# ─────────────────────────────────────────
# PARTE 2 — Chains (encadenar pasos)
# ─────────────────────────────────────────
from langchain_core.output_parsers import StrOutputParser

# Chain 1: analizar riesgo
prompt_analisis = ChatPromptTemplate.from_messages([
    ("system", "Eres un analista de riesgo. Sé conciso, máximo 2 frases."),
    ("user", "Analiza el riesgo: PD={pd}, deuda={deuda}€")
])

# Chain 2: recomendar acción basándose en el análisis anterior
prompt_accion = ChatPromptTemplate.from_messages([
    ("system", "Eres un director de riesgos. Da una acción concreta en 1 frase."),
    ("user", "Basándote en este análisis: {analisis}\n¿Qué acción recomiendas?")
])

# Encadenar: análisis → acción
chain_completa = (
    prompt_analisis
    | llm
    | StrOutputParser()
    | (lambda analisis: {"analisis": analisis})
    | prompt_accion
    | llm
    | StrOutputParser()
)

resultado = chain_completa.invoke({"pd": 0.15, "deuda": 50000})
print("\n--- Chain encadenada ---")
print(resultado)


# ─────────────────────────────────────────
# PARTE 3 — Output parsers
# ─────────────────────────────────────────

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

parser = JsonOutputParser()

prompt_json = PromptTemplate(
    template="""Analiza este cliente y devuelve un JSON con exactamente estos campos:
- riesgo: "bajo", "medio" o "alto"
- pd_porcentaje: número entre 0 y 100
- accion: string con la acción recomendada
- aprobar: true o false

Cliente: PD={pd}, deuda={deuda}€

{format_instructions}""",
    input_variables=["pd", "deuda"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain_json = prompt_json | llm | parser

resultado_json = chain_json.invoke({"pd": 0.15, "deuda": 50000})
print("\n--- Output parser JSON ---")
print(type(resultado_json))
print(resultado_json)
print(f"¿Aprobar? {resultado_json['aprobar']}")

# ─────────────────────────────────────────
# PARTE 4 — Memory (historial automático)
# ─────────────────────────────────────────

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

prompt_memory = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente de riesgo financiero. Sé conciso."),
    ("placeholder", "{historial}"),
    ("user", "{input}")
])

chain_memory = prompt_memory | llm | StrOutputParser()

# Almacén de sesiones — en producción sería Redis o una BD
sesiones = {}

def get_session(session_id):
    if session_id not in sesiones:
        sesiones[session_id] = ChatMessageHistory()
    return sesiones[session_id]

chain_con_memoria = RunnableWithMessageHistory(
    chain_memory,
    get_session,
    input_messages_key="input",
    history_messages_key="historial"
)

config = {"configurable": {"session_id": "sesion_001"}}

print("\n--- Memory ---")
print(chain_con_memoria.invoke({"input": "¿Qué es la PD?"}, config=config))
print(chain_con_memoria.invoke({"input": "¿Y cómo se calcula?"}, config=config))
print(chain_con_memoria.invoke({"input": "¿Cuál es un valor alto?"}, config=config))
