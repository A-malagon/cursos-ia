# Día 2 — OpenAI API en profundidad
# Curso 2: AI Engineer
# Temas: historial de conversación, system prompt avanzado, function calling

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(r"C:\Users\50051676\Desktop\Curso_MLOps\curso2_ai_engineer\.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─────────────────────────────────────────
# PARTE 1 — Conversación con historial
# ─────────────────────────────────────────
historial = [
    {"role": "system", "content": "Eres un asistente experto en riesgo financiero. Responde de forma concisa."}
]

def chat(mensaje):
    historial.append({"role": "user", "content": mensaje})
    respuesta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=historial,
        temperature=0.7
    )
    contenido = respuesta.choices[0].message.content
    historial.append({"role": "assistant", "content": contenido})
    return contenido

print(chat("¿Qué es el riesgo de crédito?"))
print(chat("¿Y cómo se mide?"))
print(chat("¿Cuál de esas métricas es la más usada en banca?"))

# ─────────────────────────────────────────
# PARTE 2 — System prompt avanzado
# ─────────────────────────────────────────

respuesta_estricta = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": """Eres un asistente de riesgo financiero para un banco español.
Reglas estrictas:
- Responde SIEMPRE en español
- Responde SOLO sobre temas de riesgo financiero y banca
- Si te preguntan algo fuera de ese ámbito, di exactamente: "Solo puedo ayudarte con temas de riesgo financiero."
- Máximo 3 frases por respuesta
- Nunca uses bullet points, solo párrafos"""},
        {"role": "user", "content": "¿Cuál es la capital de Francia?"}
    ],
    temperature=0
)

print("\n--- System prompt estricto ---")
print(respuesta_estricta.choices[0].message.content)


# ─────────────────────────────────────────
# PARTE 3 — Function calling
# ─────────────────────────────────────────

import json

# Función real de tu código
def obtener_riesgo_cliente(cliente_id: str) -> dict:
    # Simulamos una consulta a base de datos
    datos = {
        "C001": {"nombre": "García", "pd": 0.15, "rating": "B"},
        "C002": {"nombre": "Martínez", "pd": 0.03, "rating": "AA"},
    }
    return datos.get(cliente_id, {"error": "cliente no encontrado"})

# Descripción de la función para el LLM
herramientas = [
    {
        "type": "function",
        "function": {
            "name": "obtener_riesgo_cliente",
            "description": "Obtiene el perfil de riesgo de un cliente dado su ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "cliente_id": {
                        "type": "string",
                        "description": "El ID del cliente, ej: C001"
                    }
                },
                "required": ["cliente_id"]
            }
        }
    }
]

# El LLM decide si necesita llamar a la función
respuesta_fc = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Eres un asistente de riesgo financiero."},
        {"role": "user", "content": "¿Cuál es el riesgo del cliente C001?"}
    ],
    tools=herramientas,
    tool_choice="auto"
)

mensaje = respuesta_fc.choices[0].message

# Si el LLM decidió llamar a una función
if mensaje.tool_calls:
    tool_call = mensaje.tool_calls[0]
    argumentos = json.loads(tool_call.function.arguments)
    print(f"\n--- Function calling ---")
    print(f"El LLM quiere llamar a: {tool_call.function.name}")
    print(f"Con argumentos: {argumentos}")
    
    # Ejecutamos la función nosotros
    resultado = obtener_riesgo_cliente(**argumentos)
    print(f"Resultado de la función: {resultado}")
    
    # Mandamos el resultado de vuelta al LLM para que responda
    respuesta_final = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un asistente de riesgo financiero."},
            {"role": "user", "content": "¿Cuál es el riesgo del cliente C001?"},
            mensaje,
            {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(resultado)}
        ],
        temperature=0
    )
    print(f"\nRespuesta final: {respuesta_final.choices[0].message.content}")
