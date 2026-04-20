# Día 7 — Agentes: ReAct pattern a mano
# Curso 2: AI Engineer
# Implementamos el bucle ReAct sin frameworks para entender qué hace LangChain por debajo

from dotenv import load_dotenv
import os
import json

load_dotenv(r"C:\Users\50051676\Desktop\Curso_MLOps\curso2_ai_engineer\.env")

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─────────────────────────────────────────
# PARTE 1 — Definir las herramientas (tools)
# ─────────────────────────────────────────

# Base de datos simulada de clientes
CLIENTES = {
    "García":    {"pd": 0.15, "impagos": 3, "rating": "B",  "deuda": 50000},
    "Martínez":  {"pd": 0.02, "impagos": 0, "rating": "AA", "deuda": 10000},
    "Rodríguez": {"pd": 0.08, "impagos": 1, "rating": "BB", "deuda": 80000},
}

def obtener_cliente(nombre: str) -> dict:
    """Devuelve los datos de un cliente por nombre."""
    if nombre in CLIENTES:
        return CLIENTES[nombre]
    return {"error": f"Cliente {nombre} no encontrado"}

def consultar_normativa(tema: str) -> str:
    """Busca información en la normativa bancaria según el tema."""
    normativa = {
        "aprobacion": "PD < 5%: aprobación automática. PD 5-10%: validación gestor. PD > 10%: comité de riesgos.",
        "vigilancia":  "Más de 2 impagos en 12 meses → vigilancia especial. No pueden solicitar nuevos préstamos.",
        "limites":     "Límite máximo 500.000€. Rating inferior a BB: límite reducido al 50% (250.000€).",
        "provisiones": "Provisión = PD × LGD × EAD. Basel III exige 8% sobre activos ponderados por riesgo.",
    }
    tema_lower = tema.lower()
    for clave, texto in normativa.items():
        if clave in tema_lower:
            return texto
    return "No se encontró normativa específica para ese tema."

def calcular_provision(pd: float, lgd: float, ead: float) -> dict:
    """Calcula la provisión esperada: PE = PD × LGD × EAD."""
    pe = pd * lgd * ead
    return {"pd": pd, "lgd": lgd, "ead": ead, "provision_esperada": round(pe, 2)}


# ─────────────────────────────────────────
# PARTE 2 — Schema de tools para OpenAI
# ─────────────────────────────────────────

tools = [
    {
        "type": "function",
        "function": {
            "name": "obtener_cliente",
            "description": "Obtiene los datos de un cliente bancario por su nombre: PD, impagos, rating y deuda.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nombre": {"type": "string", "description": "Nombre del cliente, ej: García"}
                },
                "required": ["nombre"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_normativa",
            "description": "Consulta la normativa bancaria interna. DEBES llamar esta tool siempre que necesites decidir qué acción tomar con un cliente — aprobación de préstamos, vigilancia especial, límites de exposición o cálculo de provisiones. Sin esta tool no puedes dar una recomendación correcta.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tema": {"type": "string", "description": "Tema a consultar, ej: aprobacion, vigilancia"}
                },
                "required": ["tema"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calcular_provision",
            "description": "Calcula la provisión esperada (PE = PD × LGD × EAD) para un cliente.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pd":  {"type": "number", "description": "Probabilidad de Default, ej: 0.15"},
                    "lgd": {"type": "number", "description": "Loss Given Default, ej: 0.45"},
                    "ead": {"type": "number", "description": "Exposure at Default en euros, ej: 50000"}
                },
                "required": ["pd", "lgd", "ead"]
            }
        }
    }
]

# Mapa para ejecutar la función por nombre
TOOLS_MAP = {
    "obtener_cliente":   obtener_cliente,
    "consultar_normativa": consultar_normativa,
    "calcular_provision": calcular_provision,
}


# ─────────────────────────────────────────
# PARTE 3 — El bucle ReAct
# ─────────────────────────────────────────

def ejecutar_agente(pregunta: str, max_pasos: int = 5):
    print(f"\n{'='*60}")
    print(f"PREGUNTA: {pregunta}")
    print(f"{'='*60}")

    mensajes = [
        {"role": "system", "content": """Eres un agente de riesgo bancario.
Tienes herramientas para consultar datos de clientes, normativa y calcular provisiones.
Usa las herramientas necesarias para responder con precisión.
Cuando tengas toda la información, da una respuesta final clara."""},
        {"role": "user", "content": pregunta}
    ]

    for paso in range(max_pasos):
        print(f"\n--- Paso {paso + 1} ---")

        respuesta = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=mensajes,
            tools=tools,
            tool_choice="auto"   # el LLM decide si llama tool o responde
        )

        mensaje = respuesta.choices[0].message

        # ¿El LLM quiere llamar herramientas?
        if mensaje.tool_calls:
            mensajes.append(mensaje)  # añadimos el mensaje del LLM al historial

            for tool_call in mensaje.tool_calls:
                nombre = tool_call.function.name
                argumentos = json.loads(tool_call.function.arguments)

                print(f"  → Tool: {nombre}({argumentos})")

                # Ejecutar la función real
                resultado = TOOLS_MAP[nombre](**argumentos)
                print(f"  ← Resultado: {resultado}")

                # Devolver resultado al LLM
                mensajes.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(resultado, ensure_ascii=False)
                })

        else:
            # El LLM ya tiene todo — respuesta final
            print(f"\nRESPUESTA FINAL:\n{mensaje.content}")
            return mensaje.content

    return "Máximo de pasos alcanzado sin respuesta final."


# ─────────────────────────────────────────
# PARTE 4 — Probar el agente
# ─────────────────────────────────────────

# Pregunta simple — 1 tool
ejecutar_agente("¿Cuáles son los datos del cliente Martínez?")

# Pregunta que requiere 2 tools — datos + normativa
ejecutar_agente("¿Qué hago con el cliente García?")

# Pregunta que requiere 3 tools — datos + normativa + cálculo
ejecutar_agente("¿Cuál es la provisión esperada para García asumiendo LGD de 0.45?")

# Pregunta fuera del alcance — el agente no tiene tool para esto
ejecutar_agente("¿Cuál es el tipo de cambio euro/dólar?")
