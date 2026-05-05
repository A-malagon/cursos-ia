# Día 8 — LangChain Agents
# Curso 2: AI Engineer
# Lo mismo que el Día 7 pero usando el framework LangChain

from dotenv import load_dotenv
import os

load_dotenv(r"C:\Users\50051676\Desktop\Curso_MLOps\curso2_ai_engineer\.env")


# ─────────────────────────────────────────
# PARTE 1 — Definir tools con @tool decorator
# ─────────────────────────────────────────

from langchain_core.tools import tool

CLIENTES = {
    "García":    {"pd": 0.15, "impagos": 3, "rating": "B",  "deuda": 50000},
    "Martínez":  {"pd": 0.02, "impagos": 0, "rating": "AA", "deuda": 10000},
    "Rodríguez": {"pd": 0.08, "impagos": 1, "rating": "BB", "deuda": 80000},
}

@tool
def obtener_cliente(nombre: str) -> dict:
    """Obtiene los datos de un cliente bancario: PD, impagos, rating y deuda."""
    if nombre in CLIENTES:
        return CLIENTES[nombre]
    return {"error": f"Cliente {nombre} no encontrado"}

@tool
def consultar_normativa(tema: str) -> str:
    """Consulta la normativa bancaria interna. DEBES llamar esta tool siempre que
    necesites decidir qué acción tomar con un cliente — aprobación de préstamos,
    vigilancia especial, límites de exposición o cálculo de provisiones.
    Temas disponibles: aprobacion, vigilancia, limites, provisiones."""
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

@tool
def calcular_provision(pd: float, lgd: float, ead: float) -> dict:
    """Calcula la provisión esperada (PE = PD × LGD × EAD) para un cliente."""
    pe = pd * lgd * ead
    return {"pd": pd, "lgd": lgd, "ead": ead, "provision_esperada": round(pe, 2)}

tools = [obtener_cliente, consultar_normativa, calcular_provision]
