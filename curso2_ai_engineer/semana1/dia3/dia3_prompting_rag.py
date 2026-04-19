# Día 3 — Prompting avanzado + RAG conceptual
# Curso 2: AI Engineer
# Temas: few-shot prompting, chain-of-thought, mini RAG sin vectorstore

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(r"C:\Users\50051676\Desktop\Curso_MLOps\curso2_ai_engineer\.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─────────────────────────────────────────
# PARTE 1 — Few-shot prompting
# ─────────────────────────────────────────
respuesta = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Clasifica el riesgo de clientes bancarios."},
        {"role": "user", "content": "Cliente: Juan, PD: 0.02, deuda: 5000€"},
        {"role": "assistant", "content": "RIESGO: BAJO | ACCIÓN: Aprobar automáticamente"},
        {"role": "user", "content": "Cliente: María, PD: 0.25, deuda: 50000€"},
        {"role": "assistant", "content": "RIESGO: ALTO | ACCIÓN: Revisión manual urgente"},
        {"role": "user", "content": "Cliente: Pedro, PD: 0.12, deuda: 20000€"},
    ],
    temperature=0
)

print("--- Few-shot ---")
print(respuesta.choices[0].message.content)

# ─────────────────────────────────────────
# PARTE 2 — Chain-of-thought
# ─────────────────────────────────────────

# Sin chain-of-thought
sin_cot = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Eres un analista de riesgo."},
        {"role": "user", "content": """Un cliente tiene:
- PD: 0.08
- LGD: 0.45  
- EAD: 100000€
¿Cuál es la pérdida esperada y deberías aprobar el préstamo?"""}
    ],
    temperature=0
)

# Con chain-of-thought
con_cot = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": """Eres un analista de riesgo.
Antes de responder, razona paso a paso:
1. Calcula la pérdida esperada (PE = PD × LGD × EAD)
2. Evalúa si PE es aceptable para el banco (umbral: 5000€)
3. Da tu recomendación final"""},
        {"role": "user", "content": """Un cliente tiene:
- PD: 0.08
- LGD: 0.45  
- EAD: 100000€
¿Cuál es la pérdida esperada y deberías aprobar el préstamo?"""}
    ],
    temperature=0
)

print("\n--- Sin chain-of-thought ---")
print(sin_cot.choices[0].message.content)
print("\n--- Con chain-of-thought ---")
print(con_cot.choices[0].message.content)


# ─────────────────────────────────────────
# PARTE 3 — Mini RAG sin vectorstore
# ─────────────────────────────────────────

import numpy as np

# Base de conocimiento — simula documentos indexados
documentos = [
    "El cliente García tiene 3 impagos en 2023 y una PD de 0.15.",
    "La política del banco exige revisión manual para PD superiores a 0.10.",
    "El cliente Martínez tiene historial impecable y PD de 0.02.",
    "Los préstamos hipotecarios requieren garantía real como colateral.",
    "El límite de exposición por cliente es de 500.000€ según normativa interna."
]

def get_embedding(texto):
    r = client.embeddings.create(model="text-embedding-ada-002", input=texto)
    return np.array(r.data[0].embedding)

def similitud_coseno(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Indexar documentos (fase previa — en producción esto se hace una vez)
print("\n--- Mini RAG ---")
print("Indexando documentos...")
embeddings_docs = [get_embedding(doc) for doc in documentos]

# Pregunta del usuario
pregunta = "¿Qué debo hacer con el cliente García?"
embedding_pregunta = get_embedding(pregunta)

# Buscar los 2 chunks más similares
similitudes = [similitud_coseno(embedding_pregunta, emb) for emb in embeddings_docs]
print("\nSimilitudes calculadas:")
for i, (doc, sim) in enumerate(zip(documentos, similitudes)):
    print(f"  [{i}] {sim:.4f} — {doc[:50]}...")
indices_top = np.argsort(similitudes)[-5:][::-1]
chunks_relevantes = [documentos[i] for i in indices_top]

print(f"Pregunta: {pregunta}")
print(f"Chunks recuperados:")
for i, chunk in enumerate(chunks_relevantes):
    print(f"  {i+1}. {chunk}")

# Construir el prompt RAG
respuesta_rag = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Responde SOLO con la información del contexto proporcionado. Si no está en el contexto, di que no tienes esa información."},
        {"role": "user", "content": f"Contexto:\n{chr(10).join(chunks_relevantes)}\n\nPregunta: {pregunta}"}
    ],
    temperature=0
)

print(f"\nRespuesta RAG: {respuesta_rag.choices[0].message.content}")