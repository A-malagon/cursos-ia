# Día 1 — Primera llamada a OpenAI API
# Curso 2: AI Engineer

from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv(r"C:\Users\50051676\Desktop\Curso_MLOps\curso2_ai_engineer\.env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

respuesta = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Eres un asistente experto en MLOps."},
        {"role": "user", "content": "¿Qué es un embedding en 2 frases?"}
    ],
    temperature=0
)

print(respuesta.choices[0].message.content)
print(f"\nTokens usados: {respuesta.usage.prompt_tokens} entrada + {respuesta.usage.completion_tokens} salida = {respuesta.usage.total_tokens} total")


# --- Embeddings ---
embedding_respuesta = client.embeddings.create(
    model="text-embedding-ada-002",
    input="El modelo de scoring de crédito predice el riesgo de impago"
)

vector = embedding_respuesta.data[0].embedding
print(f"\nDimensiones del vector: {len(vector)}")
print(f"Primeros 5 valores: {vector[:5]}")

def similitud(texto1, texto2):
    r1 = client.embeddings.create(model="text-embedding-ada-002", input=texto1)
    r2 = client.embeddings.create(model="text-embedding-ada-002", input=texto2)
    v1 = np.array(r1.data[0].embedding)
    v2 = np.array(r2.data[0].embedding)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print(f"\nSimilitud 'impago' vs 'riesgo crédito': {similitud('impago', 'riesgo de crédito'):.4f}")
print(f"Similitud 'impago' vs 'receta de cocina': {similitud('impago', 'receta de cocina'):.4f}")
