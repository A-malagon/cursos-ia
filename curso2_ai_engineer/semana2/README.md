# Semana 2 — Agentes

> "De chains lineales a agentes que razonan y deciden"

## Días

| Día | Tema | Estado |
|-----|------|--------|
| 7  | Agentes: ReAct pattern a mano | ✅ |
| 8  | LangChain Agents: create_react_agent, AgentExecutor | ⬜ |
| 9  | Agente con RAG + tools combinadas | ⬜ |
| 10 | Multi-agente: coordinador + agentes especializados | ⬜ |

**Checkpoint:** Agente bancario que razona, consulta herramientas en múltiples pasos y combina RAG con tools estructuradas.

---

## Día 7 — Agentes: ReAct pattern a mano

### Temas: qué es un agente, patrón ReAct, tools, bucle de razonamiento

---

### ¿Qué es un agente?

En los días anteriores todas las chains eran **lineales**: pregunta → retrieval → prompt → LLM → respuesta. Siempre el mismo flujo, controlado por ti.

Un **agente** es diferente: el LLM **decide** qué hacer en cada paso. Tiene herramientas disponibles (funciones, APIs, buscadores...) y elige cuándo y cuáles usar basándose en la pregunta.

---

### El patrón ReAct (Reason + Act)

El agente sigue un bucle de razonamiento:

```
Thought:      "Necesito saber el riesgo del cliente García"
Action:       obtener_cliente(nombre="García")
Observation:  {"PD": 0.15, "impagos": 3}

Thought:      "PD 0.15 > 0.10, necesito también la normativa"
Action:       consultar_normativa(tema="aprobacion")
Observation:  "PD > 10%: comité de riesgos"

Thought:      "3 impagos, consulto también vigilancia"
Action:       consultar_normativa(tema="vigilancia")
Observation:  "Más de 2 impagos → vigilancia especial"

Thought:      "Ya tengo todo para responder"
Final Answer: "García requiere comité de riesgos y está en vigilancia especial..."
```

El LLM **razona** (Thought) y luego **actúa** (Action). Puede dar varios pasos antes de responder. Tú no controlas cuántos pasos da — el LLM decide cuándo tiene suficiente información.

---

### Diferencia clave: function calling vs agente

| | Function calling (Días 2-6) | Agente (Día 7+) |
|---|---|---|
| ¿Quién controla el flujo? | Tú | El LLM |
| ¿Cuántas llamadas al LLM? | Siempre 1-2 fijas | 1 o más, según necesite |
| ¿Puede encadenar acciones? | No | Sí |
| ¿Decide qué tools usar? | No (tú decides) | Sí |

En function calling: `if mensaje.tool_calls → ejecutar → una llamada más → respuesta`. Flujo fijo.

En agente: bucle hasta que el LLM decide que ya tiene todo lo necesario.

---

### RAG vs Tools — cuándo usar cada uno

```
Documentos extensos (PDFs, contratos, normativas largas)
→ No puedes indexar todo en el prompt
→ RAG: embedding + vectorstore + retrieval por similitud

Datos estructurados (clientes, productos, operaciones)
→ Están en una BD con campos concretos
→ Tool: función que consulta la BD y devuelve el registro
```

En producción real:
```
Datos de clientes    → PostgreSQL         → tool
Normativa bancaria   → PDFs en ChromaDB   → tool que hace RAG internamente
Tipos de cambio      → API externa        → tool
Cálculos             → función Python     → tool
```

El Día 7 usa normativa hardcodeada en un diccionario porque la normativa de prueba es pequeña. En el Día 9 esa función consultará ChromaDB con RAG real.

---

### Estructura del agente

**3 tools:**
- `obtener_cliente(nombre)` — devuelve PD, impagos, rating, deuda
- `consultar_normativa(tema)` — busca en la normativa (aprobacion, vigilancia, limites, provisiones)
- `calcular_provision(pd, lgd, ead)` — calcula PE = PD × LGD × EAD

**El bucle ReAct:**
```python
for paso in range(max_pasos):
    respuesta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=mensajes,
        tools=tools,
        tool_choice="auto"   # el LLM decide si llama tool o responde
    )
    mensaje = respuesta.choices[0].message

    if mensaje.tool_calls:
        # ejecutar tools y añadir resultados al historial
        # el LLM vuelve a razonar con la nueva información
    else:
        # el LLM ya tiene todo → respuesta final
        return mensaje.content
```

`tool_choice="auto"` es el mismo mecanismo que en function calling. La diferencia es que aquí está dentro de un bucle que se repite hasta que el LLM decide que ya tiene suficiente información.

---

### El schema de tools es crítico

El LLM decide qué tool llamar basándose **únicamente en el nombre y la descripción**. Si la descripción es vaga, el LLM se la salta.

**Problema encontrado:** con la descripción genérica de `consultar_normativa`, el LLM no la llamaba para `"¿Qué hago con García?"`. Respondía con conocimiento general en lugar de citar la normativa interna.

**Causa:** la descripción decía simplemente "busca información en la normativa bancaria según el tema" — el LLM no entendía que era **obligatorio** consultarla para dar una recomendación correcta.

**Solución:** descripción explícita que indica cuándo debe usarse:
```python
"description": "Consulta la normativa bancaria interna. DEBES llamar esta tool siempre que 
necesites decidir qué acción tomar con un cliente — aprobación de préstamos, vigilancia especial, 
límites de exposición o cálculo de provisiones. Sin esta tool no puedes dar una recomendación correcta."
```

**Regla:** las descripciones de tools son prompting. Cuanto más claro seas sobre cuándo usarlas, más fiable es el agente.

---

### Resultado del agente con las 4 preguntas de prueba

**"¿Cuáles son los datos del cliente Martínez?"**
- Paso 1: `obtener_cliente("Martínez")` → datos
- Paso 2: respuesta final
- 1 tool, flujo simple

**"¿Qué hago con el cliente García?"**
- Paso 1: `obtener_cliente("García")` → PD 0.15, 3 impagos
- Paso 2: `consultar_normativa("vigilancia")` → vigilancia especial
- Paso 3: `calcular_provision(0.15, 0.45, 50000)` → 3.375€
- Paso 4: respuesta final citando normativa
- 3 tools encadenadas, el LLM decidió el orden y qué consultar

**"¿Cuál es la provisión esperada para García asumiendo LGD de 0.45?"**
- Paso 1: `obtener_cliente("García")` → PD y deuda
- Paso 2: `calcular_provision(0.15, 0.45, 50000)` → 3.375€
- Paso 3: respuesta final
- No llamó `consultar_normativa` porque no la necesitaba — el agente es eficiente

**"¿Cuál es el tipo de cambio euro/dólar?"**
- Paso 1: respuesta directa — no tiene tool para esto
- El LLM responde honestamente que no tiene acceso a esa información

---

### Código del Día 7

- [dia7/dia7_agente_react.py](dia7/dia7_agente_react.py) — agente ReAct completo con 3 tools

---
