# Curso 2 — AI Engineer Intensivo

**Perfil:** DevOps/Cloud Engineer con nociones básicas de ChatGPT
**Objetivo:** Dominar LangChain + Semantic Kernel + Azure AI Foundry + Agentes LLM
**Ritmo:** 5-6h/día · Formato mixto: teoría + práctica + proyectos + tests
**Estado:** ⬜ Pendiente — empezar tras completar Semana 1 del Curso MLOps

---

## Visión general del stack

```
LLMs (GPT-4o, Claude...)
        ↓
Orquestación (LangChain / Semantic Kernel)
        ↓
Agentes autónomos (razonamiento + planificación + herramientas)
        ↓
Azure AI Foundry (despliegue + gestión + observabilidad)
        ↓
MLflow (tracking de experimentos y versiones)
```

---

## Instalaciones necesarias

```bash
pip install langchain langchain-openai langchain-community
pip install semantic-kernel
pip install chromadb faiss-cpu
pip install langsmith openai
```

Azure AI Foundry se gestiona desde el portal web, sin instalación local.

---

## Progreso

| Semana | Día | Tema | Estado |
|--------|-----|------|--------|
| 1 | 1 | LLMs: tokens, contexto, temperatura, embeddings | ⬜ |
| 1 | 2 | OpenAI API / Azure OpenAI: llamadas, roles, function calling | ⬜ |
| 1 | 3 | Prompting avanzado: few-shot, chain-of-thought, RAG conceptual | ⬜ |
| 1 | 4 | LangChain core: chains, prompts, parsers, memory | ⬜ |
| 1 | 5 | LangChain avanzado: retrievers, vectorstores, FAISS, ChromaDB | ⬜ |
| 1 | 6 | Proyecto: Chatbot RAG con memoria y búsqueda en documentos | ⬜ |
| 1 | 7 | Test semana 1 + repaso | ⬜ |
| 2 | 8 | Agentes: ReAct, planificación, ciclo observe-think-act | ⬜ |
| 2 | 9 | LangChain Agents: tools, toolkits, AgentExecutor | ⬜ |
| 2 | 10 | Semantic Kernel: kernels, plugins, functions, memoria semántica | ⬜ |
| 2 | 11 | Semantic Kernel avanzado: planners, orquestación multi-step | ⬜ |
| 2 | 12 | Multi-agent systems: coordinación, LangGraph intro | ⬜ |
| 2 | 13 | Proyecto: Agente financiero con herramientas externas | ⬜ |
| 2 | 14 | Test semana 2 + repaso | ⬜ |
| 3 | 15 | Azure AI Foundry: arquitectura, hubs, proyectos, modelos | ⬜ |
| 3 | 16 | Despliegue de modelos: endpoints, versiones, cuotas | ⬜ |
| 3 | 17 | Azure OpenAI Service vs AI Foundry: cuándo usar cada uno | ⬜ |
| 3 | 18 | MLflow en contexto LLM: tracking de prompts, evaluación | ⬜ |
| 3 | 19 | Evaluación de LLMs: métricas, LLM-as-judge | ⬜ |
| 3 | 20 | Proyecto: Agente financiero desplegado en Azure AI Foundry | ⬜ |
| 3 | 21 | Test semana 3 + repaso | ⬜ |
| 4 | 22 | Arquitecturas agentic: patrones, guardrails, manejo de errores | ⬜ |
| 4 | 23 | Seguridad en LLMs: prompt injection, jailbreaking, datos sensibles | ⬜ |
| 4 | 24 | Observabilidad: trazas de agentes, LangSmith, Azure Monitor | ⬜ |
| 4 | 25 | Escalabilidad: caching, rate limiting, costes de tokens | ⬜ |
| 4 | 26 | Integración con sistemas financieros: APIs REST, compliance | ⬜ |
| 4 | 27-28 | Proyecto Final: Plataforma de asesoramiento financiero con agentes | ⬜ |
| 4 | 29 | Test final + simulacro entrevista técnica | ⬜ |
| 4 | 30 | Repaso + preparación presentación | ⬜ |

---

## Proyecto Final — Plataforma de Asesoramiento con Agentes

```
Usuario hace consulta financiera
        ↓
Agente Orquestador (Semantic Kernel / LangChain)
        ↓
  ┌─────────────────────────────────┐
  │  Agente RAG        Agente Datos │
  │  (documentos)   (APIs mercado)  │
  └─────────────────────────────────┘
        ↓
Razonamiento multi-step + generación de informe
        ↓
Azure AI Foundry (despliegue)
        ↓
MLflow (tracking) + Azure Monitor (observabilidad)
```

---

## Stack de herramientas

| Categoría | Herramienta |
|-----------|-------------|
| LLMs | Azure OpenAI (GPT-4o), modelos open source |
| Orquestación | LangChain, LangGraph, Semantic Kernel |
| Vectorstore | ChromaDB, FAISS |
| Despliegue | Azure AI Foundry |
| Tracking | MLflow, LangSmith |
| Observabilidad | Azure Monitor, trazas de agentes |
| Lenguaje | Python 3.11 |

---

## Semanas

- [Semana 1 — Fundamentos LLM](./semana1/README.md)
- [Semana 2 — Agentes Autónomos](./semana2/README.md)
- [Semana 3 — Azure AI Foundry + Despliegue](./semana3/README.md)
- [Semana 4 — Arquitectura + Seguridad + Proyecto Final](./semana4/README.md)
