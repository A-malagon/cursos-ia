# Plan de Formación — AI Engineer + MLOps + Cloud Engineer

**Perfil:** DevOps/Cloud Engineer (AWS, Terraform, Jenkins, Ansible)
**Objetivo:** Preparación para roles MLOps Senior, AI Engineer y Cloud Engineer (KPMG)
**Ritmo:** 5-6h/día · Sesiones interactivas con Claude Code + documentación acumulativa

---

## Los tres cursos

| # | Curso | Objetivo | Estado |
|---|-------|----------|--------|
| 1 | [MLOps Intensivo](./curso1_mlops/semana1/README.md) | Docker + Kubernetes + AKS + Azure ML + MLflow | ⏳ En curso — S1D3 |
| 2 | [AI Engineer](./curso2_ai_engineer/README.md) | LangChain + Semantic Kernel + Azure AI Foundry + Agentes LLM | ⬜ Pendiente |
| 3 | [Cloud Engineer — KPMG Ready](./curso3_cloud_engineer/README.md) | Azure Core + Networking + Gobernanza + IaC + Multi-cloud | ⬜ Pendiente |

> Los cursos 1 y 2 se diseñaron para hacerse en paralelo (mañana/tarde).
> El curso 3 se añadió para complementar con lo que pide KPMG específicamente.

---

## Cómo retomar en una sesión nueva

1. Abre este README para ver el estado de cada curso
2. Ve al README de la semana en curso
3. Di a Claude: **"sigamos con el curso, estoy en [Curso X, Semana Y, Día Z]"**
4. Claude tiene toda la documentación acumulada y el contexto completo

---

## Estructura de carpetas

```
Curso_MLOps/
├── README.md                          ← este fichero, índice global
│
├── curso1_mlops/                      ← Curso 1: MLOps
│   ├── semana1/
│   │   ├── README.md                  ← teoría + ejercicios + respuestas
│   │   ├── dia2/                      ← Dockerfile, app.py, etc.
│   │   └── dia3/
│   ├── semana2/
│   ├── semana3/
│   └── semana4/
│
├── curso2_ai_engineer/                ← Curso 2: AI Engineer
│   ├── README.md
│   ├── semana1/
│   ├── semana2/
│   ├── semana3/
│   └── semana4/
│
└── curso3_cloud_engineer/             ← Curso 3: Cloud Engineer KPMG
    ├── README.md
    ├── semana1/
    ├── semana2/
    ├── semana3/
    └── semana4/
```

---

## Stack completo de herramientas (los 3 cursos)

| Categoría | Herramienta |
|-----------|-------------|
| Contenedores | Docker, Docker Compose |
| Orquestación | Kubernetes, Helm, minikube, AKS |
| IaC | Terraform (Azure + AWS provider), Bicep |
| CI/CD | GitHub Actions, Azure DevOps, Jenkins |
| MLOps core | MLflow, DVC, Airflow, Kubeflow (conceptual) |
| Model serving | FastAPI, Azure ML Pipelines |
| Data quality | Great Expectations, Evidently (drift) |
| LLMs | Azure OpenAI (GPT-4o), modelos open source |
| Embeddings | text-embedding-ada-002, sentence-transformers |
| Orquestación LLM | LangChain, LangGraph, Semantic Kernel |
| Vectorstore | ChromaDB, FAISS, Milvus |
| Despliegue IA | Azure AI Foundry |
| Tracking | MLflow, LangSmith |
| Observabilidad | Prometheus, Grafana, Evidently, Azure Monitor, Log Analytics |
| Seguridad | Microsoft Defender for Cloud, Key Vault, Zero Trust |
| Gobernanza | Azure Policy, Management Groups, Landing Zones |
| Multi-cloud | Azure, AWS (base previa), GCP (fundamentos) |
| Lenguaje | Python 3.11, PowerShell, Bash |

---

## Progreso detallado

### Curso 1 — MLOps

| Semana | Día | Tema | Estado |
|--------|-----|------|--------|
| 1 | 1 | Docker: conceptos, arquitectura, VM vs contenedor | ✅ |
| 1 | 2 | Dockerfile, imágenes, capas, build y optimización | ✅ |
| 1 | 3 | Docker Compose, redes, volúmenes, multi-container + **FastAPI model serving** | ⏳ |
| 1 | 4 | Azure fundamentos: Resource Groups, AKS, ACR, Azure ML | ⬜ |
| 1 | 5 | Azure CLI + Terraform provider Azure | ⬜ |
| 1 | 6 | Proyecto: modelo scikit-learn + FastAPI + Docker + MLflow local | ⬜ |
| 1 | 7 | Test semana 1 + **DVC básico** (versionar dataset y modelo) | ⬜ |
| 2 | 8 | Arquitectura K8s: Control Plane, Nodes, etcd, kubelet | ⬜ |
| 2 | 9 | Pods, ReplicaSets, Deployments, Services | ⬜ |
| 2 | 10 | ConfigMaps, Secrets, Variables de entorno | ⬜ |
| 2 | 11 | Ingress, LoadBalancer, NetworkPolicies | ⬜ |
| 2 | 12 | Helm: charts, values, releases | ⬜ |
| 2 | 13 | Proyecto: despliega modelo ML en K8s local con Helm | ⬜ |
| 2 | 14 | Test semana 2 + repaso | ⬜ |
| 3 | 15 | AKS: cluster, node pools, autoscaling, ACR | ⬜ |
| 3 | 15b | **Kubeflow** conceptual + **OpenShift AI** conceptual | ⬜ |
| 3 | 16 | Azure DevOps Pipelines vs GitHub Actions | ⬜ |
| 3 | 17 | CI/CD para modelos ML + **Great Expectations** (validación datos) | ⬜ |
| 3 | 17b | **Airflow**: DAGs, tasks, scheduling — pipeline datos ML | ⬜ |
| 3 | 18 | Azure ML: workspaces, compute, model registry | ⬜ |
| 3 | 19 | MLflow en Azure: tracking, registry, experimentos + **DVC avanzado** | ⬜ |
| 3 | 20 | Proyecto: Pipeline CI/CD completo con validación y tracking | ⬜ |
| 3 | 21 | Test semana 3 + repaso | ⬜ |
| 4 | 22 | Monitorización: data drift, model drift + **Evidently** | ⬜ |
| 4 | 23 | Prometheus + Grafana en AKS + dashboard Evidently integrado | ⬜ |
| 4 | 24 | Logging centralizado: Azure Monitor, Log Analytics | ⬜ |
| 4 | 25 | Gobierno ML: versionado, rollback, A/B testing, canary | ⬜ |
| 4 | 26 | IaC completo: Terraform para todo el stack | ⬜ |
| 4 | 27-28 | Proyecto Final: Plataforma MLOps end-to-end en Azure | ⬜ |
| 4 | 29 | Test final + simulacro entrevista técnica | ⬜ |
| 4 | 30 | Repaso + preparación presentación | ⬜ |

### Curso 2 — AI Engineer

| Semana | Día | Tema | Estado |
|--------|-----|------|--------|
| 1 | 1 | LLMs: tokens, contexto, temperatura, embeddings | ⬜ |
| 1 | 2 | OpenAI API / Azure OpenAI: llamadas, roles, function calling | ⬜ |
| 1 | 3 | Prompting avanzado: few-shot, chain-of-thought, RAG conceptual | ⬜ |
| 1 | 4 | LangChain core: chains, prompts, parsers, memory | ⬜ |
| 1 | 5 | LangChain avanzado: retrievers, vectorstores — **ChromaDB, FAISS y Milvus** | ⬜ |
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

### Curso 3 — Cloud Engineer KPMG

| Semana | Día | Tema | Estado |
|--------|-----|------|--------|
| 1 | 1 | Azure fundamentals: suscripciones, recursos, portal, CLI | ⬜ |
| 1 | 2 | Networking I: VNets, subnets, NSGs, peering | ⬜ |
| 1 | 3 | Networking II: VPN Gateway, ExpressRoute, Azure Firewall, Private Endpoints | ⬜ |
| 1 | 4 | Identidad: Azure AD, RBAC, Managed Identities, Key Vault | ⬜ |
| 1 | 5 | Storage: Blob, Files, redundancia, lifecycle policies | ⬜ |
| 1 | 6 | Proyecto: arquitectura hub-spoke con acceso seguro | ⬜ |
| 1 | 7 | Test semana 1 + repaso AZ-104 bloque redes | ⬜ |
| 2 | 8 | Gobernanza: Management Groups, subscriptions, Azure Policy | ⬜ |
| 2 | 9 | Landing Zones + Cloud Adoption Framework (CAF) | ⬜ |
| 2 | 10 | FinOps: Cost Management, budgets, tagging, optimización | ⬜ |
| 2 | 11 | Bicep: sintaxis, módulos, comparativa con Terraform | ⬜ |
| 2 | 12 | Terraform avanzado: módulos, remote state en Azure Storage, workspaces | ⬜ |
| 2 | 13 | Proyecto: Landing Zone completa con Terraform | ⬜ |
| 2 | 14 | Test semana 2 + repaso AZ-104 bloque gobernanza | ⬜ |
| 3 | 15 | HA y DR: Availability Zones, Azure Backup, Site Recovery | ⬜ |
| 3 | 16 | Azure SQL + Cosmos DB + PostgreSQL Flexible | ⬜ |
| 3 | 17 | Microsoft Defender for Cloud: postura de seguridad, alertas | ⬜ |
| 3 | 18 | Zero Trust en Azure: Conditional Access, PIM, Just-in-Time | ⬜ |
| 3 | 19 | Monitorización empresarial: Azure Monitor, Log Analytics, App Insights | ⬜ |
| 3 | 20 | Proyecto: arquitectura app bancaria con HA, DR y seguridad | ⬜ |
| 3 | 21 | Test semana 3 + repaso AZ-305 bloque diseño | ⬜ |
| 4 | 22 | GCP fundamentos: Compute Engine, GKE, IAM, Cloud Storage | ⬜ |
| 4 | 23 | Comparativa AWS vs Azure vs GCP para recomendar a clientes | ⬜ |
| 4 | 24 | Azure Well-Architected Framework: los 5 pilares aplicados | ⬜ |
| 4 | 25 | GitOps: ArgoCD o Flux en AKS para despliegue continuo | ⬜ |
| 4 | 26 | Documentación y presentación de arquitecturas para cliente | ⬜ |
| 4 | 27-28 | Proyecto Final: arquitectura empresarial para cliente ficticio de banca | ⬜ |
| 4 | 29 | Simulacro entrevista técnica KPMG + preguntas de arquitectura | ⬜ |
| 4 | 30 | Repaso de gaps + preparación presentación del proyecto | ⬜ |
