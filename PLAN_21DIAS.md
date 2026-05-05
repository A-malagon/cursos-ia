# Plan 21 días — KPMG Ready (4 mayo → 24 mayo 2026)

> **Inicio KPMG:** 25 mayo 2026 como Manager en Financial Services
> **Objetivo:** Cubrir los 3 cursos al máximo posible en 21 días reales a jornada completa
> **Ritmo:** ~3.5 días de curso por día real → ~73 días de curso = ~80% del programa

---

## Cómo retomar en sesión nueva

1. Lee este archivo (`PLAN_21DIAS.md`)
2. Lee `memory/project_course.md` para contexto completo
3. Pregunta al usuario en qué día real está y qué completó ayer
4. Retoma desde donde indica el tracking de abajo

---

## Estado actual de cada curso

| Curso | Último completado | Siguiente |
|-------|-------------------|-----------|
| **C1 — MLOps** | D2 Dockerfile + D5b scikit-learn/PyTorch | D3 Docker Compose + FastAPI |
| **C2 — AI Engineer** | D7 ReAct manual (dia7_agente_react.py) | D9 LangChain Agents — AgentExecutor (dia8_langchain_agents.py, tools definidas) |
| **C3 — Cloud Engineer** | D1 Azure fundamentals (dia1_azure_fundamentos.md) | D2 Networking I: VNets, subnets, NSGs, peering |

---

## Plan día a día

| Día real | Fecha | Bloque 1 (mañana) | Bloque 2 (tarde) | Estado |
|----------|-------|-------------------|------------------|--------|
| 1 | 4-5 mayo | C2-D9: LangChain Agents (AgentExecutor) | C3-S1D2: Networking I — VNets, NSGs, peering | ⏳ |
| 2 | 5-6 mayo | C1-D3: Docker Compose + FastAPI serving | C3-S1D3: Networking II — VPN, Firewall, Private Endpoints | ⬜ |
| 3 | 6-7 mayo | C3-S1D4: Azure AD, RBAC, Key Vault | C1-D4: Azure CLI + ACR + AKS overview | ⬜ |
| 4 | 7-8 mayo | C3-S1D5: Storage — Blob, Files, lifecycle | C1-D6: Proyecto Docker + FastAPI + MLflow + ACR | ⬜ |
| 5 | 8-9 mayo | C3-S1D6: Proyecto Hub-Spoke | C1-D7: Test S1 + DVC básico | ⬜ |
| 6 | 9-10 mayo | C3-S2D8: Governance, Management Groups, Azure Policy | C2-D10: Semantic Kernel — kernels, plugins | ⬜ |
| 7 | 10-11 mayo | C3-S2D9: Landing Zones + Cloud Adoption Framework | C2-D11: Semantic Kernel avanzado — planners | ⬜ |
| 8 | 11-12 mayo | C3-S2D10: FinOps + Cost Management + tagging | C1-S2D8: K8s — Control Plane, Nodes, etcd | ⬜ |
| 9 | 12-13 mayo | C3-S2D11: Bicep — sintaxis, módulos | C1-S2D9: K8s — Pods, Deployments, Services | ⬜ |
| 10 | 13-14 mayo | C3-S2D12: Terraform avanzado — remote state, workspaces | C1-S2D10: K8s — ConfigMaps, Secrets | ⬜ |
| 11 | 14-15 mayo | C3-S2D13: Proyecto Landing Zone con Terraform | C1-S2D11-12: Ingress + Helm | ⬜ |
| 12 | 15-16 mayo | C3-S3D15: HA/DR — Availability Zones, Backup, Site Recovery | C1-S2D13: Proyecto K8s local con Helm | ⬜ |
| 13 | 16-17 mayo | C3-S3D16: Azure SQL + Cosmos DB + PostgreSQL | C2-D12: Multi-agent + LangGraph intro | ⬜ |
| 14 | 17-18 mayo | C3-S3D17: Defender for Cloud + C3-S3D17b: Sentinel + Purview | C2-D12b: LangGraph avanzado | ⬜ |
| 15 | 18-19 mayo | C3-S3D18: Zero Trust — Conditional Access, PIM, JIT | C2-D13: Proyecto agente financiero con tools externas | ⬜ |
| 16 | 19-20 mayo | C3-S3D19: Azure Monitor + Log Analytics + App Insights | C1-S3D15: AKS cluster + ACR integración | ⬜ |
| 17 | 20-21 mayo | C3-S3D20: Proyecto bancario HA+DR+Seguridad | C2-D15: Azure AI Foundry — hubs, proyectos, modelos | ⬜ |
| 18 | 21-22 mayo | C3-S4D22b: Azure API Management (Open Banking) | C3-S4D24: Well-Architected + TCO estimation | ⬜ |
| 19 | 22-23 mayo | C3-S4D24b: Service Bus + Event Hub | C2-D19: Evaluación LLMs + RAGAS + benchmarks | ⬜ |
| 20 | 23-24 mayo | C3-S4D25b: Microsoft Fabric + Power BI | C2-D19b: Azure Prompt Flow | ⬜ |
| 21 | 24 mayo | C3-S4D26: SOW + propuesta cliente + C3-S4D26b: TOGAF | C2-D23b: Responsible AI + EU AI Act (FS) | ⬜ |

---

## Lo que se queda para después del 25 de mayo (on-the-job)

- C1 S3 completa: Kubeflow, Airflow, Great Expectations, CI/CD AKS (contenido ya escrito en semana3/README.md)
- C1 S4 completa: Evidently, Prometheus+Grafana, proyecto final MLOps
- C2 S4: arquitecturas agentic, escalabilidad, proyecto final
- C3 S4: GCP fundamentos, GitOps ArgoCD, proyecto final bancario

---

## Nuevos temas añadidos al programa original

| Tema | Curso | Día |
|------|-------|-----|
| LangGraph avanzado (graph state, conditional edges) | C2 | D12b |
| Azure Prompt Flow | C2 | D19b |
| Responsible AI + EU AI Act para FS | C2 | D23b |
| Microsoft Sentinel + Microsoft Purview | C3 | D17b |
| Azure API Management (Open Banking) | C3 | D22b |
| TCO estimation para cliente | C3 | D24 (ampliado) |
| Azure Service Bus + Event Hub | C3 | D24b |
| Microsoft Fabric + Power BI | C3 | D25b |
| SOW / propuesta cliente | C3 | D26 (ampliado) |
| TOGAF básico cloud | C3 | D26b |

---

## Flujo de fin de día

```bash
git add .
git commit -m "Plan DíaX — temas completados"
git push
```

Actualizar el estado en este archivo: ⏳ → ✅ cuando el día esté completo.

---

## Azure Free Account

- Crear cuando lleguemos a C3-S1D2 (Día real 1, bloque tarde) o antes
- URL: `azure.microsoft.com/free` — 200€ crédito 30 días + tarjeta para verificar
- Microsoft Learn Sandboxes como complemento gratuito sin tarjeta
