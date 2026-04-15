# Curso 3 — Cloud Engineer (KPMG Ready)

**Perfil:** DevOps/Cloud Engineer con base en AWS, Terraform, Jenkins, Ansible
**Objetivo:** Dominar Azure Core + Networking + Gobernanza + IaC profesional + Multi-cloud
**Target:** Rol Cloud Engineer en consultoría (KPMG, Accenture, Deloitte Tech)
**Certificaciones objetivo:** AZ-104 (Azure Administrator) + AZ-305 (Azure Solutions Architect)
**Ritmo:** 5-6h/día · 60% práctica · 25% arquitectura · 15% teoría
**Estado:** ⬜ Pendiente

---

## Por qué este curso existe

Los cursos 1 (MLOps) y 2 (AI Engineer) cubren la capa de datos e inteligencia.
Este curso cubre la **infraestructura empresarial** que hay debajo — lo que KPMG necesita:

- Diseñar y desplegar arquitecturas en Azure para clientes grandes (banca, seguros, sector público)
- Implementar gobernanza, seguridad y compliance a escala enterprise
- Justificar decisiones de arquitectura con trade-offs documentados
- Hablar multi-cloud con criterio

---

## Coste estimado

**Casi todo gratis.** Estrategia:

| Recurso | Coste | Uso |
|---------|-------|-----|
| Microsoft Learn Sandboxes | Gratis | Labs principales (Azure real, temporal) |
| Azure Free Account (200€ crédito 30 días) | Gratis | Labs propios con más libertad |
| GCP Free Tier (300$ crédito 90 días) | Gratis | Semana 4 |
| GitHub | Gratis | GitOps, CI/CD, Terraform state |
| VPN Gateway, ExpressRoute | Caro → evitamos | Solo teoría + sandbox |

**Estimación total: 0-20€**

---

## Progreso

| Semana | Día | Tema | Estado |
|--------|-----|------|--------|
| 1 | 1 | Azure fundamentals: suscripciones, recursos, portal, CLI | ⬜ |
| 1 | 2 | Networking I: VNets, subnets, NSGs, peering | ⬜ |
| 1 | 3 | Networking II: VPN Gateway, ExpressRoute, Azure Firewall, Private Endpoints | ⬜ |
| 1 | 4 | Identidad: Azure AD, RBAC, Managed Identities, Key Vault | ⬜ |
| 1 | 5 | Storage: Blob, Files, redundancia (LRS/GRS/ZRS), lifecycle policies | ⬜ |
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
| 3 | 16 | Azure SQL + Cosmos DB + PostgreSQL Flexible: cuándo usar cada uno | ⬜ |
| 3 | 17 | Microsoft Defender for Cloud: postura de seguridad, alertas, compliance | ⬜ |
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

---

## Proyecto Final — Arquitectura Empresarial para Banca

```
Cliente ficticio: banco mediano que migra a Azure

Management Group
        ↓
Subscriptions (prod / dev / shared-services)
        ↓
Hub VNet (Azure Firewall + VPN Gateway)
   ↕ peering
Spoke VNets (app / datos / seguridad)
        ↓
AKS (apps containerizadas) + Azure SQL + Cosmos DB
        ↓
Key Vault + Managed Identities + RBAC
        ↓
Microsoft Defender for Cloud + Azure Policy
        ↓
Azure Monitor + Log Analytics + Alertas
        ↓
Terraform (toda la infra como código) + GitOps con ArgoCD
```

Documentación entregable: diagrama de arquitectura + decisiones justificadas + coste estimado.

---

## Prioridad si la entrevista KPMG es pronto

Orden de impacto para preparar la entrevista:

1. **Azure networking** (VNets, NSGs, Private Endpoints) — pregunta casi segura
2. **RBAC + Managed Identities + Key Vault** — seguridad crítica en consultoría
3. **Landing Zones + Azure Policy** — diferencia junior de senior en KPMG
4. **Terraform avanzado + Bicep** — IaC es el día a día
5. **Well-Architected Framework** — para justificar decisiones de arquitectura

---

## Semanas

- [Semana 1 — Azure Core + Networking](./semana1/README.md)
- [Semana 2 — Gobernanza + IaC Profesional](./semana2/README.md)
- [Semana 3 — Alta Disponibilidad + Seguridad](./semana3/README.md)
- [Semana 4 — Multi-cloud + Well-Architected + Proyecto Final](./semana4/README.md)
