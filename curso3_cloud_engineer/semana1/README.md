# Semana 1 — Azure Core + Networking

> "Antes de desplegar nada, hay que dominar la red"

## Días

| Día | Tema | Estado |
|-----|------|--------|
| 1 | Azure fundamentals: suscripciones, recursos, portal, CLI | ✅ |
| 2 | Networking I: VNets, subnets, NSGs, peering | ✅ |
| 3 | Networking II: VPN Gateway, ExpressRoute, Azure Firewall, Private Endpoints | ✅ |
| 4 | Identidad: Azure AD, RBAC, Managed Identities, Key Vault | ⬜ |
| 5 | Storage: Blob, Files, redundancia, lifecycle policies | ⬜ |
| 6 | Proyecto: arquitectura hub-spoke con acceso seguro | ⬜ |
| 7 | Test semana 1 + repaso AZ-104 bloque redes | ⬜ |

**Checkpoint:** Desplegar una arquitectura Hub-Spoke en Azure con VNets, NSGs, acceso seguro mediante Private Endpoints y gestión de identidades con RBAC + Key Vault.

---

## Día 1 — Azure fundamentals

> Contenido en [dia1_azure_fundamentos.md](dia1_azure_fundamentos.md)

---

## Día 2 — Networking I: VNets, subnets, NSGs, peering

> Contenido en [dia2_networking_I.md](dia2_networking_I.md)

---

## Día 3 — Networking II: VPN Gateway, ExpressRoute, Azure Firewall, Private Endpoints

### Arquitectura típica banca

```
DATACENTER BANCO
(core banking, mainframe)
         │
         │ ExpressRoute (fibra dedicada)
         │
         ▼
┌─────────────────────────────────────────────┐
│              HUB VNet                       │
│  ┌──────────────┐    ┌──────────────────┐   │
│  │ VPN Gateway  │    │  Azure Firewall  │   │
│  │(sucursales)  │    │  (todo el tráf.) │   │
│  └──────────────┘    └──────────────────┘   │
└──────────────┬──────────────────────────────┘
               │ VNet Peering
    ┌──────────┴──────────┐
    ▼                     ▼
┌────────────┐      ┌────────────┐
│ SPOKE App  │      │ SPOKE Data │
│ FastAPI    │      │ Azure SQL  │◀─── Private Endpoint
│ AKS        │      │ Storage    │     (IP privada, sin internet)
└────────────┘      └────────────┘
```

### Conceptos clave

| Concepto | Qué es | Cuándo usarlo |
|----------|--------|---------------|
| **VPN Gateway** | Túnel cifrado sobre internet (IPSec/IKE) | Sucursales remotas, dev/test, bajo coste |
| **ExpressRoute** | Fibra privada dedicada a Azure (sin internet) | Core banking, trading, datos regulados |
| **Azure Firewall** | Firewall gestionado Layer 4+7 con reglas FQDN | Perímetro corporativo, visibilidad centralizada |
| **NSG** | Filtrado IP+puerto a nivel subnet/NIC | Control básico entre subnets, gratis |
| **Private Endpoint** | NIC con IP privada que representa un servicio Azure dentro de tu VNet | Acceso a Storage/SQL sin pasar por internet |

### Trade-offs VPN Gateway vs ExpressRoute

| | VPN Gateway | ExpressRoute |
|---|---|---|
| Va por internet | Sí (cifrado) | No (fibra privada) |
| Velocidad | Hasta 10 Gbps | Hasta 100 Gbps |
| Latencia | Variable | Predecible y baja |
| Coste | ~100-300€/mes | ~1.000-5.000€/mes |
| SLA | 99,9% | 99,95% |

### NSG vs Azure Firewall

| | NSG | Azure Firewall |
|---|---|---|
| Nivel | Subnet / NIC | VNet completa / hub |
| Reglas | IP + puerto | IP, puerto, FQDN, threat intelligence |
| Coste | Gratis | ~1.000€/mes |
| Logs centralizados | Limitado | Sí, integrado con Log Analytics |

### Aplicación en KPMG

- **ExpressRoute** es el estándar en banca para conectar datacenters a Azure — latencia predecible y cumplimiento regulatorio
- **Private Endpoints** son obligatorios en proyectos FS: los datos financieros nunca deben tocar internet público (DORA, GDPR)
- **Azure Firewall** en el hub da visibilidad centralizada de todo el tráfico — requisito habitual de los CISO en auditorías
- Coste orientativo de red en un proyecto bancario completo: **5.000-8.000€/mes** solo en infraestructura de red

---
