# Día 1 — Azure Fundamentos: Jerarquía, Recursos y CLI
# Curso 3: Cloud Engineer (KPMG Ready)
# Orientado a banca y consultoría enterprise

---

## Por qué empezamos aquí

Antes de desplegar una VM, una VNet o un AKS en Azure necesitas entender
**cómo está organizado Azure por dentro**. Sin esto no puedes:
- Diseñar governance para un banco
- Separar entornos (prod/pre/dev) de forma segura
- Controlar quién puede hacer qué y dónde
- Gestionar costes por área de negocio

Es la base de todo lo demás.

---

## 1. La jerarquía de Azure — de arriba a abajo

```
Tenant (Azure AD)
│
├── Management Group — Raíz
│       │
│       ├── Management Group — Banca Retail
│       │       ├── Subscription — Producción
│       │       ├── Subscription — Pre-producción
│       │       └── Subscription — Desarrollo
│       │
│       └── Management Group — Banca Corporativa
│               ├── Subscription — Producción
│               └── Subscription — Desarrollo
│
└── (cada Subscription contiene...)
        │
        ├── Resource Group — rg-networking-prod
        │       ├── Virtual Network
        │       ├── NSG
        │       └── VPN Gateway
        │
        ├── Resource Group — rg-aplicaciones-prod
        │       ├── AKS Cluster
        │       ├── Azure SQL
        │       └── App Service
        │
        └── Resource Group — rg-seguridad-prod
                ├── Key Vault
                └── Log Analytics Workspace
```

### Qué es cada nivel

**Tenant**
- Tu organización en Azure AD (el directorio de identidades)
- Un banco tiene UN tenant — todos los empleados y aplicaciones viven aquí
- Es el límite de confianza: lo que está dentro del tenant comparte identidad

**Management Group**
- Agrupación lógica de suscripciones para aplicar políticas a escala
- Ejemplo banco: MG "Producción" → política que obliga cifrado en todos los recursos
- Se aplica una vez en el MG y se hereda en todas sus suscripciones automáticamente
- Hasta 6 niveles de jerarquía

**Subscription (Suscripción)**
- Unidad de facturación y límite administrativo
- Cada suscripción tiene sus propios límites (quotas): max VMs, max IPs, etc.
- En banca: suscripción separada para prod y dev → si alguien borra algo en dev, prod no se ve afectado
- También permite asignar presupuesto por suscripción (clave para FinOps)

**Resource Group**
- Contenedor lógico de recursos que comparten ciclo de vida
- Regla de oro: si eliminas el Resource Group, eliminas TODO lo que contiene
- Un recurso solo puede estar en UN Resource Group
- Están en una región, pero pueden contener recursos de otras regiones
- En banca: agrupar por capa (networking, apps, datos, seguridad) o por aplicación

**Resource**
- El recurso en sí: una VM, una VNet, un Key Vault, un AKS...
- Todo recurso tiene: nombre, tipo, región, Resource Group, suscripción, tags

---

## 2. Azure Resource Manager (ARM) — el motor de todo

```
Tú (Portal / CLI / Terraform / SDK)
            │
            ▼
    Azure Resource Manager (ARM)
    ─────────────────────────────
    · Autenticación (Azure AD)
    · Autorización (RBAC)
    · Throttling (límites de API)
    · Auditoría (Activity Log)
            │
            ▼
    Resource Providers
    ─────────────────
    Microsoft.Compute   → VMs, discos
    Microsoft.Network   → VNets, NSGs, Load Balancers
    Microsoft.Storage   → Storage Accounts, Blobs
    Microsoft.KeyVault  → Key Vaults
    Microsoft.ContainerService → AKS
    Microsoft.Sql       → Azure SQL
```

**ARM es la API unificada de Azure.** Da igual si usas el Portal, la CLI,
Terraform o un script Python — todos pasan por ARM. Esto significa:
- Todo queda registrado en el Activity Log (auditoría)
- Los permisos RBAC se aplican en todos los canales por igual
- Terraform no hace "magia" — traduce tu HCL a llamadas ARM

---

## 3. Regiones y Availability Zones

```
Región: West Europe (Ámsterdam)
├── Availability Zone 1  ── Datacenter físico A
├── Availability Zone 2  ── Datacenter físico B
└── Availability Zone 3  ── Datacenter físico C

Región: North Europe (Irlanda)   ← region pair de West Europe
```

**Región**
- Ubicación geográfica con uno o más datacenters
- Para banca española: datos tienen que estar en la UE (RGPD)
- Regiones recomendadas: West Europe (Ámsterdam) o Spain Central (Madrid) — disponible desde 2023

**Region Pair**
- Cada región tiene una pareja para Disaster Recovery
- West Europe ↔ North Europe
- Azure garantiza que nunca actualiza ambas regiones del par a la vez
- Azure Site Recovery replica entre pares de regiones automáticamente

**Availability Zone (AZ)**
- Datacenter físicamente separado dentro de la misma región
- Tienen alimentación, red y refrigeración independiente
- Si cae una AZ, las otras siguen funcionando
- SLA 99.99% con recursos distribuidos en ≥2 AZs vs 99.9% con una sola AZ
- Para un banco: bases de datos y AKS SIEMPRE en múltiples AZs

---

## 4. Naming Convention — crítico en enterprise

Sin naming convention, en 6 meses nadie sabe qué es qué.
Estándar recomendado por Microsoft (CAF):

```
{tipo}-{aplicación}-{entorno}-{región}-{instancia}

Ejemplos:
rg-pagos-prod-we-001        → Resource Group, app pagos, producción, West Europe
vnet-hub-prod-we-001        → VNet hub, producción, West Europe
aks-fraude-prod-we-001      → AKS cluster, app fraude, producción
kv-secretos-prod-we-001     → Key Vault, producción
sql-clientes-prod-we-001    → Azure SQL, app clientes, producción
nsg-app-prod-we-001         → Network Security Group, capa app, producción

Abreviaturas de entorno:  prod / pre / dev / test
Abreviaturas de región:   we (West Europe) / ne (North Europe) / sc (Spain Central)
```

**Por qué importa en banca:**
- Auditorías regulatorias — el auditor necesita entender la infraestructura de un vistazo
- Automatización — los scripts de governance buscan recursos por nombre/tag
- FinOps — el tagging + naming permite asignar costes a centros de coste

---

## 5. Tagging — la base de FinOps y Governance

Los tags son pares clave-valor que se aplican a cualquier recurso.

**Taxonomía estándar para banca:**

| Tag | Valores ejemplo | Para qué sirve |
|-----|----------------|----------------|
| `environment` | prod / pre / dev | Separar costes y aplicar políticas |
| `application` | pagos / fraude / clientes | Asignar costes por aplicación |
| `owner` | equipo-pagos / equipo-datos | Saber a quién llamar si hay un problema |
| `cost-center` | CC-1234 / CC-5678 | Imputar costes al área de negocio |
| `business-unit` | retail / corporativo / riesgos | Reporting financiero |
| `criticality` | critical / high / medium / low | Priorizar respuesta ante incidencias |
| `data-classification` | confidential / internal / public | Compliance y seguridad de datos |

**Regla:** sin tag `environment` y `cost-center` → el recurso no se crea.
Se enforce con Azure Policy (lo veremos en Semana 2).

---

## 6. Azure CLI — el día a día de un Cloud Engineer

El Portal está bien para explorar. En producción y automatización se usa CLI.

### Instalación y login

```bash
# Verificar instalación
az --version

# Login (abre el navegador)
az login

# Login con service principal (para automatización / CI/CD)
az login --service-principal \
  --username $APP_ID \
  --password $PASSWORD \
  --tenant $TENANT_ID

# Ver tu cuenta activa
az account show

# Listar todas las suscripciones a las que tienes acceso
az account list --output table

# Cambiar a una suscripción específica
az account set --subscription "Subscription-Produccion"
```

### Comandos fundamentales

```bash
# ─── RESOURCE GROUPS ───────────────────────────────────────

# Crear Resource Group
az group create \
  --name rg-laboratorio-dev-we-001 \
  --location westeurope \
  --tags environment=dev owner=alejandro cost-center=CC-LAB

# Listar Resource Groups
az group list --output table

# Ver detalles de un RG
az group show --name rg-laboratorio-dev-we-001

# Eliminar RG y TODOS sus recursos (cuidado)
az group delete --name rg-laboratorio-dev-we-001 --yes


# ─── RECURSOS GENÉRICOS ────────────────────────────────────

# Listar todos los recursos de un RG
az resource list \
  --resource-group rg-laboratorio-dev-we-001 \
  --output table

# Buscar recursos por tag
az resource list \
  --tag environment=prod \
  --output table

# Ver el Activity Log (auditoría) de un RG
az monitor activity-log list \
  --resource-group rg-laboratorio-dev-we-001 \
  --output table


# ─── TAGS ──────────────────────────────────────────────────

# Añadir tags a un recurso existente
az resource tag \
  --resource-group rg-laboratorio-dev-we-001 \
  --name mi-recurso \
  --resource-type "Microsoft.Storage/storageAccounts" \
  --tags environment=dev owner=alejandro

# Ver tags de un recurso
az resource show \
  --resource-group rg-laboratorio-dev-we-001 \
  --name mi-recurso \
  --resource-type "Microsoft.Storage/storageAccounts" \
  --query tags


# ─── GESTIÓN DE SUSCRIPCIONES Y MANAGEMENT GROUPS ──────────

# Listar Management Groups
az account management-group list --output table

# Ver jerarquía completa
az account management-group show \
  --name "mg-banco-produccion" \
  --expand --recurse
```

### Outputs y queries útiles

```bash
# --output controla el formato: table / json / yaml / tsv / none
az group list --output table    # humano
az group list --output json     # para scripts
az group list --output tsv      # para pipes en bash

# --query filtra con JMESPath (muy útil en scripts)

# Solo los nombres de los RGs
az group list --query "[].name" --output tsv

# RGs en westeurope
az group list \
  --query "[?location=='westeurope'].{Nombre:name, Estado:properties.provisioningState}" \
  --output table

# Recursos de tipo VM en un RG
az resource list \
  --resource-group rg-aplicaciones-prod-we-001 \
  --query "[?type=='Microsoft.Compute/virtualMachines'].{Nombre:name}" \
  --output table
```

---

## 7. Caso práctico — Estructura para un banco mediano

Escenario: un banco mediano te contrata para diseñar su estructura Azure desde cero.
Requisitos: entornos separados, governance centralizado, cumplimiento RGPD, costes por área.

### Estructura propuesta

```
Tenant: banco.onmicrosoft.com
│
Management Group: MG-Banco-Raiz
├── MG-Plataforma              → políticas globales de seguridad
│   └── Sub-SharedServices     → DNS, conectividad, monitorización central
│
├── MG-Produccion              → política: cifrado obligatorio, no IPs públicas
│   ├── Sub-Prod-Retail        → aplicaciones banca retail
│   ├── Sub-Prod-Corporativo   → aplicaciones banca corporativa
│   └── Sub-Prod-Riesgos       → modelos de riesgo y fraude
│
└── MG-NoProduccion            → política: auto-shutdown a las 20h (ahorro coste)
    ├── Sub-PreProd            → pre-producción
    └── Sub-Dev                → desarrollo y pruebas
```

**Por qué esta estructura:**
- Producción y no-producción en Management Groups separados → políticas distintas
- Shared Services centralizado → el DNS y la conectividad son únicos, no duplicados
- Una suscripción por dominio de negocio en prod → blast radius limitado
- Auto-shutdown en dev → ahorro del 60% en costes de entornos de desarrollo

### Ejercicio — créalo con CLI

```bash
# 1. Crear los Resource Groups del entorno de laboratorio
# (simulamos la Sub-Dev con RGs en tu suscripción personal)

az group create \
  --name rg-networking-dev-we-001 \
  --location westeurope \
  --tags environment=dev owner=alejandro cost-center=CC-LAB criticality=low

az group create \
  --name rg-aplicaciones-dev-we-001 \
  --location westeurope \
  --tags environment=dev owner=alejandro cost-center=CC-LAB criticality=low

az group create \
  --name rg-datos-dev-we-001 \
  --location westeurope \
  --tags environment=dev owner=alejandro cost-center=CC-LAB criticality=low

az group create \
  --name rg-seguridad-dev-we-001 \
  --location westeurope \
  --tags environment=dev owner=alejandro cost-center=CC-LAB criticality=low

# 2. Verificar que se crearon todos con sus tags
az group list \
  --query "[?starts_with(name, 'rg-') && tags.environment=='dev'].{Nombre:name, Region:location}" \
  --output table

# 3. Simular búsqueda FinOps: recursos sin tag cost-center
# (en un banco real esto detecta recursos huérfanos como los de Uterque)
az resource list \
  --query "[?tags.\"cost-center\" == null].{Nombre:name, Tipo:type, RG:resourceGroup}" \
  --output table

# 4. Limpiar al terminar (elimina los 4 RGs)
for rg in rg-networking-dev-we-001 rg-aplicaciones-dev-we-001 rg-datos-dev-we-001 rg-seguridad-dev-we-001; do
  az group delete --name $rg --yes --no-wait
  echo "Eliminando $rg..."
done
```

---

## 8. Comparativa: Portal vs CLI vs Terraform

| | Portal Azure | Azure CLI | Terraform |
|---|---|---|---|
| **Cuándo usarlo** | Exploración, diagnóstico puntual | Automatización, scripts, CI/CD | Infraestructura permanente como código |
| **Reproducible** | No | Sí (script) | Sí (declarativo) |
| **Control de versiones** | No | Sí (el script) | Sí (el .tf) |
| **Drift detection** | No | No | Sí (`terraform plan`) |
| **Curva de aprendizaje** | Baja | Media | Media-alta |
| **En consultoría KPMG** | Para demos y troubleshooting | Para automatización y onboarding | Para todo lo que va a producción |

**Regla de oro:** si vas a crear algo más de una vez → CLI o Terraform. Si es un experimento puntual → Portal.

---

## 9. Servicios Azure por categoría — mapa mental para banca

```
IDENTIDAD Y ACCESO
  Azure Active Directory    → autenticación, usuarios, grupos
  RBAC                      → quién puede hacer qué en qué recurso
  PIM                       → acceso admin just-in-time
  Managed Identity          → apps que se autentican sin credenciales

RED
  Virtual Network (VNet)    → red privada en Azure
  Subnet                    → segmento de la VNet
  NSG                       → firewall a nivel de subred/NIC
  Azure Firewall            → firewall centralizado para todo el tráfico
  VPN Gateway               → conexión cifrada con datacenter on-prem
  ExpressRoute              → conexión privada dedicada (lo que usa banca)
  Private Endpoint          → servicio Azure sin IP pública
  Azure Load Balancer       → balanceo L4 entre VMs
  Application Gateway + WAF → balanceo L7 + protección web
  Traffic Manager           → balanceo DNS entre regiones (DR)

COMPUTE
  Virtual Machines          → IaaS clásico
  AKS                       → Kubernetes gestionado (CaaS)
  App Service               → PaaS para aplicaciones web/APIs
  Azure Functions           → serverless, eventos
  Container Instances (ACI) → contenedor puntual sin cluster

DATOS
  Azure SQL Database        → SQL relacional gestionado
  Cosmos DB                 → NoSQL distribuido globalmente
  Azure Storage (Blob)      → objetos (documentos, imágenes, backups)
  Azure Data Lake           → datos masivos para analítica
  Azure Synapse             → analytics a escala (antes SQL DW)
  Microsoft Fabric          → plataforma unificada datos + analytics

IA
  Azure OpenAI Service      → GPT-4o, embeddings
  Azure AI Foundry          → plataforma agentes IA enterprise
  Azure AI Search           → vectorstore + búsqueda semántica
  Azure Machine Learning    → MLOps, entrenamiento, endpoints

SEGURIDAD
  Key Vault                 → secretos, claves, certificados
  Microsoft Sentinel        → SIEM con IA (detección amenazas)
  Defender for Cloud        → postura de seguridad, alertas
  Microsoft Purview         → governance datos, DLP, clasificación

OBSERVABILIDAD
  Azure Monitor             → métricas y logs de toda la infraestructura
  Log Analytics Workspace   → motor de consultas sobre logs
  Application Insights      → monitorización de aplicaciones
  Azure Alerts              → notificaciones ante umbrales

GOVERNANCE Y FINOPS
  Azure Policy              → reglas que se aplican automáticamente
  Management Groups         → jerarquía para aplicar políticas a escala
  Cost Management           → análisis y alertas de gasto
  Azure Advisor             → recomendaciones de optimización (rightsizing)
  Microsoft Purview         → clasificación y governance de datos

RESILIENCIA
  Azure Backup              → backup gestionado de VMs, BBDDs
  Azure Site Recovery       → replicación y failover entre regiones
  Availability Zones        → redundancia dentro de una región
```

---

## Mini-test Día 1

1. Un banco tiene 3 entornos (prod, pre, dev) y 2 áreas de negocio (retail, corporativo). ¿Cuántas suscripciones mínimas recomendarías y por qué?

2. ¿Qué diferencia hay entre un Management Group y una Suscripción?

3. ¿Por qué los datos de clientes de un banco español deben estar en West Europe o Spain Central y no en East US?

4. Si eliminas un Resource Group, ¿qué pasa con todos los recursos que contiene?

5. Un auditor del Banco de España te pide "quiero ver todos los cambios que se hicieron en la infraestructura de producción el mes pasado". ¿Qué servicio Azure usarías?

6. Inditex tiene recursos cloud sin tag `cost-center`. ¿Cómo lo detectarías con CLI y cómo lo prevendrías en el futuro?

---

## Respuestas mini-test

1. **Mínimo 4 suscripciones:** prod-retail, prod-corporativo, pre (compartida), dev (compartida). Prod separada por área de negocio para que un incidente en retail no afecte a corporativo. Pre y dev pueden compartir suscripción porque el blast radius es menor.

2. **Management Group** es una agrupación lógica de suscripciones para aplicar políticas heredadas. **Suscripción** es la unidad de facturación y límite administrativo real donde se crean los recursos. Los MGs no contienen recursos directamente — contienen suscripciones.

3. **RGPD** — el Reglamento General de Protección de Datos de la UE obliga a que los datos personales de ciudadanos europeos se almacenen y procesen dentro de la UE. East US está fuera de la UE. Además, la regulación bancaria española (Banco de España, ENS) refuerza esta exigencia.

4. **Se eliminan todos** — el Resource Group es el contenedor y su eliminación es irreversible para todos los recursos que contiene. Por eso en producción se aplican Resource Locks (candados) que impiden el borrado accidental.

5. **Azure Activity Log** — registra todas las operaciones realizadas sobre los recursos (quién, qué, cuándo, desde dónde). Está integrado en Azure Monitor y se puede consultar con CLI: `az monitor activity-log list`.

6. **Detectar:** `az resource list --query "[?tags.\"cost-center\" == null]"`. **Prevenir:** Azure Policy con efecto `deny` — cualquier recurso que se intente crear sin el tag `cost-center` es rechazado automáticamente antes de existir.

---

## Próximo día

**Día 2 — Networking I: VNets, Subnets, NSGs y Peering**
La red es el fundamento de la seguridad en Azure. Un banco no puede tener sus sistemas core
accesibles desde internet — toda la arquitectura de red se diseña para aislar y controlar el tráfico.
