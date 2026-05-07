# Día 2 — Networking I: VNets, Subnets, NSGs, Peering
# Curso 3: Cloud Engineer (KPMG Ready)

> "Todo en Azure vive dentro de una red. Si no dominas la red, no dominas Azure."

---

## ¿Por qué importa en consultoría bancaria?

En un banco, la red no es un detalle técnico — es un requisito regulatorio. ENS, DORA y RGPD exigen:
- Segmentación de entornos (producción aislado de desarrollo)
- Control de tráfico entre sistemas (qué puede hablar con qué)
- Sin exposición directa a internet de sistemas críticos

Un arquitecto cloud en KPMG diseña estas redes antes de desplegar una sola VM.

---

## 1. VNet — Virtual Network

Una **VNet** es una red privada en Azure. Todo recurso que necesite comunicarse (VMs, AKS, bases de datos, App Services...) vive dentro de una VNet.

```
Internet
    │
    │  (solo lo que tú permitas entra)
    │
┌───▼────────────────────────────────────┐
│  VNet: 10.0.0.0/16  (65.536 IPs)      │
│                                        │
│  ┌──────────────┐  ┌────────────────┐  │
│  │ Subnet Web   │  │ Subnet App     │  │
│  │ 10.0.1.0/24  │  │ 10.0.2.0/24   │  │
│  │ (254 hosts)  │  │ (254 hosts)    │  │
│  └──────────────┘  └────────────────┘  │
│                                        │
│  ┌──────────────┐  ┌────────────────┐  │
│  │ Subnet DB    │  │ Subnet Mgmt    │  │
│  │ 10.0.3.0/24  │  │ 10.0.4.0/24   │  │
│  └──────────────┘  └────────────────┘  │
└────────────────────────────────────────┘
```

### Conceptos clave de VNet

| Concepto | Qué es | Ejemplo bancario |
|----------|--------|-----------------|
| **Address space** | Rango IP total de la VNet | `10.0.0.0/16` — 65.536 IPs para el banco |
| **Region** | La VNet existe en una región Azure | `westeurope` para un banco español |
| **DNS** | Resolución de nombres interna | VMs se resuelven por hostname, no por IP |
| **Subscription** | Una VNet pertenece a una suscripción | VNet de producción en suscripción prod |

### Reglas de diseño de VNets

1. **No solapar rangos IP** entre VNets que vayan a conectarse (peering o VPN)
2. **Reservar suficiente espacio**: mejor `/16` que `/24` aunque uses poco — escalar un address space después es costoso
3. **Una VNet por entorno** como mínimo: prod, dev, shared-services separadas

---

## 2. Subnets

Una **subnet** es una subdivisión de la VNet. Los recursos se despliegan en subnets, no directamente en la VNet.

### ¿Por qué dividir en subnets?

- Aplicar seguridad diferente a cada capa (NSGs por subnet)
- Controlar el enrutamiento (UDRs por subnet)
- Algunos servicios Azure requieren su propia subnet dedicada (Gateway, Firewall, Bastion)

### Subnets especiales en Azure

| Subnet | Para qué | Nombre obligatorio |
|--------|----------|-------------------|
| `GatewaySubnet` | VPN Gateway / ExpressRoute | Sí — nombre exacto |
| `AzureFirewallSubnet` | Azure Firewall | Sí — nombre exacto |
| `AzureBastionSubnet` | Azure Bastion | Sí — nombre exacto |

### Arquitectura típica bancaria (hub-spoke)

```
Hub VNet (shared-services)
├── GatewaySubnet        → VPN hacia on-premise
├── AzureFirewallSubnet  → inspección de tráfico
└── AzureBastionSubnet   → acceso seguro a VMs sin IP pública

Spoke VNet (producción banco)
├── snet-web             → App Gateway + apps frontend
├── snet-app             → microservicios, AKS nodes
├── snet-data            → Azure SQL, Cosmos DB
└── snet-mgmt            → VMs de gestión internas
```

### Tamaño de subnets — regla Azure

Azure **reserva 5 IPs** de cada subnet (red, broadcast + 3 de Azure). Una `/24` tiene 256 IPs → 251 utilizables.

| CIDR | IPs totales | IPs utilizables |
|------|-------------|----------------|
| /24 | 256 | 251 |
| /25 | 128 | 123 |
| /26 | 64 | 59 |
| /27 | 32 | 27 |
| /28 | 16 | 11 |

Para AKS: usar `/22` o mayor — cada pod consume una IP.

---

## 3. NSG — Network Security Group

Un **NSG** es un firewall básico que se aplica a una subnet o a una NIC (tarjeta de red). Contiene reglas que permiten o deniegan tráfico.

### Estructura de una regla NSG

```
Prioridad | Nombre        | Puerto | Protocolo | Origen        | Destino      | Acción
-----------------------------------------------------------------------------------------------
100       | Allow-HTTPS   | 443    | TCP       | Internet      | 10.0.1.0/24  | Allow
200       | Allow-HTTP    | 80     | TCP       | Internet      | 10.0.1.0/24  | Allow
300       | Allow-App-DB  | 5432   | TCP       | 10.0.2.0/24   | 10.0.3.0/24  | Allow
4096      | Deny-All      | *      | *         | *             | *            | Deny  ← siempre al final
```

**Prioridad:** de 100 a 4096. **Menor número = mayor prioridad.** Se evalúan en orden hasta que una regla coincide.

### Reglas por defecto (no se pueden eliminar)

Azure añade automáticamente estas reglas en todo NSG:

**Inbound por defecto:**
| Prioridad | Nombre | Qué permite |
|-----------|--------|-------------|
| 65000 | AllowVnetInBound | Tráfico dentro de la misma VNet |
| 65001 | AllowAzureLoadBalancerInBound | Health probes del load balancer |
| 65500 | DenyAllInBound | Deniega todo lo demás |

**Outbound por defecto:**
| Prioridad | Nombre | Qué permite |
|-----------|--------|-------------|
| 65000 | AllowVnetOutBound | Tráfico dentro de la misma VNet |
| 65001 | AllowInternetOutBound | Salida a Internet |
| 65500 | DenyAllOutBound | Deniega todo lo demás |

### NSG en subnet vs NSG en NIC

```
Internet
    │
    ▼
[NSG subnet]  ← se evalúa primero para inbound
    │
    ▼
[VM con NIC]
    │
[NSG NIC]     ← se evalúa después
```

Para **inbound**: primero NSG de subnet, luego NSG de NIC.  
Para **outbound**: primero NSG de NIC, luego NSG de subnet.  

**Práctica habitual:** NSG en subnet (más fácil de gestionar). NSG en NIC solo para casos muy específicos.

### Caso bancario: NSG para 3 capas

```
snet-web  (NSG-web)
  Inbound: Allow 443 desde Internet
  Inbound: Allow 80 desde Internet (redirect a 443)
  Outbound: Allow 8080 hacia snet-app

snet-app  (NSG-app)
  Inbound: Allow 8080 desde snet-web ONLY
  Outbound: Allow 5432 hacia snet-data

snet-data  (NSG-data)
  Inbound: Allow 5432 desde snet-app ONLY
  Inbound: DENY todo desde Internet
```

Ninguna base de datos expuesta a internet. El tráfico solo fluye en la dirección correcta.

---

## 4. VNet Peering

**VNet Peering** conecta dos VNets para que sus recursos se comuniquen como si estuvieran en la misma red, usando la red backbone privada de Microsoft (sin pasar por Internet).

```
VNet-Hub (10.0.0.0/16)          VNet-Spoke-Prod (10.1.0.0/16)
┌──────────────────┐             ┌──────────────────┐
│  Azure Firewall  │◄────────────►│  AKS + SQL       │
│  VPN Gateway     │  Peering    │  App Services    │
└──────────────────┘             └──────────────────┘
         │
         │ Peering
         │
         ▼
VNet-Spoke-Dev (10.2.0.0/16)
┌──────────────────┐
│  VMs desarrollo  │
│  Test databases  │
└──────────────────┘
```

### Tipos de peering

| Tipo | Cuándo | Latencia |
|------|--------|----------|
| **Regional peering** | Misma región Azure | ~0ms extra |
| **Global peering** | Distintas regiones | Latencia de red entre regiones |

### Peering NO es transitivo

**Importante:** si Hub ↔ Spoke-Prod y Hub ↔ Spoke-Dev, Spoke-Prod y Spoke-Dev **no se ven entre sí** directamente. El tráfico tiene que pasar por Hub.

```
Spoke-Prod → Hub → Spoke-Dev   ✅ (con configuración correcta)
Spoke-Prod → Spoke-Dev         ❌ (no funciona sin peering directo)
```

Esto es por diseño — en Hub-Spoke, el Hub controla todo el tráfico entre spokes mediante Azure Firewall.

### Configuración: el peering es bidireccional pero se crea en ambos sentidos

```bash
# Peering de Hub hacia Spoke
az network vnet peering create \
  --name hub-to-spoke \
  --vnet-name vnet-hub \
  --resource-group rg-networking \
  --remote-vnet vnet-spoke-prod \
  --allow-vnet-access true

# Peering de Spoke hacia Hub (obligatorio — el peering no es automáticamente bidireccional)
az network vnet peering create \
  --name spoke-to-hub \
  --vnet-name vnet-spoke-prod \
  --resource-group rg-networking \
  --remote-vnet vnet-hub \
  --allow-vnet-access true
```

---

## 5. Comandos Azure CLI — Networking

```bash
# Crear VNet con su primer subnet
az network vnet create \
  --name vnet-prod-westeu \
  --resource-group rg-networking-prod \
  --location westeurope \
  --address-prefix 10.0.0.0/16 \
  --subnet-name snet-web \
  --subnet-prefix 10.0.1.0/24

# Añadir más subnets
az network vnet subnet create \
  --vnet-name vnet-prod-westeu \
  --resource-group rg-networking-prod \
  --name snet-app \
  --address-prefix 10.0.2.0/24

az network vnet subnet create \
  --vnet-name vnet-prod-westeu \
  --resource-group rg-networking-prod \
  --name snet-data \
  --address-prefix 10.0.3.0/24

# Crear NSG
az network nsg create \
  --name nsg-web \
  --resource-group rg-networking-prod \
  --location westeurope

# Añadir regla al NSG
az network nsg rule create \
  --nsg-name nsg-web \
  --resource-group rg-networking-prod \
  --name Allow-HTTPS \
  --priority 100 \
  --direction Inbound \
  --access Allow \
  --protocol Tcp \
  --destination-port-range 443

# Asociar NSG a subnet
az network vnet subnet update \
  --vnet-name vnet-prod-westeu \
  --resource-group rg-networking-prod \
  --name snet-web \
  --network-security-group nsg-web

# Ver las reglas efectivas de una NIC (muy útil para debugging)
az network nic list-effective-nsg \
  --name nic-vm-web-01 \
  --resource-group rg-prod
```

---

## 6. Caso práctico — Diseña la red de un banco mediano

**Escenario:** Banco con 3 entornos (prod, dev, shared-services). Migra a Azure.

**Requisitos:**
- Producción aislada de desarrollo
- Acceso a on-premise por VPN
- Firewall centralizado
- Bases de datos sin exposición a Internet

**Solución Hub-Spoke:**

```
Suscripción: sub-shared-services
  VNet-Hub: 10.0.0.0/16
    GatewaySubnet:        10.0.0.0/27  → VPN hacia on-premise
    AzureFirewallSubnet:  10.0.1.0/26  → inspección tráfico
    AzureBastionSubnet:   10.0.2.0/26  → acceso admin seguro

Suscripción: sub-prod
  VNet-Spoke-Prod: 10.1.0.0/16
    snet-web:   10.1.1.0/24  → App Gateway, WAF
    snet-app:   10.1.2.0/22  → AKS (necesita IPs para pods)
    snet-data:  10.1.6.0/24  → Azure SQL, Cosmos DB

Suscripción: sub-dev
  VNet-Spoke-Dev: 10.2.0.0/16
    snet-dev:   10.2.1.0/24  → VMs desarrollo
    snet-test:  10.2.2.0/24  → entorno test
```

**NSGs:**
- `nsg-web`: Allow 443/80 inbound desde Internet
- `nsg-app`: Allow solo desde `snet-web`
- `nsg-data`: Allow solo desde `snet-app`, DENY todo lo demás

**Peerings:**
- Hub ↔ Spoke-Prod (con `UseRemoteGateways` en Spoke para que use la VPN del Hub)
- Hub ↔ Spoke-Dev

---

## Mini-test — Respuestas

1. **¿Cuántas IPs utilizables tiene una subnet /26?**
   59 (64 totales - 5 reservadas por Azure)

2. **¿Por qué los rangos IP de VNets que van a hacer peering no pueden solaparse?**
   Porque el enrutamiento sería ambiguo — Azure no sabría a qué VNet dirigir el tráfico si ambas tienen el mismo rango.

3. **Una regla NSG con prioridad 200 y otra con prioridad 100 para el mismo tráfico — ¿cuál gana?**
   La de prioridad 100 (menor número = mayor prioridad). Se evalúan en orden ascendente y se aplica la primera que coincide.

4. **¿Por qué el peering se crea en ambos sentidos?**
   Azure crea el enlace en un sentido pero no habilita automáticamente el tráfico en el otro. Hay que crear el peering explícitamente desde cada VNet hacia la otra.

5. **En hub-spoke, ¿pueden Spoke-Prod y Spoke-Dev comunicarse directamente?**
   No. El peering no es transitivo. El tráfico entre spokes debe pasar por el Hub (Azure Firewall). Esto es por diseño — el Hub controla y audita todo el tráfico inter-spoke.
