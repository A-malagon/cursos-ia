# CLAUDE.md — Contexto del proyecto

## Quién es el usuario

Alejandro, Manager en consultoría (Financial Services, KPMG — inicio 25 mayo 2026).
Tiene background de desarrollador y sabe programar, pero su rol en KPMG no es desarrollo sino:
- Entender componentes, arquitecturas y cómo se ensamblan
- Tomar decisiones técnicas óptimas en coste y ejecución
- Supervisar y dirigir equipos técnicos
- Hablar con clientes sobre soluciones cloud/IA

## Enfoque de trabajo acordado

**Dar el código directamente** — Alejandro lo copia, tú lo explicas. No esperar a que él lo escriba.

Para cada pieza de código o concepto:
1. Decir dónde va ("copia esto en `archivo.py`")
2. Dar el código completo
3. Explicar qué hace, por qué se eligió esa solución y no otra (trade-offs)
4. Conectar con casos reales de KPMG / banca / consultoría cuando aplique

## Estado del plan

El plan completo día a día está en `PLAN_21DIAS.md` (raíz del repo).
**Leerlo siempre al inicio de sesión** para saber en qué día real estamos y qué toca.

## Cierre de cada bloque — obligatorio

Al terminar cada bloque (sin esperar a que el usuario lo pida):

1. **Actualizar la tabla de estado** del README de la semana correspondiente: ⬜ → ✅
2. **Añadir la sección del día** al README con: arquitectura, conceptos clave, trade-offs, código relevante
3. **Actualizar PLAN_21DIAS.md**: marcar bloque ✅ y actualizar "PRÓXIMO PASO"
4. **Commit + push**:
   ```bash
   git add .
   git commit -m "CursoX SYdZ — tema completado"
   git push
   ```

## Dónde documentar cada bloque

| Bloque | README a actualizar |
|--------|---------------------|
| C1 Semana 1 (D1-D7) | `curso1_mlops/semana1/README.md` |
| C1 Semana 2 (D8-D14) | `curso1_mlops/semana2/README.md` |
| C1 Semana 3 (D15-D21) | `curso1_mlops/semana3/README.md` |
| C1 Semana 4 (D22-D28) | `curso1_mlops/semana4/README.md` |
| C2 Semana 1 (D1-D6) | `curso2_ai_engineer/semana1/README.md` |
| C2 Semana 2 (D7-D14) | `curso2_ai_engineer/semana2/README.md` |
| C3 Semana 1 (D1-D7) | `curso3_cloud_engineer/semana1/README.md` |
| C3 Semana 2 (D8-D14) | `curso3_cloud_engineer/semana2/README.md` |
| C3 Semana 3 (D15-D21) | `curso3_cloud_engineer/semana3/README.md` |
| C3 Semana 4 (D22-D30) | `curso3_cloud_engineer/semana4/README.md` |

## Formato del README por día

Cada día completado añade una sección al README de su semana con esta estructura:

```markdown
## Día N — Título

### Arquitectura / Diagrama (si aplica)

### Conceptos clave
Tabla o lista con componentes, qué hacen y por qué se usan

### Trade-offs y decisiones
Por qué esta solución y no otra (coste, complejidad, escalabilidad)

### Aplicación en KPMG
Cómo aplica esto en proyectos reales de banca/consultoría

### Código / Archivos
Referencias a los archivos creados
```

## Los 3 cursos

| Curso | Carpeta | Foco |
|-------|---------|------|
| C1 — MLOps Intensivo | `curso1_mlops/` | Docker, K8s, AKS, MLflow, CI/CD, monitorización |
| C2 — AI Engineer | `curso2_ai_engineer/` | LLMs, Agentes, RAG, Azure AI Foundry |
| C3 — Cloud Engineer | `curso3_cloud_engineer/` | Azure completo, networking, seguridad, FinOps, Well-Architected |
