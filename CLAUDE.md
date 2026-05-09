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

Al cerrar cada bloque:
1. Actualizar README de la semana correspondiente (⬜ → ✅ + contenido)
2. Marcar el bloque en PLAN_21DIAS.md y actualizar "PRÓXIMO PASO"
3. Commit + push

## Los 3 cursos

| Curso | Carpeta | Foco |
|-------|---------|------|
| C1 — MLOps Intensivo | `curso1_mlops/` | Docker, K8s, AKS, MLflow, CI/CD, monitorización |
| C2 — AI Engineer | `curso2_ai_engineer/` | LLMs, Agentes, RAG, Azure AI Foundry |
| C3 — Cloud Engineer | `curso3_cloud_engineer/` | Azure completo, networking, seguridad, FinOps, Well-Architected |
