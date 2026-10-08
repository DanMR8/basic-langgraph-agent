# basic-langgraph-agent

[![CI](https://github.com/DanMR8/basic-langgraph-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/DanMR8/basic-langgraph-agent/actions/workflows/ci.yml)

Agente conversacional en español con herramientas, construido con **LangGraph** y un **LLM local** (Ollama). Proyecto de muestra: demuestra un asistente que conversa, consulta datos externos bajo aprobación humana y responde con streaming y memoria de sesión.

## Qué hace

- Conversa en español de forma directa y concisa.
- Clasifica cada turno y usa la herramienta adecuada: calculadora, clima actual y búsqueda y listado de archivos locales.
- **Pide tu autorización antes de ejecutar** cualquier herramienta, mostrándote qué va a hacer y con qué argumentos.
- Si rechazas o algo falla, lo dice claramente y el turno termina sin inventar resultados.
- Recuerda la conversación dentro de la sesión.

## Herramientas incluidas

| Herramienta | Descripción |
|---|---|
| `calculator` | Aritmética segura (sin ejecutar código) |
| `get_weather` | Clima actual de una ciudad |
| `search_files` | Busca texto en archivos `.txt`/`.md` de `data/` |
| `list_files` | Lista qué archivos hay disponibles en `data/` |

## Quickstart

```bash
git clone https://github.com/DanMR8/basic-langgraph-agent.git
cd basic-langgraph-agent

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e .
```

```bash
ollama pull llama3.1:8b   # https://ollama.com
cp .env.example .env      # solo si usas otro modelo o puerto
python main.py
```

> En Windows, si tu consola no es UTF-8 usa Windows Terminal o `chcp 65001`,
> o ejecuta con `python -X utf8 main.py`.

## Ejemplos de uso

```
Tú › hola
🤖 › Hola, ¿en qué puedo ayudarte?

Tú › cuánto es (4 + 5) * 3 / 2
🤖 › ⚠️  El agente quiere ejecutar:
       • calculator(expression='(4 + 5) * 3 / 2')
       ¿Autorizas? [s/N] s
      ⚙️  Resultado: 13.5
      El resultado de la expresión (4+5)*3/2 es 13.5.

Tú › qué clima hace en Guadalajara
🤖 › ⚠️  El agente quiere ejecutar:
       • get_weather(city='Guadalajara')
       ¿Autorizas? [s/N] n
      Rechazaste el uso de la herramienta necesaria: get_weather. No se ejecutó y no tengo su resultado.
```

## Arquitectura

```
                        ┌─ texto ────────► hablador ──────────────────► respuesta
                        │
START ──► router ───────┤
                        │  herramienta
                        └────────────────► trabajador ──► approve ──┬─► tools ──► trabajador
                                                                   │                 (bucle acotado:
                                                                   │                  error→reintenta,
                                                                   │                  ok→redacta)
                                                                   └── rechazo ──► respuesta fija
```

El diseño separa tres responsabilidades que en un agente clásico van
mezcladas:

- **router** — Clasifica cada turno en una palabra (`TEXTO | CALCULO | CLIMA |
  ARCHIVOS`). Es la única pieza que decide, y decide poco a propósito: ante la
  duda, se habla, no se actúa.
- **hablador** — Responde la conversación sin herramientas vinculadas. Puede
  hablar de ellas pero nunca usarlas, así que por construcción no anuncia
  decisiones internas: solo existe la respuesta.
- **trabajador** — Mini-ciclo ReAct acotado: pide la herramienta de su ruta,
  espera aprobación humana, ejecuta y redacta. Solo ve las herramientas de su
  ruta y verifica que cada argumento venga del mensaje del usuario antes de
  molestar con la autorización.

Dos garantías estructurales, no promesas del prompt:

1. **Nada se ejecuta sin aprobación.** El grafo se suspende (`interrupt`) y se
   reanuda con tu respuesta; sin ella, la herramienta no corre.
2. **Todo cierre es determinista.** Rechazo, dato sin respaldo y reintentos
   agotados terminan en mensajes fijos, no en improvisación del modelo.

El estado (`mensajes` + `ruta` + `reintentos`) vive en un checkpoint por
sesión (`thread_id`), con `recursion_limit` como red de seguridad.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

La suite sustituye el LLM por un doble, así que **no necesita Ollama ni red**.

## Dependencias principales

- [LangGraph](https://github.com/langchain-ai/langgraph) — orquestación con grafos, `interrupt()`, checkpointing
- [LangChain](https://github.com/langchain-ai/langchain) — abstracciones de LLM y herramientas
- [Ollama](https://ollama.com) — runtime de modelos locales

## Licencia

MIT — úsalo, modifícalo, compártelo.
