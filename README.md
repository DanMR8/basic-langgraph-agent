# basic-langgraph-agent

[![CI](https://github.com/DanMR8/basic-langgraph-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/DanMR8/basic-langgraph-agent/actions/workflows/ci.yml)

Agente conversacional con herramientas, construido con **LangGraph** y un **LLM local** (Ollama). Demuestra el patrón agente LLM + orquestación con grafos de estado, e incluye las tres piezas que se agregan antes de poner un agente en producción: **aprobación humana**, **streaming** y **memoria de sesión**.

| Concepto | Dónde |
|---|---|
| Agente LLM | `agent/nodes.py` — el modelo propone, un validador decide si merece la pena |
| Orquestación con grafos | `agent/graph.py` — flujo explícito con ciclos |
| Estado compartido | `agent/state.py` — memoria entre nodos |
| **Human-in-the-loop** | `agent/nodes.py` — `interrupt()` + `Command(resume=...)`: el agente **pregunta antes de actuar** |
| **Checkpointing** | `agent/graph.py` — `InMemorySaver` + `thread_id`: una sesión por hilo |
| **Streaming** | `main.py` — la respuesta llega token a token |
| **Límite de pasos** | `recursion_limit` de LangGraph, con `GraphRecursionError` |
| Herramientas | `agent/tools.py` — calculadora, clima, búsqueda en archivos |
| Modelo local | Ollama — sin dependencia de APIs en la nube |

## 🛠️ Herramientas incluidas

| Herramienta | Descripción |
|---|---|
| `calculator` | Evalúa aritmética sobre el AST, sin ejecutar código ni agotar memoria |
| `get_weather` | Clima actual vía [wttr.in](https://wttr.in) (sin API key) |
| `search_files` | Busca texto en archivos `.txt`/`.md` de `data/` |

> ⚠️ **Ninguna herramienta se ejecuta sin tu autorización.** Cuando el modelo
> pide una, un **validador** decide primero si la petición tiene sentido; si la
> veta, ni te enteras y el agente responde en texto. Si la deja pasar, el grafo
> se suspende y te muestra qué quiere hacer y con qué argumentos. Responde
> `[s/N]`; si dices que no, el turno termina sin ejecutar nada.

## 🚀 Quickstart

### 1. Clonar e instalar

```bash
git clone https://github.com/DanMR8/basic-langgraph-agent.git
cd basic-langgraph-agent

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e .
```

> En Windows, si tu consola no es UTF-8 verás `UnicodeEncodeError` al imprimir
> los emojis. Usa Windows Terminal o `chcp 65001`, o redirige con
> `python -X utf8 main.py`.

### 2. Levantar Ollama

```bash
# Instalar Ollama: https://ollama.com
ollama pull llama3.1:8b
ollama serve               # si no está corriendo ya
```

### 3. Configurar

```bash
cp .env.example .env
# Editar .env si usas otro modelo o puerto
```

### 4. Ejecutar

```bash
python main.py
```

## Ejemplos de uso

```
Tú › hola
🤖 › Hola, ¿en qué puedo ayudarte hoy?

Tú › cuánto es (4 + 5) * 3 / 2
🤖 › ⚠️  El agente quiere ejecutar:
       • calculator(expression='(4 + 5) * 3 / 2')
       ¿Autorizas? [s/N] s
      ⚙️  Resultado: 13.5
      El resultado es 13.5

Tú › qué clima hace en Guadalajara
🤖 › ⚠️  El agente quiere ejecutar:
       • get_weather(city='Guadalajara')
       ¿Autorizas? [s/N] n
      🚫  Operación rechazada por el usuario: get_weather. La herramienta NO se ejecutó.

Tú › acabas de consultarlo verdad
🤖 › No, no lo he consultado. Solo te he respondido en texto sobre lo que es un
     grafo de estado y sobre la expresión matemática que me pediste calcular.
```
---
##  Arquitectura del grafo

```
START
 │
 ▼
conversar ──── respondió en texto ──────────────────────────► END
  │
  │ tool_calls
  ▼
validar ────── vetó la petición ──► responder ──────────────► END
  │
  │ la deja pasar
  ▼
approve ────── rechazaste ──────────────────────────────────► END
  │
  │ autorizaste
  ▼
tools ────────► responder ──────────────────────────────────► END
```

- **conversar**: abre el turno hablando con el usuario. Con herramientas
  vinculadas tiende a pedirlas casi siempre — de un saludo le saca
  `calculator(expression='hola')` —, así que su juicio no es fiable y su
  petición no va directo al humano.
- **validar**: la puerta que faltaba. Ve la pregunta **y** la propuesta
  concreta con sus argumentos, y responde `SI` o `NO`. Si veta, deja un
  `ToolMessage` de texto fijo que cierra la `tool_call` abierta —la señal de
  que hubo veto, no un registro que deba conservarse— sin tocar `AgentState`,
  que sigue siendo solo `messages`.
- **approve**: **punto de control humano**. `interrupt()` suspende el grafo y
  guarda el estado; se reanuda con `Command(resume="aprobar")`.
- **tools**: ejecuta solo lo ya aprobado y pasa el resultado a `responder`.
- **responder**: cierra el turno **sin herramientas**. Es lo que garantiza que
  el ciclo termine siempre en una respuesta: no puede volver a pedir nada, ni
  siquiera volviendo a `conversar`. Si el turno cierra con un veto, se le
  añade al `system prompt` de esa llamada —y solo ahí— la orden de no
  inventarse el resultado, y en el mismo update borra el par `tool_call` +
  veto con `RemoveMessage`. Lo que queda en el historial es pregunta y
  respuesta, como en un chat normal: escrito en un mensaje, tanto la orden
  como el registro salían en la boca del asistente tres turnos después y a
  partir de ahí se repetían solos.
- **Conditional edges**: el grafo elige la ruta en runtime; `validar` en cambio
  devuelve `Command(goto=...)` para decidir sin añadir campos al estado.

> El ciclo está acotado por estructura, no por `recursion_limit`: `responder`
> va directo a `END`, así que una aprobación es una ejecución y el turno se
> cierra. `recursion_limit` queda como red de seguridad, no como freno.

## 📁 Estructura

```
basic-langgraph-agent/
├── pyproject.toml         # Deps, metadatos y config de pytest/ruff
├── main.py                # CLI: streaming + gateway de aprobación
├── agent/
│   ├── graph.py           # Definición y compilación del grafo
│   ├── nodes.py           # Nodos: conversar, validar, aprobar, ejecutar
│   ├── state.py           # Estado compartido
│   └── tools.py           # Herramientas del agente
├── config/
│   ├── __init__.py
│   └── settings.py        # Config centralizada
├── tests/
│   ├── conftest.py        # LLM falso: la suite corre sin Ollama
│   ├── test_graph.py      # Aprobación, enrutado y límites
│   └── test_tools.py      # Calculadora y búsqueda
└── data/                  # Archivos para search_files
```

## Personalización

- **Agregar herramientas**: define una función con `@tool` en `tools.py` y agrégala a `TOOLS`.
- **Cambiar modelo**: edita `OLLAMA_MODEL` en `.env` (cualquier modelo de Ollama).
- **Temperatura por nodo**: `TEMPERATURA_CONVERSAR`, `TEMPERATURA_RESPONDER` y
  `TEMPERATURA_VALIDADOR` en `config/settings.py`. Ahora mismo 0.5 / 0.7 / 0.
  El validador se queda en 0 porque clasifica SI/NO a partir de ejemplos y ahí
  lo que importa es repetir, no improvisar. `conversar` es el único con
  herramientas vinculadas y por eso no sube de 0.5: medido con llama3.1:8b, a
  0.7 escribía la llamada como texto en 2 de 8 sesiones. La tabla completa de
  medidas está en el comentario de la constante.
- **Otro LLM local**: cambia `ChatOllama` por `ChatOpenAI` apuntando a LM Studio, llama.cpp, etc.
- **Ajustar el tope de pasos**: `RECURSION_LIMIT` en `config/settings.py`.
- **Quitar la aprobación**: borra el nodo `approve` y sus aristas en `graph.py`. Ojo: sin él,
  el validador deja pasar lo que le parezca necesario y nada se lo discute.
- **Ajustar el umbral del validador**: los ejemplos de `VALIDADOR_PROMPT` en
  `config/settings.py` decidieron el 8/8 de las pruebas; cambia ahí qué cuenta
  como herramienta necesaria.
- **Ajustar la orden de veto**: `INSTRUCCION_VETO` en `config/settings.py`. Se
  inyecta en el `system prompt` de `responder` solo cuando el turno cierra con
  un veto, y no se guarda en ningún sitio.
- **Ajustar el tono**: `SYSTEM_PROMPT` en `config/settings.py`. El formato de
  sus ejemplos decide lo que el asistente dice: enunciar la decisión
  (`hola -> responde en texto`) hacía que el modelo la anunciara en cada
  respuesta; un diálogo de ejemplo le enseña el formato sin nombrarla. La
  tabla de medidas está en su comentario.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

La suite sustituye el LLM por un doble, así que **no necesita Ollama ni red**:
56 tests que cubren el ciclo de aprobación completo, el validador, el enrutado,
el historial que deja cada turno, la temperatura de cada nodo y las tres
herramientas.

## Dependencias principales

- [LangGraph](https://github.com/langchain-ai/langgraph) — orquestación con grafos, `interrupt()`, checkpointing
- [LangChain](https://github.com/langchain-ai/langchain) — abstracciones de LLM y herramientas
- [Ollama](https://ollama.com) — runtime de modelos locales

## Licencia

MIT — úsalo, modifícalo, compártelo.
