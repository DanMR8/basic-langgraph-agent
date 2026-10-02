# basic-langgraph-agent

Agente conversacional con herramientas, construido con **LangGraph** y un **LLM local** (Ollama). Demuestra el patrón agente LLM + orquestación con grafos de estado, e incluye las tres piezas que se agregan antes de poner un agente en producción: **aprobación humana**, **streaming** y **memoria de sesión**.

## 🧠 Conceptos que demuestra

| Concepto | Dónde |
|---|---|
| Agente LLM | `agent/nodes.py` — el modelo decide qué herramienta usar |
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
> pide una, el grafo se suspende y te muestra qué quiere hacer y con qué
> argumentos. Responde `[s/N]`; si dices que no, el motivo se le devuelve al
> modelo y sigue pensando sin haber ejecutado nada.

## 🚀 Quickstart

### 1. Clonar e instalar

```bash
git clone https://github.com/DanMR8/basic-langgraph-agent.git
cd basic-langgraph-agent

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -e .
```

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

## 💬 Ejemplos de uso

```
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
      Entendido, no voy a consultar el clima sin tu autorización.
```

## 🗺️ Arquitectura del grafo

```
                    ┌──────────────────────────────┐
                    │  ¿autorizas?  interrupt()     │
                    ▼                              │
   ┌────────┐  tool_calls   ┌──────────┐  sí     ┌───────┐
   │ agent  │ ────────────► │ approve  │ ──────► │ tools │
   │ (LLM)  │               │ (gate)   │         │       │
   └───┬────┘               └────┬─────┘         └───┬───┘
       │                         │ no                 │
       │ ◄───────────────────────┘                   │
       │ ◄───────────────────────────────────────────┘
       │ respuesta sin herramientas
       ▼
      END
```

- **agent**: el LLM razona y decide si pide herramientas.
- **approve**: **punto de control humano**. `interrupt()` suspende el grafo y
  guarda el estado; se reanuda con `Command(resume="aprobar")`.
- **tools**: ejecuta solo lo ya aprobado y devuelve el resultado al agente.
- **Conditional edges**: el grafo elige la ruta en runtime, según la salida del
  LLM o según la decisión humana.

> El ciclo está acotado por diseño: cada vuelta al `agent` pasa por
> `approve`, así que **el humano es el freno real**. `recursion_limit` queda
> como red de seguridad para graphs patológicos, no como el mecanismo principal.

## 📁 Estructura

```
basic-langgraph-agent/
├── pyproject.toml         # Deps, metadatos y config de pytest/ruff
├── main.py                # CLI: streaming + gateway de aprobación
├── agent/
│   ├── graph.py           # Definición y compilación del grafo
│   ├── nodes.py           # Nodos: LLM, aprobación y ejecución
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

## 🔧 Personalización

- **Agregar herramientas**: define una función con `@tool` en `tools.py` y agrégala a `TOOLS`.
- **Cambiar modelo**: edita `OLLAMA_MODEL` en `.env` (cualquier modelo de Ollama).
- **Otro LLM local**: cambia `ChatOllama` por `ChatOpenAI` apuntando a LM Studio, llama.cpp, etc.
- **Ajustar el tope de pasos**: `RECURSION_LIMIT` en `config/settings.py`.
- **Quitar la aprobación**: borra el nodo `approve` y sus aristas en `graph.py`. Ojo: sin él,
  el LLM puede pedir herramientas en bucle hasta agotar `RECURSION_LIMIT`.

## 🧪 Tests

```bash
pip install -e ".[dev]"
pytest
```

La suite sustituye el LLM por un doble, así que **no necesita Ollama ni red**:
37 tests que cubren el ciclo de aprobación completo, el enrutado y las tres
herramientas.

## 📦 Dependencias principales

- [LangGraph](https://github.com/langchain-ai/langgraph) — orquestación con grafos, `interrupt()`, checkpointing
- [LangChain](https://github.com/langchain-ai/langchain) — abstracciones de LLM y herramientas
- [Ollama](https://ollama.com) — runtime de modelos locales

## 📄 Licencia

MIT — úsalo, modifícalo, compártelo.
