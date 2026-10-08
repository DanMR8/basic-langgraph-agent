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
router ─────── texto ──────────────► hablador ─────────────────► END
  │
  │ herramienta (CALCULO | CLIMA | ARCHIVOS)
  ▼
trabajador ─── pide ──► approve ──┬── autorizaste ──► tools ──► trabajador (bucle)
  ▲                              │                                  ├─ error ──► reintenta (máx 2)
  │                              │                                  ├─ agotado ──► fallo ──► END
  │                              │                                  └─ ok ──► redacta ──► END
  │                              └── rechazaste / dato inventado ──► mensaje fijo ──► END
  └────────────────────────────── (el trabajador solo ve las herramientas de su ruta)
```

- **router**: clasifica cada turno en UNA palabra (`TEXTO | CALCULO | CLIMA |
  ARCHIVOS`) a temperatura 0.0. Lo desconocido va a `TEXTO`: ante la duda se
  habla, no se actúa.
- **hablador**: responde en texto **sin herramientas vinculadas**. Las conoce
  en texto (puede describirlas) pero no puede usarlas: no hay decisión que
  anunciar porque nunca se le ofreció ninguna.
- **trabajador**: pide la herramienta de su ruta (solo ve ese subconjunto) o
  redacta el resultado. Si falta el dato no lo inventa: responde en texto
  pidiéndolo.
- **approve**: **punto de control humano**. `interrupt()` suspende el grafo y
  guarda el estado; se reanuda con `Command(resume="aprobar")`. Antes de
  preguntar verifica que la ciudad o el término vengan del mensaje del
  usuario; si el modelo los inventó, ni molesta: mensaje fijo pidiendo el dato.
- **tools**: ejecuta solo lo ya aprobado.
- **fallo**: presupuesto de 2 reintentos agotado. Mensaje fijo con el motivo,
  fin del turno. Lo terminal (rechazo, dato inventado, fallo) es siempre un
  mensaje fijo, nunca una narración del LLM.

> El ciclo está acotado por estructura, no por `recursion_limit`: el
> presupuesto de reintentos más el gate humano cierran el bucle.
> `recursion_limit` queda como red de seguridad, no como freno.

## 📁 Estructura

```
basic-langgraph-agent/
├── pyproject.toml         # Deps, metadatos y config de pytest/ruff
├── main.py                # CLI: streaming + gateway de aprobación
├── agent/
│   ├── graph.py           # Definición y compilación del grafo
  │   ├── nodes.py           # Nodos: router, hablador, trabajador, aprobar, ejecutar y fallo
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
- **Temperatura por nodo**: `TEMPERATURA_ROUTER`, `TEMPERATURA_HABLADOR` y
  `TEMPERATURA_TRABAJADOR` en `config/settings.py`. Ahora mismo 0.0 / 0.5 /
  0.5. El router se queda en 0 porque clasifica en una palabra y ahí lo que
  importa es repetir, no improvisar.
- **Otro LLM local**: cambia `ChatOllama` por `ChatOpenAI` apuntando a LM Studio, llama.cpp, etc.
- **Ajustar el tope de pasos**: `RECURSION_LIMIT` en `config/settings.py`.
- **Quitar la aprobación**: borra el nodo `approve` y sus aristas en `graph.py`.
  Ojo: sin él, nada discute lo que el trabajador pide.
- **Ajustar el router**: los ejemplos de `ROUTER_PROMPT` en
  `config/settings.py` deciden la ruta de cada turno; cambia ahí qué cuenta
  como `CALCULO`, `CLIMA` o `ARCHIVOS`.
- **Ajustar los mensajes fijos**: `mensaje_rechazo`, `mensaje_falta_dato` y
  `mensaje_fallo` en `config/settings.py`. Son deterministas: el LLM no los
  redacta, solo los muestra.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

La suite sustituye el LLM por un doble, así que **no necesita Ollama ni red**:
44 tests que cubren el enrutado, la rama texto, la aprobación con `interrupt()`,
el respaldo de argumentos, los reintentos con presupuesto, los mensajes fijos
y las cuatro herramientas.

## Dependencias principales

- [LangGraph](https://github.com/langchain-ai/langgraph) — orquestación con grafos, `interrupt()`, checkpointing
- [LangChain](https://github.com/langchain-ai/langchain) — abstracciones de LLM y herramientas
- [Ollama](https://ollama.com) — runtime de modelos locales

## Licencia

MIT — úsalo, modifícalo, compártelo.
