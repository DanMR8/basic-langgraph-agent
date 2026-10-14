# basic-langgraph-agent

[![CI](https://github.com/DanMR8/basic-langgraph-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/DanMR8/basic-langgraph-agent/actions/workflows/ci.yml)

Agente conversacional en español con herramientas, construido con **LangGraph** y un **LLM local** (Ollama). Proyecto de muestra: consulta disponibilidad de bicis Ecobici en tiempo real, y además calcula y da el clima, siempre bajo aprobación humana, con streaming y memoria de sesión.

<video src="https://github.com/user-attachments/assets/6856d162-a0d9-4c5d-9480-b195f61a6fb7" controls width="100%"></video>

## Qué hace

- Conversa en español de forma directa y concisa.
- Consulta disponibilidad de bicis **Ecobici en tiempo real** (más calculadora y clima actual) clasificando cada turno.
- Si hay pocas bicis, te ofrece revisar estaciones cercanas antes de cerrar.
- Si preguntas dónde estás, eliges ubicación precisa (GPS) o aproximada.
- **Pide tu autorización antes de ejecutar** cualquier herramienta, mostrándote qué va a hacer y con qué argumentos.
- Si rechazas o algo falla, lo dice claramente y el turno termina sin inventar resultados.
- Recuerda la conversación dentro de la sesión.

## Herramientas incluidas

| Herramienta | Descripción |
|---|---|
| `ecobici` | Bicis y espacios libres por cicloestación (CDMX, tiempo real, sin key) |
| `ecobici_cercanas` | Bicis disponibles cerca de una estación (radio 1 km) |
| `calculator` | Aritmética segura (sin ejecutar código) |
| `get_weather` | Clima actual de una ciudad |
| `clima_aqui` | Clima actual de tu ubicación aproximada (por IP, sin ciudad) |
| `ubicacion` | Tu ciudad aproximada por IP (solo si preguntas dónde estás) |
| `geocodificar_inverso` | Colonia y alcaldía de unas coordenadas (solo uso interno) |

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
```

### Terminal (CLI)

```bash
python main.py
```

### App completa (Flet)

```bash
python -m front.app
```

La app abre ventana propia con chat, burbujas y diálogos de aprobación. Para
la ubicación precisa por GPS en Windows: `pip install -e ".[gps]"` y activa
Ubicación en Configuración del sistema.

> En Windows, si tu consola no es UTF-8 usa Windows Terminal o `chcp 65001`,
> o ejecuta con `python -X utf8 main.py`.

## Ejemplos de uso

```
Tú › hola
🤖 › Hola, ¿en qué puedo ayudarte?

Tú › ¿hay bicis en reforma?
🤖 › ⚠️  El agente quiere ejecutar:
       • ecobici(estacion='reforma')
       ¿Autorizas? [s/N] s
      ⚙️  Ecobici — 'reforma' (dato de hace 11s):
      - 24 coincidencias, top 5 por bicis:
      | Estación | Bicis | Espacios | Estado |
      |---|---|---|---|
      | CE-261 Paseo de la Reforma-Río de la Plata (261) | 42 | 0 | abierta |
      | CE-022 Reforma - Manchester (022) | 35 | 0 | abierta |
      - …y 19 más con 'reforma'.
      Hay bicis en Reforma: la de Paseo de la Reforma-Río de la Plata tiene 42.

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

Tú › ¿dónde estoy?
🤖 › ⚠️  El agente quiere ejecutar:
       • ubicacion()
       ¿Autorizas? [s/N] s
      Estás en Naucalpan, México (aproximado por IP). No es tu dirección exacta.
```

## Arquitectura

```
                        ┌─ texto ────────► hablador ──────────────────► respuesta
                        │
START ──► router ───────┤
                        │  herramienta
                        └────────────────► trabajador ──► approve ──┬─► tools ──┬─► trabajador
                                                                   │           │         (redacta / reintenta
                                                                   │           │          con presupuesto)
                                                                   │           ├─► ofrecer ──► tools ──► resumen
                                                                   │           │     (s/n)      (2.º eslabón)
                                                                   │           └─► fallo (sin reintentos)
                                                                   └── rechazo ──► respuesta fija
```

"dónde estoy" tiene rama propia (precisa con GPS, aproximada con IP):

```
    trabajador ──► elegir ──┬─ precisa ──► geocodifica ──► respuesta
                             ├─ aproximada ──► approve ──► tools ──► respuesta
                             └─ rechazo ──► respuesta fija
```

El diseño separa tres responsabilidades que en un agente clásico van
mezcladas:

- **router** — Clasifica cada turno en una palabra (`TEXTO | CALCULO | CLIMA |
  MOVILIDAD | UBICACION`). Es la única pieza que decide, y decide poco a propósito: ante la
  duda, se habla, no se actúa.
- **hablador** — Responde la conversación sin herramientas vinculadas. Puede
  hablar de ellas pero nunca usarlas, así que por construcción no anuncia
  decisiones internas: solo existe la respuesta.
- **trabajador** — Mini-ciclo ReAct acotado: pide la herramienta de su ruta,
  espera aprobación humana, ejecuta y redacta. Solo ve las herramientas de su
  ruta y verifica que cada argumento venga del mensaje del usuario antes de
  molestar con la autorización.
- **ofrecer / resumir / fallo** — Si hay pocas bicis, ofrece revisar cercanas
  (el "sí" ya autoriza); el cierre del segundo eslabón es determinista y los
  reintentos agotados terminan en mensaje fijo.

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

## Interfaz

Además de la terminal (`python main.py`), hay front en Flet (`python -m front.app`):
chat con burbujas, diálogos de aprobación y toggle de ubicación precisa.

## Dependencias principales

- [LangGraph](https://github.com/langchain-ai/langgraph) — orquestación con grafos, `interrupt()`, checkpointing
- [LangChain](https://github.com/langchain-ai/langchain) — abstracciones de LLM y herramientas
- [Ollama](https://ollama.com) — runtime de modelos locales
- [Flet](https://flet.dev) — interfaz de escritorio (`python -m front.app`)

## Licencia

MIT — úsalo, modifícalo, compártelo.
