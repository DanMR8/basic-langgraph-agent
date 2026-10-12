"""Tests del grafo: router, rama texto, ReAct con gate humano y reintentos.

Usan un LLM falso (ver `conftest.py`), así que la suite corre sin Ollama. Las
colas vacías significan "este nodo no debe hablar": el doble falla si lo
invocan.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from agent.graph import agent
from agent.nodes import _es_error, _pide_ubicacion, _rescatar_llamada, route_router
from agent.state import AgentState
from agent.tools import TOOLS, list_files
from config.settings import (
    TEMPERATURA_HABLADOR,
    TEMPERATURA_ROUTER,
    TEMPERATURA_TRABAJADOR,
)
from tests.conftest import pide_herramienta, responde


def textos_ai(grafo, cfg) -> list[str]:
    return [
        m.content
        for m in grafo.get_state(cfg).values["messages"]
        if m.type == "ai" and not getattr(m, "tool_calls", None)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Estructura
# ─────────────────────────────────────────────────────────────────────────────


def test_nodos_del_grafo():
    assert set(agent.get_graph().nodes) == {
        "__start__",
        "router",
        "hablador",
        "trabajador",
        "approve",
        "tools",
        "fallo",
        "__end__",
    }


def test_herramientas_expuestas():
    assert {t.name for t in TOOLS} == {
        "calculator",
        "get_weather",
        "ecobici",
    }


def test_estado_lleva_ruta_y_reintentos():
    assert set(AgentState.__annotations__) == {"messages", "ruta", "reintentos"}


def test_temperatura_segun_el_trabajo_de_cada_nodo():
    """El que clasifica no varía; los que generan, sí."""
    assert TEMPERATURA_ROUTER == 0.0
    assert (TEMPERATURA_HABLADOR, TEMPERATURA_TRABAJADOR) == (0.5, 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Router y rama texto
# ─────────────────────────────────────────────────────────────────────────────


def test_texto_no_toca_herramientas(montar_grafo):
    grafo, cfg = montar_grafo(router="TEXTO", habla=[responde("Hola")])
    grafo.invoke({"messages": [HumanMessage(content="hola")]}, config=cfg)
    assert textos_ai(grafo, cfg) == ["Hola"]
    assert grafo.get_state(cfg).values["ruta"] == "TEXTO"


def test_etiqueta_desconocida_va_a_texto(montar_grafo):
    grafo, cfg = montar_grafo(router="CUALQUIERA", habla=[responde("Hola")])
    grafo.invoke({"messages": [HumanMessage(content="hola")]}, config=cfg)
    assert textos_ai(grafo, cfg) == ["Hola"]


@pytest.mark.parametrize(
    ("ruta", "esperado"),
    [
        ("TEXTO", "texto"),
        ("CALCULO", "herramienta"),
        ("CLIMA", "herramienta"),
        ("MOVILIDAD", "herramienta"),
    ],
)
def test_route_router(ruta, esperado):
    assert route_router({"ruta": ruta}) == esperado


# ─────────────────────────────────────────────────────────────────────────────
# Aprobación humana y mensajes fijos
# ─────────────────────────────────────────────────────────────────────────────


def test_calculo_aprobado_responde_con_resultado(montar_grafo):
    grafo, cfg = montar_grafo(
        router="CALCULO",
        trabaja=[
            pide_herramienta("calculator", {"expression": "2 + 2"}),
            responde("Es 4."),
        ],
    )
    grafo.invoke({"messages": [HumanMessage(content="cuánto es 2+2")]}, config=cfg)
    assert grafo.get_state(cfg).next == ("approve",)

    grafo.invoke(Command(resume="aprobar"), config=cfg)
    assert textos_ai(grafo, cfg) == ["Es 4."]
    assert grafo.get_state(cfg).values["reintentos"] == 0


def test_rechazo_es_mensaje_fijo_y_no_ejecuta(montar_grafo):
    grafo, cfg = montar_grafo(
        router="CALCULO",
        trabaja=[pide_herramienta("calculator", {"expression": "2 + 2"})],
    )
    grafo.invoke({"messages": [HumanMessage(content="cuánto es 2+2")]}, config=cfg)
    grafo.invoke(Command(resume="rechazar"), config=cfg)

    assert textos_ai(grafo, cfg) == [
        "Rechazaste el uso de la herramienta necesaria: calculator. "
        "No se ejecutó y no tengo su resultado."
    ]
    assert "Resultado:" not in str(grafo.get_state(cfg).values["messages"])


def test_sin_resume_la_herramienta_no_se_ejecuta(montar_grafo):
    grafo, cfg = montar_grafo(
        router="CALCULO",
        trabaja=[pide_herramienta("calculator", {"expression": "2 + 2"})],
    )
    grafo.invoke({"messages": [HumanMessage(content="cuánto es 2+2")]}, config=cfg)
    assert grafo.get_state(cfg).next == ("approve",)
    assert [
        m for m in grafo.get_state(cfg).values["messages"] if isinstance(m, ToolMessage)
    ] == []


# ─────────────────────────────────────────────────────────────────────────────
# Respaldo de argumentos: lo inventado ni pide permiso
# ─────────────────────────────────────────────────────────────────────────────


def test_ciudad_inventada_ni_siquiera_pide_permiso(montar_grafo):
    """Sin ciudad en el mensaje no hay interrupt: fijo y fin de turno."""
    grafo, cfg = montar_grafo(
        router="CLIMA",
        trabaja=[pide_herramienta("get_weather", {"city": "Madrid"})],
    )
    grafo.invoke(
        {"messages": [HumanMessage(content="cómo está el clima?")]}, config=cfg
    )
    assert grafo.get_state(cfg).next == ()
    assert textos_ai(grafo, cfg) == [
        "Me falta la ciudad en tu mensaje: dime de qué ciudad quieres "
        "el clima (p. ej. 'Madrid')."
    ]


def test_ciudad_respaldada_si_pide_permiso(montar_grafo):
    grafo, cfg = montar_grafo(
        router="CLIMA",
        trabaja=[pide_herramienta("get_weather", {"city": "Madrid"})],
    )
    grafo.invoke({"messages": [HumanMessage(content="clima en Madrid?")]}, config=cfg)
    assert grafo.get_state(cfg).next == ("approve",)


def test_query_normalizada_con_respaldo(montar_grafo):
    """"lealtad" respalda la petición aunque el nombre completo sea largo."""
    grafo, cfg = montar_grafo(
        router="MOVILIDAD",
        trabaja=[
            pide_herramienta("ecobici", {"estacion": "lealtad"}),
            responde("Listo."),
        ],
    )
    grafo.invoke(
        {"messages": [HumanMessage(content="bicis en Glorieta de la Lealtad")]},
        config=cfg,
    )
    assert grafo.get_state(cfg).next == ("approve",)


def test_estacion_inventada_ni_siquiera_pide_permiso(montar_grafo):
    grafo, cfg = montar_grafo(
        router="MOVILIDAD",
        trabaja=[pide_herramienta("ecobici", {"estacion": "Reforma"})],
    )
    grafo.invoke({"messages": [HumanMessage(content="¿hay bicis?")]}, config=cfg)
    assert grafo.get_state(cfg).next == ()
    assert textos_ai(grafo, cfg) == [
        "Me falta una referencia en tu mensaje: dime la calle donde está "
        "(p. ej. 'Reforma', '20 de Noviembre' o 'Insurgentes') o el número de la cicloestación "
        "(p. ej. '261')."
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Reintentos con presupuesto
# ─────────────────────────────────────────────────────────────────────────────


def test_error_reintenta_una_vez_con_el_motivo(montar_grafo):
    grafo, cfg = montar_grafo(
        router="CALCULO",
        trabaja=[
            pide_herramienta("calculator", {"expression": "sqrt(16)"}),
            pide_herramienta("calculator", {"expression": "16 ** 0.5"}),
            responde("Es 4."),
        ],
    )
    grafo.invoke({"messages": [HumanMessage(content="raíz de 16")]}, config=cfg)
    grafo.invoke(Command(resume="aprobar"), config=cfg)
    assert grafo.get_state(cfg).next == ("approve",)
    grafo.invoke(Command(resume="aprobar"), config=cfg)

    assert textos_ai(grafo, cfg) == ["Es 4."]
    assert grafo.get_state(cfg).values["reintentos"] == 1


def test_presupuesto_agotado_termina_con_fijo(montar_grafo):
    grafo, cfg = montar_grafo(
        router="CALCULO",
        trabaja=[
            pide_herramienta("calculator", {"expression": "sqrt(16)"}) for _ in range(3)
        ],
    )
    grafo.invoke({"messages": [HumanMessage(content="raíz de 16")]}, config=cfg)
    grafo.invoke(Command(resume="aprobar"), config=cfg)
    grafo.invoke(Command(resume="aprobar"), config=cfg)
    grafo.invoke(Command(resume="aprobar"), config=cfg)

    final = textos_ai(grafo, cfg)
    assert len(final) == 1 and final[0].startswith("La herramienta falló dos veces")
    assert grafo.get_state(cfg).next == ()


def test_sin_coincidencias_no_se_reintenta():
    """"No encontré" no es un error: la consulta funcionó, se redacta."""
    from agent.nodes import route_tools

    estado = {
        "messages": [ToolMessage(content="No encontré estación con 'z'.", tool_call_id="1")],
        "reintentos": 0,
    }
    assert route_tools(estado) == "trabajador"


def test_recursion_limit_sigue_siendo_un_freno(montar_grafo):
    grafo, cfg = montar_grafo(
        router="CALCULO",
        trabaja=[pide_herramienta("calculator", {"expression": "1 + 1"})],
    )
    cfg = {**cfg, "recursion_limit": 1}

    with pytest.raises(GraphRecursionError):
        grafo.invoke({"messages": [HumanMessage(content="empieza")]}, config=cfg)


def test_es_error_solo_lo_reintentable():
    assert _es_error(
        ToolMessage(content="Error: división entre cero.", tool_call_id="1")
    )
    assert _es_error(
        ToolMessage(
            content="No pude obtener el clima de 'X': timeout", tool_call_id="1"
        )
    )
    assert not _es_error(ToolMessage(content="No se encontró 'z'.", tool_call_id="1"))
    assert not _es_error(AIMessage(content="Error: esto es texto, no un fallo."))


def test_list_files_muestra_sample():
    assert "sample.txt" in str(list_files.invoke({}))


# ─────────────────────────────────────────────────────────────────────────────
# Rescate: JSON escrito como texto se convierte en llamada real
# ─────────────────────────────────────────────────────────────────────────────


def test_rescatar_llamada_valida():
    nombre, args = _rescatar_llamada(
        '{"name": "ecobici", "parameters": {"estacion": "reforma"}}'
    )
    assert (nombre, args) == ("ecobici", {"estacion": "reforma"})


def test_rescatar_llamada_acepta_args():
    nombre, args = _rescatar_llamada(
        '{"name": "calculator", "args": {"expression": "2+2"}}'
    )
    assert (nombre, args) == ("calculator", {"expression": "2+2"})


def test_rescatar_llamada_acepta_booleanos_python():
    """El modelo escribe True/False, no true/false: igual rescata."""
    nombre, args = _rescatar_llamada(
        '{"name": "ecobici", "parameters": {"conclusion": False, "estacion": "x"}}'
    )
    assert (nombre, args) == ("ecobici", {"conclusion": False, "estacion": "x"})


def test_rescatar_llamada_acepta_cercas_markdown():
    nombre, args = _rescatar_llamada(
        '```json\n{"name": "ecobici", "parameters": {"estacion": "x"}}\n```'
    )
    assert (nombre, args) == ("ecobici", {"estacion": "x"})


@pytest.mark.parametrize(
    "texto",
    [
        "Hola, ¿en qué puedo ayudarte?",
        '{"name": "inventada", "parameters": {}}',
        '{"name": "ecobici", "parameters": "reforma"}',
        '{"name": "ecobici"',
        "No tengo acceso a tu ubicación.",
    ],
)
def test_rescatar_llamada_rechaza(texto):
    assert _rescatar_llamada(texto) is None


def test_json_como_texto_sigue_a_aprobacion(montar_grafo):
    """El trabajador escribió la llamada en texto: se rescata con gate."""
    grafo, cfg = montar_grafo(
        router="MOVILIDAD",
        trabaja=[
            responde('{"name": "ecobici", "parameters": {"estacion": "reforma"}}'),
            responde("Listo."),
        ],
    )
    grafo.invoke({"messages": [HumanMessage(content="hay bicis en reforma?")]}, config=cfg)
    pendientes = grafo.get_state(cfg)
    assert pendientes.next == ("approve",)
    herramientas = [
        i for t in pendientes.tasks for i in t.interrupts
    ][0].value["herramientas"]
    assert herramientas == [{"name": "ecobici", "args": {"estacion": "reforma"}}]


# ─────────────────────────────────────────────────────────────────────────────
# Ubicación del usuario: detector + frase con backstop
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("hay bicis cerca de mí", True),
        ("bicis donde estoy", True),
        ("mi ubicación actual", True),
        ("algo cerca de aquí", True),
        ("cerca de la Lealtad", False),
        ("hay bicis en reforma", False),
        ("hola", False),
    ],
)
def test_detector_ubicacion(texto, esperado):
    assert _pide_ubicacion(texto) is esperado


def test_ubicacion_compuesta_frase_mas_guia(montar_grafo):
    grafo, cfg = montar_grafo(
        router="MOVILIDAD",
        trabaja=[pide_herramienta("ecobici", {"estacion": "cercanas"})],
        frase=[responde("No tengo acceso a tu ubicación.")],
    )
    grafo.invoke({"messages": [HumanMessage(content="bicis cerca de mí")]}, config=cfg)
    assert grafo.get_state(cfg).next == ()
    assert textos_ai(grafo, cfg) == [
        "No tengo acceso a tu ubicación. Me falta una referencia en tu mensaje: "
        "dime la calle donde está (p. ej. 'Reforma', '20 de Noviembre' o 'Insurgentes') "
        "o el número de la cicloestación (p. ej. '261')."
    ]


def test_backstop_descarta_frase_con_bicis(montar_grafo):
    """Si el LLM adorna, va el fijo, no su texto."""
    grafo, cfg = montar_grafo(
        router="MOVILIDAD",
        trabaja=[pide_herramienta("ecobici", {"estacion": "cercanas"})],
        frase=[responde("Claro, hay 5 bicis cerca de ti.")],
    )
    grafo.invoke({"messages": [HumanMessage(content="bicis cerca de mí")]}, config=cfg)
    assert textos_ai(grafo, cfg) == [
        "No tengo acceso a tu ubicación. Me falta una referencia en tu mensaje: "
        "dime la calle donde está (p. ej. 'Reforma', '20 de Noviembre' o 'Insurgentes') "
        "o el número de la cicloestación (p. ej. '261')."
    ]


def test_fragmento_corto_no_respalda_aunque_case(montar_grafo):
    """"mi" casa con "de mi" por subcadena: el mínimo de 3 lo bloquea."""
    grafo, cfg = montar_grafo(
        router="MOVILIDAD",
        trabaja=[pide_herramienta("ecobici", {"estacion": "mi"})],
        frase=[responde("No tengo acceso a tu ubicación.")],
    )
    grafo.invoke({"messages": [HumanMessage(content="bicis cerca de mí")]}, config=cfg)
    assert grafo.get_state(cfg).next == ()
    assert textos_ai(grafo, cfg)[0].startswith("No tengo acceso a tu ubicación.")


def test_tartamudeo_de_la_frase_se_colapsa(montar_grafo):
    grafo, cfg = montar_grafo(
        router="MOVILIDAD",
        trabaja=[pide_herramienta("ecobici", {"estacion": "mi"})],
        frase=[responde("No tengo acceso a tu ubicación.No tengo acceso a tu ubicación.")],
    )
    grafo.invoke({"messages": [HumanMessage(content="bicis cerca de mí")]}, config=cfg)
    assert textos_ai(grafo, cfg)[0].startswith(
        "No tengo acceso a tu ubicación. Me falta"
    )


def test_segunda_frase_se_recorta(montar_grafo):
    grafo, cfg = montar_grafo(
        router="MOVILIDAD",
        trabaja=[pide_herramienta("ecobici", {"estacion": "mi"})],
        frase=[responde("No tengo acceso a tu ubicación. Con gusto te ayudo.")],
    )
    grafo.invoke({"messages": [HumanMessage(content="bicis cerca de mí")]}, config=cfg)
    assert textos_ai(grafo, cfg)[0].startswith(
        "No tengo acceso a tu ubicación. Me falta"
    )


def test_tartamudeo_unido_por_coma_se_colapsa(montar_grafo):
    grafo, cfg = montar_grafo(
        router="MOVILIDAD",
        trabaja=[pide_herramienta("ecobici", {"estacion": "mi"})],
        frase=[responde("No tengo acceso a tu ubicación, no tengo acceso a tu ubicación.")],
    )
    grafo.invoke({"messages": [HumanMessage(content="bicis cerca de mí")]}, config=cfg)
    assert textos_ai(grafo, cfg)[0].startswith(
        "No tengo acceso a tu ubicación. Me falta"
    )


def test_tartamudeo_unido_por_y_se_colapsa(montar_grafo):
    grafo, cfg = montar_grafo(
        router="MOVILIDAD",
        trabaja=[pide_herramienta("ecobici", {"estacion": "mi"})],
        frase=[responde("No tengo acceso a tu ubicación y no tengo acceso a tu ubicación")],
    )
    grafo.invoke({"messages": [HumanMessage(content="bicis cerca de mí")]}, config=cfg)
    assert textos_ai(grafo, cfg)[0].startswith(
        "No tengo acceso a tu ubicación. Me falta"
    )
