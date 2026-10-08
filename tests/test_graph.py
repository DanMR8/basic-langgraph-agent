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
from agent.nodes import _es_error, route_router
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
        "search_files",
        "list_files",
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
    [("TEXTO", "texto"), ("CALCULO", "herramienta"), ("CLIMA", "herramienta")],
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
        "Me falta la ciudad en tu mensaje: dímelo y lo consulto."
    ]


def test_ciudad_respaldada_si_pide_permiso(montar_grafo):
    grafo, cfg = montar_grafo(
        router="CLIMA",
        trabaja=[pide_herramienta("get_weather", {"city": "Madrid"})],
    )
    grafo.invoke({"messages": [HumanMessage(content="clima en Madrid?")]}, config=cfg)
    assert grafo.get_state(cfg).next == ("approve",)


def test_query_normalizada_con_respaldo(montar_grafo):
    """"lang graph" respalda a "langgraph": comparar sin espacios."""
    grafo, cfg = montar_grafo(
        router="ARCHIVOS",
        trabaja=[
            pide_herramienta("search_files", {"query": "langgraph"}),
            responde("Listo."),
        ],
    )
    grafo.invoke(
        {"messages": [HumanMessage(content="resumen de lang graph")]}, config=cfg
    )
    assert grafo.get_state(cfg).next == ("approve",)


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


def test_sin_match_no_se_reintenta(montar_grafo):
    """"No se encontró" no es un error: la búsqueda funcionó, se redacta."""
    grafo, cfg = montar_grafo(
        router="ARCHIVOS",
        trabaja=[
            pide_herramienta("search_files", {"query": "zzz"}),
            responde("No encontré nada sobre zzz."),
        ],
    )
    grafo.invoke({"messages": [HumanMessage(content="busca zzz")]}, config=cfg)
    grafo.invoke(Command(resume="aprobar"), config=cfg)

    assert textos_ai(grafo, cfg) == ["No encontré nada sobre zzz."]
    assert grafo.get_state(cfg).values["reintentos"] == 0


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
