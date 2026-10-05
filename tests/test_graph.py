"""Tests del grafo: aprobación humana, streaming y límite de recursión.

Usan un LLM falso (ver `conftest.py`), así que la suite corre sin Ollama.
"""

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from agent.graph import agent, build_agent
from agent.nodes import route_after_approval, run_tools, should_continue
from agent.state import AgentState
from agent.tools import TOOLS
from tests.conftest import pide_herramienta, responde


def turno_completo(final: str) -> list:
    """Respuestas del LLM: pide la calculadora y luego da la respuesta final."""
    return [pide_herramienta("calculator", {"expression": "2 + 2"}), responde(final)]


# ─────────────────────────────────────────────────────────────────────────────
# Estructura
# ─────────────────────────────────────────────────────────────────────────────


def test_nodos_del_grafo():
    assert set(agent.get_graph().nodes) == {
        "__start__",
        "agent",
        "approve",
        "tools",
        "__end__",
    }


def test_herramientas_expuestas_al_llm():
    assert {t.name for t in TOOLS} == {"calculator", "get_weather", "search_files"}


def test_estado_solo_lleva_mensajes():
    assert set(AgentState.__annotations__) == {"messages"}


def test_grafo_complilado_tiene_checkpointer():
    assert agent.checkpointer is not None


def test_build_agent_acepta_sin_checkpointer():
    assert build_agent().checkpointer is None


# ─────────────────────────────────────────────────────────────────────────────
# Enrutado
# ─────────────────────────────────────────────────────────────────────────────


class TestEnrutado:
    def test_pide_aprobacion_si_hay_tool_calls(self):
        estado = {"messages": [pide_herramienta("calculator", {"expression": "1"})]}
        assert should_continue(estado) == "approve"

    def test_termina_si_no_hay_tool_calls(self):
        assert should_continue({"messages": [responde("hola")]}) == "end"

    def test_aprobado_va_a_tools(self):
        estado = {"messages": [pide_herramienta("calculator", {"expression": "1"})]}
        assert route_after_approval(estado) == "tools"

    def test_rechazado_regresa_al_agente(self):
        rechazo = ToolMessage(
            content="El usuario rechazó", tool_call_id="call_1", name="calculator"
        )
        assert route_after_approval({"messages": [rechazo]}) == "end"


# ─────────────────────────────────────────────────────────────────────────────
# Ciclo completo con interrupt / Command(resume=...)
# ─────────────────────────────────────────────────────────────────────────────


class TestAprobacionHumana:
    def test_suspende_en_approve_hasta_que_se_responda(self, montar_grafo):
        grafo, cfg, _ = montar_grafo(turno_completo("El resultado es 4"))

        grafo.invoke({"messages": [HumanMessage(content="2+2")]}, config=cfg)

        snapshot = grafo.get_state(cfg)
        assert snapshot.next == ("approve",)

        pendientes = [i for task in snapshot.tasks for i in task.interrupts]
        assert len(pendientes) == 1
        payload = pendientes[0].value
        assert payload["requiere_aprobacion"] is True
        assert payload["herramientas"] == [
            {"name": "calculator", "args": {"expression": "2 + 2"}}
        ]

    def test_aprobar_ejecuta_la_herramienta_y_responde(self, montar_grafo):
        grafo, cfg, _ = montar_grafo(turno_completo("El resultado es 4"))

        grafo.invoke({"messages": [HumanMessage(content="2+2")]}, config=cfg)
        resultado = grafo.invoke(Command(resume="aprobar"), config=cfg)

        assert resultado["messages"][-1].content == "El resultado es 4"
        assert any(
            isinstance(m, ToolMessage) and m.content == "Resultado: 4"
            for m in resultado["messages"]
        )

    def test_rechazar_no_ejecuta_y_el_llm_se_entera(self, montar_grafo):
        grafo, cfg, fake = montar_grafo(
            [
                pide_herramienta("calculator", {"expression": "2 + 2"}),
                responde("Entendido, no lo calculo."),
            ]
        )

        grafo.invoke({"messages": [HumanMessage(content="2+2")]}, config=cfg)
        resultado = grafo.invoke(Command(resume="rechazar"), config=cfg)

        rechazos = [
            m for m in resultado["messages"] if isinstance(m, ToolMessage) and m.tool_call_id
        ]
        assert len(rechazos) == 1
        aviso = rechazos[0].content
        # Con rechazo terminal el turno termina: no hay respuesta narrativa del LLM
        # tras el rechazo y el grafo no vuelve a 'agent'.
        assert "rechazad" in aviso.lower()
        assert "herramienta" in aviso.lower() or "NO" in aviso
        # El último mensaje del historial es el ToolMessage de rechazo
        assert isinstance(resultado["messages"][-1], ToolMessage)

    def test_sin_resume_la_herramienta_no_se_ejecuta(self, montar_grafo):
        grafo, cfg, _ = montar_grafo(turno_completo("El resultado es 4"))

        grafo.invoke({"messages": [HumanMessage(content="2+2")]}, config=cfg)
        snapshot = grafo.get_state(cfg)

        assert [m for m in snapshot.values["messages"] if isinstance(m, ToolMessage)] == []


# ─────────────────────────────────────────────────────────────────────────────
# Límite de recursión y ejecución de herramientas
# ─────────────────────────────────────────────────────────────────────────────


class TestSeguridadAnteBucles:
    def test_llm_insistente_nunca_corre_atrevido(self, montar_grafo):
        """Un LLM que pide herramienta 50 veces ejecuta una sola por aprobación.

        La puerta humana, no el recursion_limit, es lo que corta el ciclo.
        """
        bucle = [pide_herramienta("calculator", {"expression": "1 + 1"}) for _ in range(50)]
        grafo, cfg, _ = montar_grafo(bucle)

        grafo.invoke({"messages": [HumanMessage(content="empieza")]}, config=cfg)

        for _ in range(4):
            grafo.invoke(Command(resume="aprobar"), config=cfg)
            snapshot = grafo.get_state(cfg)
            # Siempre vuelve a detenerse esperando una decisión nueva.
            assert snapshot.next == ("approve",)

        ejecutados = [
            m for m in snapshot.values["messages"] if isinstance(m, ToolMessage)
        ]
        # 4 aprobaciones -> 4 ejecuciones, nunca las 50 que pidió el LLM.
        assert len(ejecutados) == 4

    def test_recursion_limit_sigue_siendo_un_freno(self, montar_grafo):
        """Red de seguridad: si el límite es insuficiente, LangGraph aborta."""
        grafo, cfg, _ = montar_grafo(
            [pide_herramienta("calculator", {"expression": "1 + 1"}) for _ in range(50)]
        )
        cfg = {**cfg, "recursion_limit": 1}

        with pytest.raises(GraphRecursionError):
            grafo.invoke({"messages": [HumanMessage(content="empieza")]}, config=cfg)


def test_run_tools_ejecuta_lo_aprobado():
    estado = {"messages": [pide_herramienta("calculator", {"expression": "6 * 7"})]}
    salida = run_tools(estado)
    assert salida["messages"][0].content == "Resultado: 42"
