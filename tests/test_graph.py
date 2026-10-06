"""Tests del grafo: aprobación humana, streaming y límite de recursión.

Usan un LLM falso (ver `conftest.py`), así que la suite corre sin Ollama.
"""

import pytest
from langchain_core.messages import HumanMessage, RemoveMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from agent import nodes
from agent.graph import agent, build_agent
from agent.nodes import route_after_approval, run_tools, should_continue, validar
from agent.state import AgentState
from agent.tools import TOOLS
from config.settings import (
    INSTRUCCION_VETO,
    TEMPERATURA_CONVERSAR,
    TEMPERATURA_RESPONDER,
    TEMPERATURA_VALIDADOR,
)
from tests.conftest import pide_herramienta, responde


def turno_completo(final: str) -> list:
    """Respuestas del LLM: pide la calculadora y luego da la respuesta final."""
    return [pide_herramienta("calculator", {"expression": "2 + 2"}), responde(final)]


class _Veredicto:
    """Doble de `llm_validador`: devuelve el veredicto del test y anota lo visto."""

    def __init__(self, test):
        self._test = test

    def invoke(self, messages):
        self._test.propuestas = [m["content"] for m in messages if m["role"] == "user"]
        return responde(self._test.veredicto)


# ─────────────────────────────────────────────────────────────────────────────
# Estructura
# ─────────────────────────────────────────────────────────────────────────────


def test_nodos_del_grafo():
    assert set(agent.get_graph().nodes) == {
        "__start__",
        "conversar",
        "validar",
        "approve",
        "tools",
        "responder",
        "__end__",
    }


def test_herramientas_expuestas_al_llm():
    assert {t.name for t in TOOLS} == {"calculator", "get_weather", "search_files"}


def test_estado_solo_lleva_mensajes():
    assert set(AgentState.__annotations__) == {"messages"}


def test_temperatura_segun_el_trabajo_de_cada_nodo():
    """El que habla con el usuario varía; el que clasifica SI/NO no."""
    assert (TEMPERATURA_CONVERSAR, TEMPERATURA_RESPONDER) == (0.5, 0.7)
    assert TEMPERATURA_VALIDADOR == 0.0

    assert nodes.llm_con_herramientas.bound.temperature == TEMPERATURA_CONVERSAR
    assert nodes.llm_sin_herramientas.temperature == TEMPERATURA_RESPONDER
    assert nodes.llm_validador.temperature == TEMPERATURA_VALIDADOR


def test_grafo_complilado_tiene_checkpointer():
    assert agent.checkpointer is not None


def test_build_agent_acepta_sin_checkpointer():
    assert build_agent().checkpointer is None


# ─────────────────────────────────────────────────────────────────────────────
# Enrutado
# ─────────────────────────────────────────────────────────────────────────────


class TestEnrutado:
    def test_pide_validacion_si_hay_tool_calls(self):
        estado = {"messages": [pide_herramienta("calculator", {"expression": "1"})]}
        assert should_continue(estado) == "validar"

    def test_termina_si_no_hay_tool_calls(self):
        assert should_continue({"messages": [responde("hola")]}) == "end"

    def test_aprobado_va_a_tools(self):
        estado = {"messages": [pide_herramienta("calculator", {"expression": "1"})]}
        assert route_after_approval(estado) == "tools"

    def test_rechazado_cierra_el_turno(self):
        """El rechazo es terminal: END, no de vuelta al agente."""
        rechazo = ToolMessage(
            content="El usuario rechazó", tool_call_id="call_1", name="calculator"
        )
        assert route_after_approval({"messages": [rechazo]}) == "end"


# ─────────────────────────────────────────────────────────────────────────────
# Validador: la puerta que decide si la petición llega al humano
# ─────────────────────────────────────────────────────────────────────────────


class TestValidar:
    @pytest.fixture(autouse=True)
    def _falso(self, monkeypatch):
        self.veredicto = "SI"
        monkeypatch.setattr(nodes, "llm_validador", _Veredicto(self))

    def _petir(self, pregunta="hola", nombre="search_files", args=None):
        return validar(
            {
                "messages": [
                    HumanMessage(content=pregunta),
                    pide_herramienta(nombre, args or {"query": "hola"}),
                ]
            }
        )

    def test_deja_pasar_y_va_a_aprobacion(self):
        assert self._petir().goto == "approve"

    def test_veta_y_va_al_cierre_sin_herramientas(self):
        self.veredicto = "NO"
        assert self._petir().goto == "responder"

    def test_al_vetar_cierra_la_tool_call_abierta(self):
        """La API de chat exige que toda tool_call tenga su ToolMessage."""
        self.veredicto = "NO"
        cmd = self._petir()
        mensajes = cmd.update["messages"]
        assert len(mensajes) == 1
        assert isinstance(mensajes[0], ToolMessage)
        assert mensajes[0].content == nodes.VETO_CONTENIDO

    def test_al_vetar_no_escribe_ordenes_en_el_estado(self):
        """Lo que persiste es un registro, no una orden."""
        self.veredicto = "NO"
        contenido = self._petir().update["messages"][0].content
        assert INSTRUCCION_VETO not in contenido
        assert not any(
            verbo in contenido for verbo in ("Contesta", "No digas", "dilo")
        )

    def test_al_vetar_no_escribe_nada_en_el_estado(self):
        assert self._petir().update is None

    def test_le_envia_la_propuesta_concreta_al_validador(self):
        self._petir(nombre="calculator", args={"expression": "2+2"}, pregunta="2+2")
        assert self.propuestas == ["Pregunta: 2+2\nPropuesta: calculator({'expression': '2+2'})"]


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

    def test_rechazar_no_ejecuta_y_cierra_el_turno(self, montar_grafo):
        """Tras el rechazo el LLM no narra nada: no se le llega a llamar."""
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
        # Con rechazo terminal el turno termina: no hay respuesta narrativa del
        # LLM tras el rechazo, así que la cola de respuestas sigue intacta.
        assert "rechazad" in aviso.lower()
        assert "herramienta" in aviso.lower() or "NO" in aviso
        # El último mensaje del historial es el ToolMessage de rechazo
        assert isinstance(resultado["messages"][-1], ToolMessage)
        assert len(fake.responses) == 1

    def test_sin_resume_la_herramienta_no_se_ejecuta(self, montar_grafo):
        grafo, cfg, _ = montar_grafo(turno_completo("El resultado es 4"))

        grafo.invoke({"messages": [HumanMessage(content="2+2")]}, config=cfg)
        snapshot = grafo.get_state(cfg)

        assert [m for m in snapshot.values["messages"] if isinstance(m, ToolMessage)] == []


# ─────────────────────────────────────────────────────────────────────────────
# Veto: la petición ni llega a pedir aprobación
# ─────────────────────────────────────────────────────────────────────────────


class TestVetoDeHerramienta:
    def test_no_se_interrumpe_al_usuario_si_el_validador_veta(self, montar_grafo):
        grafo, cfg, _ = montar_grafo(
            [
                pide_herramienta("search_files", {"query": "hola"}),
                responde("Hola, ¿en qué puedo ayudarte?"),
            ],
            veredicto="NO",
        )

        grafo.invoke({"messages": [HumanMessage(content="hola")]}, config=cfg)

        assert grafo.get_state(cfg).next == ()

    def test_se_responde_igualmente_al_usuario(self, montar_grafo):
        grafo, cfg, _ = montar_grafo(
            [
                pide_herramienta("search_files", {"query": "hola"}),
                responde("Hola, ¿en qué puedo ayudarte?"),
            ],
            veredicto="NO",
        )

        resultado = grafo.invoke(
            {"messages": [HumanMessage(content="hola")]}, config=cfg
        )

        assert resultado["messages"][-1].content == "Hola, ¿en qué puedo ayudarte?"
        # El par vetado no sobrevive al turno: solo quedan pregunta y respuesta.
        assert [m for m in resultado["messages"] if isinstance(m, ToolMessage)] == []
        assert not any(getattr(m, "tool_calls", None) for m in resultado["messages"])

    def test_si_el_validador_deja_pasar_se_pide_aprobacion(self, montar_grafo):
        grafo, cfg, _ = montar_grafo(turno_completo("El resultado es 4"))

        grafo.invoke({"messages": [HumanMessage(content="2+2")]}, config=cfg)

        assert grafo.get_state(cfg).next == ("approve",)


# ─────────────────────────────────────────────────────────────────────────────
# Veto sin huella: la orden vive en el turno, el historial queda de chat
# ─────────────────────────────────────────────────────────────────────────────


class TestInstruccionTransitoria:
    def test_responder_recibe_la_orden_si_el_turno_cierra_con_veto(self, montar_grafo):
        _, _, fake = montar_grafo([responde("No tengo esa información.")])

        nodes.responder(
            {
                "messages": [
                    HumanMessage(content="¿tienes nombre?"),
                    pide_herramienta("search_files", {"query": "nombre"}),
                    ToolMessage(
                        content=nodes.VETO_CONTENIDO,
                        tool_call_id="call_1",
                        name="search_files",
                    ),
                ]
            }
        )

        assert INSTRUCCION_VETO in fake.prompts[0][0]["content"]

    def test_responder_no_recibe_la_orden_si_se_ejecuto_algo(self, montar_grafo):
        _, _, fake = montar_grafo([responde("El resultado es 4")])

        nodes.responder(
            {
                "messages": [
                    HumanMessage(content="2+2"),
                    ToolMessage(
                        content="Resultado: 4",
                        tool_call_id="call_1",
                        name="calculator",
                    ),
                ]
            }
        )

        assert INSTRUCCION_VETO not in fake.prompts[0][0]["content"]

    def test_responder_borra_el_par_de_veto(self, montar_grafo):
        """Borra la tool_call y su ToolMessage: el turno cierra de chat puro."""
        _, _, _ = montar_grafo([responde("No tengo esa información.")])
        peticion = pide_herramienta("search_files", {"query": "nombre"})
        peticion.id = "ai_1"
        veto = ToolMessage(
            content=nodes.VETO_CONTENIDO,
            tool_call_id="call_1",
            name="search_files",
            id="tool_1",
        )

        salida = nodes.responder(
            {"messages": [HumanMessage(content="¿tienes nombre?"), peticion, veto]}
        )

        borrados = [m.id for m in salida["messages"] if isinstance(m, RemoveMessage)]
        assert borrados == ["ai_1", "tool_1"]

    def test_responder_no_borra_nada_si_se_ejecuto_algo(self, montar_grafo):
        """El resultado de una herramienta aprobada sí se conserva."""
        _, _, _ = montar_grafo([responde("El resultado es 4")])

        salida = nodes.responder(
            {
                "messages": [
                    HumanMessage(content="2+2"),
                    ToolMessage(
                        content="Resultado: 4",
                        tool_call_id="call_1",
                        name="calculator",
                        id="tool_1",
                    ),
                ]
            }
        )

        assert not any(isinstance(m, RemoveMessage) for m in salida["messages"])

    def test_sin_ids_no_borra_nada(self, montar_grafo):
        """Si no puede identificar el par, deja el turno intacto."""
        _, _, _ = montar_grafo([responde("No tengo esa información.")])

        salida = nodes.responder(
            {
                "messages": [
                    HumanMessage(content="¿tienes nombre?"),
                    pide_herramienta("search_files", {"query": "nombre"}),
                    ToolMessage(
                        content=nodes.VETO_CONTENIDO,
                        tool_call_id="call_1",
                        name="search_files",
                    ),
                ]
            }
        )

        assert not any(isinstance(m, RemoveMessage) for m in salida["messages"])

    def test_la_orden_no_queda_nunca_en_el_historial(self, montar_grafo):
        """Regresión de la fuga: ni la orden ni el par vetado persisten.

        Escritos en un mensaje, el modelo los devolvía como si fueran suyos
        tres turnos después.
        """
        grafo, cfg, _ = montar_grafo(
            [
                pide_herramienta("search_files", {"query": "nombre"}),
                responde("Soy un asistente conversacional."),
            ],
            veredicto="NO",
        )

        grafo.invoke({"messages": [HumanMessage(content="¿tienes nombre?")]}, config=cfg)

        mensajes = grafo.get_state(cfg).values["messages"]
        assert not any(INSTRUCCION_VETO in str(m.content) for m in mensajes)
        assert not any("Contesta tú en texto" in str(m.content) for m in mensajes)
        assert [m for m in mensajes if isinstance(m, ToolMessage)] == []
        assert not any(getattr(m, "tool_calls", None) for m in mensajes)

    def test_el_siguiente_turno_no_arrarra_la_orden(self, montar_grafo):
        """`conversar` del turno 2 no ve nada del veto del turno 1."""
        grafo, cfg, fake = montar_grafo(
            [
                pide_herramienta("search_files", {"query": "nombre"}),
                responde("Soy un asistente conversacional."),
                responde("Hola de nuevo."),
            ],
            veredicto="NO",
        )

        grafo.invoke({"messages": [HumanMessage(content="¿tienes nombre?")]}, config=cfg)
        grafo.invoke({"messages": [HumanMessage(content="hola")]}, config=cfg)

        # prompts[0] = conversar t1, [1] = responder t1, [2] = conversar t2
        assert INSTRUCCION_VETO not in fake.prompts[2][0]["content"]


# ─────────────────────────────────────────────────────────────────────────────
# Límite de recursión y ejecución de herramientas
# ─────────────────────────────────────────────────────────────────────────────


class TestSeguridadAnteBucles:
    def test_llm_insistente_nunca_reabre_el_ciclo(self, montar_grafo):
        """Una aprobación, una ejecución, y el turno se cierra.

        El LLM insiste en pedir herramientas: aquí devuelve 50 peticiones
        seguidas. Pero `responder` es el último nodo y va directo a END, así
        que esas 49 restantes no las mira nadie. Lo que corta el bucle es la
        estructura del grafo, no el recursion_limit ni la paciencia del humano.
        """
        cola = [pide_herramienta("calculator", {"expression": "1 + 1"})]
        cola += [pide_herramienta("calculator", {"expression": str(i)}) for i in range(50)]
        cola += [responde("Listo")]
        grafo, cfg, fake = montar_grafo(cola)

        grafo.invoke({"messages": [HumanMessage(content="empieza")]}, config=cfg)
        resultado = grafo.invoke(Command(resume="aprobar"), config=cfg)

        assert grafo.get_state(cfg).next == ()
        ejecutados = [
            m for m in resultado["messages"] if isinstance(m, ToolMessage)
        ]
        assert len(ejecutados) == 1
        # Solo conversar y responder consumieron la cola; el resto sigue intacto.
        assert len(fake.responses) == len(cola) - 2

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
