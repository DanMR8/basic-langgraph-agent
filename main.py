"""CLI interactiva para chatear con el agente local."""

import uuid

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from agent.graph import agent
from config.settings import RECURSION_LIMIT

# Un hilo por sesión: el checkpointer guarda aquí el historial entre turnos.
CONFIG = {
    "configurable": {"thread_id": uuid.uuid4().hex},
    "recursion_limit": RECURSION_LIMIT,
}

_SI = {"s", "si", "sí", "y", "yes"}


def _pedir_aprobacion(herramientas: list[dict]) -> str:
    """Pide confirmación antes de que el agente ejecute algo."""
    print("\n⚠️  El agente quiere ejecutar:")
    for h in herramientas:
        # `conclusion` es indicación de formato para la herramienta, no dato
        # de la petición: no se muestra.
        args = ", ".join(
            f"{k}={v!r}" for k, v in h["args"].items() if k != "conclusion"
        )
        print(f"   • {h['name']}({args})")
    try:
        respuesta = input("   ¿Autorizas? [s/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "rechazar"
    return "aprobar" if respuesta in _SI else "rechazar"


def _pedir_si_no(pregunta: str) -> str:
    """Pregunta fija del grafo (p. ej. ofrecer el segundo eslabón)."""
    print(f"\n❓  {pregunta}")
    try:
        respuesta = input("   [s/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "no"
    return "si" if respuesta in _SI else "no"


def _emitir(payload) -> None:
    """Avanza el grafo imprimiendo tokens y respondiendo a las aprobaciones.

    El grafo se detiene en `interrupt()`; aquí se detecta el punto de espera,
    se pregunta al usuario y se reanuda con `Command(resume=...)`.

    Los nodos LLM (`hablador`, `trabajador`) emiten deltas y se imprimen al
    vuelo. Los nodos con mensaje completo (`approve`, `fallo`, `tools`) a
    veces llegan en chunks acumulativos: se guarda el último por nodo y se
    vuelca al cambiar de nodo, para no imprimir dos veces lo mismo.
    """
    while True:
        fijos: dict[str, str] = {}
        nodo_actual: str | None = None
        for chunk, meta in agent.stream(payload, config=CONFIG, stream_mode="messages"):
            nodo = meta.get("langgraph_node")
            if nodo != nodo_actual:
                _volcar_fijos(fijos)
                nodo_actual = nodo
            if nodo in ("hablador", "trabajador") and chunk.content:
                print(chunk.content, end="", flush=True)
            elif nodo in ("approve", "fallo", "resumir", "tools") and chunk.content:
                fijos[nodo] = chunk.content
        _volcar_fijos(fijos)

        snapshot = agent.get_state(CONFIG)
        if not snapshot.next:
            break

        pendientes = [i for task in snapshot.tasks for i in task.interrupts]
        valor = pendientes[0].value
        if "herramientas" in valor:
            payload = Command(resume=_pedir_aprobacion(valor["herramientas"]))
        else:
            payload = Command(resume=_pedir_si_no(valor["pregunta"]))

    print()


def _volcar_fijos(fijos: dict[str, str]) -> None:
    """Imprime los mensajes completos acumulados, cada uno una sola vez."""
    for nodo, texto in fijos.items():
        if nodo == "tools":
            print(f"\n   ⚙️  {texto}")
        else:
            print(texto, end="", flush=True)
    fijos.clear()


def main() -> None:
    print("🤖 Agente local (LangGraph + Ollama)")
    print("   Herramientas: calculadora · clima · ecobici")
    print("   Las herramientas requieren tu aprobación antes de ejecutarse.")
    print("   Escribe 'salir' para terminar.\n")

    while True:
        try:
            user_input = input("Tú › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Adiós.")
            break

        if user_input.lower() in {"salir", "exit", "quit"}:
            print("👋 Adiós.")
            break

        if not user_input:
            continue

        print("🤖 › ", end="", flush=True)
        try:
            _emitir({"messages": [HumanMessage(content=user_input)]})
        except GraphRecursionError:
            print(
                f"\n⚠️  El agente superó el límite de {RECURSION_LIMIT} pasos "
                "y se detuvo para evitar un bucle."
            )
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
