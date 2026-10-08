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
        args = ", ".join(f"{k}={v!r}" for k, v in h["args"].items())
        print(f"   • {h['name']}({args})")
    try:
        respuesta = input("   ¿Autorizas? [s/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "rechazar"
    return "aprobar" if respuesta in _SI else "rechazar"


def _emitir(payload) -> None:
    """Avanza el grafo imprimiendo tokens y respondiendo a las aprobaciones.

    El grafo se detiene en `interrupt()`; aquí se detecta el punto de espera,
    se pregunta al usuario y se reanuda con `Command(resume=...)`.
    """
    while True:
        for chunk, meta in agent.stream(payload, config=CONFIG, stream_mode="messages"):
            nodo = meta.get("langgraph_node")
            # Hablan con el usuario: el hablador, el trabajador cuando redacta
            # y los mensajes fijos (rechazo, dato inventado, fallo agotado).
            # `router` y `tools` son turno interno y no narran nada.
            if nodo in ("hablador", "trabajador", "approve", "fallo") and chunk.content:
                print(chunk.content, end="", flush=True)
            elif nodo == "tools" and chunk.content:
                print(f"\n   ⚙️  {chunk.content}")

        snapshot = agent.get_state(CONFIG)
        if not snapshot.next:
            break

        pendientes = [i for task in snapshot.tasks for i in task.interrupts]
        payload = Command(resume=_pedir_aprobacion(pendientes[0].value["herramientas"]))

    print()


def main() -> None:
    print("🤖 Agente local (LangGraph + Ollama)")
    print("   Herramientas: calculadora · clima · búsqueda y listado en archivos")
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
