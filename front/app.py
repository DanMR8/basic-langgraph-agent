"""Front en Flet. Uso: python -X utf8 -m front.app (con Ollama corriendo)."""

import contextlib
import datetime
import re
import threading
import uuid

import flet as ft
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent.graph import agent
from config.settings import RECURSION_LIMIT

CONFIG = {
    "configurable": {"thread_id": uuid.uuid4().hex},
    "recursion_limit": RECURSION_LIMIT,
}

# Paleta del proyecto + dos familias (display y texto).
FONDO = "#11162B"
CREMA = "#E6D5B7"
NARANJA = "#F54F1B"
PISTA = "#991E223D"
FUENTE_TITULO = "Sora"
FUENTE_TEXTO = "Inter"


def main(page: ft.Page) -> None:
    page.title = "Agente local (LangGraph + Ollama)"
    page.bgcolor = FONDO
    page.fonts = {
        FUENTE_TITULO: "https://github.com/google/fonts/raw/main/ofl/sora/Sora%5Bwght%5D.ttf",
        FUENTE_TEXTO: "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf",
    }
    page.theme = ft.Theme(font_family=FUENTE_TEXTO, color_scheme_seed=NARANJA)

    ubi = {"on": False}

    def cristal(contenido: ft.Control, radio: int = 16) -> ft.Container:
        """Superficie glass: translúcida + blur + borde sutil."""
        return ft.Container(
            content=contenido,
            bgcolor=ft.Colors.with_opacity(0.08, ft.Colors.WHITE),
            blur=ft.Blur(18, 18),
            border=ft.Border.all(1, ft.Colors.with_opacity(0.14, ft.Colors.WHITE)),
            border_radius=radio,
            padding=10,
        )

    def fondo_ambiental() -> ft.Stack:
        brillo = lambda color, x, y: ft.Container(  # noqa: E731 - factory local
            width=420,
            height=420,
            gradient=ft.RadialGradient(
                colors=[
                    ft.Colors.with_opacity(0.28, color),
                    ft.Colors.with_opacity(0.0, color),
                ]
            ),
            left=x,
            top=y,
        )
        return ft.Stack(
            [
                ft.Container(
                    expand=True,
                    margin=18,
                    border_radius=28,
                    gradient=ft.LinearGradient(
                        begin=ft.Alignment.TOP_LEFT,
                        end=ft.Alignment.BOTTOM_RIGHT,
                        colors=[FONDO, "#1A2340"],
                    ),
                ),
                brillo(NARANJA, 60, 60),
                brillo(CREMA, 420, 560),
            ],
            expand=True,
        )

    chat = ft.ListView(expand=True, spacing=8, auto_scroll=True)
    entrada = ft.TextField(
        hint_text="Escribe…",
        hint_style=ft.TextStyle(color=PISTA),
        expand=True,
        autofocus=True,
        filled=True,
        fill_color=CREMA,
        color=FONDO,
        border=ft.OutlineInputBorder(border_radius=12),
    )
    enviar = ft.IconButton(icon=ft.Icons.SEND, icon_color=FONDO, bgcolor=NARANJA)
    ocupado = ft.ProgressRing(visible=False, width=20, height=20, color=NARANJA)
    ubi_btn = ft.IconButton(
        icon=ft.Icons.LOCATION_OFF,
        icon_color=CREMA,
        tooltip="Ubicación precisa (apagada)",
    )

    def fijar_ubi(encendida: bool) -> None:
        """Una sola fuente de verdad para el toggle, manual o por diálogo."""
        ubi["on"] = encendida
        ubi_btn.icon = ft.Icons.LOCATION_ON if encendida else ft.Icons.LOCATION_OFF
        ubi_btn.icon_color = NARANJA if encendida else CREMA
        ubi_btn.tooltip = (
            "Ubicación precisa (encendida)"
            if encendida
            else "Ubicación precisa (apagada)"
        )
        page.update()

    def al_ubi(_) -> None:
        fijar_ubi(not ubi["on"])

    ubi_btn.on_click = al_ubi

    def burbuja(texto: str, mia: bool) -> ft.Container:
        if mia:
            return ft.Container(
                content=ft.Markdown(
                    texto,
                    selectable=True,
                    extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                    md_style_sheet=ft.MarkdownStyleSheet(
                        p_text_style=ft.TextStyle(
                            color=FONDO, weight=ft.FontWeight.BOLD
                        ),
                    ),
                ),
                bgcolor=NARANJA,
                border_radius=12,
                padding=10,
                expand=8,
            )
        return ft.Container(
            content=ft.Markdown(
                texto,
                selectable=True,
                extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                md_style_sheet=ft.MarkdownStyleSheet(
                    p_text_style=ft.TextStyle(
                        color=CREMA, weight=ft.FontWeight.BOLD
                    ),
                ),
            ),
            bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.WHITE),
            blur=ft.Blur(14, 14),
            border=ft.Border.all(1, ft.Colors.with_opacity(0.12, ft.Colors.WHITE)),
            border_radius=12,
            padding=10,
            expand=8,
        )

    def hueco() -> ft.Container:
        return ft.Container(expand=2)

    def fila_mensaje(texto: str, mia: bool) -> ft.Row:
        globo = burbuja(texto, mia)
        return ft.Row(
            [avatar(mia), globo, hueco()] if not mia else [hueco(), globo, avatar(mia)],
            vertical_alignment=ft.CrossAxisAlignment.START,
        )

    def avatar(mio: bool) -> ft.CircleAvatar:
        return ft.CircleAvatar(
            content=ft.Icon(
                ft.Icons.PERSON if mio else ft.Icons.SMART_TOY,
                color=FONDO if mio else CREMA,
                size=20,
            ),
            bgcolor=NARANJA if mio else ft.Colors.with_opacity(0.16, ft.Colors.WHITE),
            radius=17,
        )

    def linea_herramienta(texto: str) -> ft.Container:
        # La conclusión determinista es para el trabajador (va en el
        # historial), no para el usuario: se filtra solo en pantalla.
        visible = re.sub(
            r"(?m)^- Conclusión: .*$\n?", "", texto
        ).strip()
        return ft.Container(
            content=ft.Markdown(
                f"⚙️  {visible}",
                selectable=True,
                extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                fit_content=True,
                md_style_sheet=ft.MarkdownStyleSheet(
                    p_text_style=ft.TextStyle(size=12, color=CREMA),
                    table_head_text_style=ft.TextStyle(
                        size=12, color=CREMA, weight=ft.FontWeight.BOLD
                    ),
                    table_body_text_style=ft.TextStyle(size=12, color=CREMA),
                    table_cells_padding=ft.Padding.all(4),
                    table_padding=ft.Padding.symmetric(vertical=4),
                ),
            ),
            bgcolor=ft.Colors.with_opacity(0.06, ft.Colors.WHITE),
            blur=ft.Blur(14, 14),
            border=ft.Border.all(1, ft.Colors.with_opacity(0.1, ft.Colors.WHITE)),
            border_radius=12,
            padding=8,
        )

    def mostrar_nuevos(desde: int) -> int:
        mensajes = agent.get_state(CONFIG).values.get("messages", [])
        for m in mensajes[desde:]:
            # El mensaje humano ya se pintó al enviar (respuesta inmediata);
            # aquí solo entra lo que genera el turno.
            if m.type == "human":
                continue
            if m.type == "ai" and not getattr(m, "tool_calls", None):
                if str(m.content or "").strip():
                    chat.controls.append(fila_mensaje(str(m.content), mia=False))
            # El eco de ubicación ya es la respuesta: no duplicarlo aquí.
            elif m.type == "tool" and not str(m.content or "").strip().startswith(
                "Estás en"
            ):
                chat.controls.append(linea_herramienta(str(m.content)))
        page.update()
        return len(mensajes)

    def preguntar(titulo: str, lineas: list[str], aprueba: bool) -> str:
        """Diálogo modal; devuelve "si"/"no" para `Command(resume=...)`."""
        evento, caja = threading.Event(), {}

        def responde(valor: str):
            def _cerrar(_):
                caja["r"] = valor
                dlg.open = False
                page.update()
                evento.set()

            return _cerrar

        if aprueba:
            acciones = [
                ft.FilledButton(
                    "Aprobar",
                    on_click=responde("si"),
                    bgcolor=NARANJA,
                    color=FONDO,
                ),
                ft.TextButton(
                    "Rechazar",
                    on_click=responde("no"),
                    style=ft.ButtonStyle(color=CREMA),
                ),
            ]
        else:
            acciones = [
                ft.FilledButton(
                    "Sí", on_click=responde("si"), bgcolor=NARANJA, color=FONDO
                ),
                ft.TextButton("No", on_click=responde("no")),            ]
        dlg = ft.AlertDialog(
            modal=True,
            bgcolor=ft.Colors.with_opacity(0.92, FONDO),
            title=ft.Text(titulo, font_family=FUENTE_TITULO, color=CREMA),
            content=ft.Column(
                [
                    ft.Text(linea, selectable=True, color=CREMA)
                    for linea in lineas
                ],
                tight=True,
            ),
            actions=acciones,
        )
        page.show_dialog(dlg)
        page.update()
        with contextlib.suppress(Exception):
            page.run_task(page.window.to_front)
        evento.wait()
        return caja.get("r", "no")

    def pedir_aprobacion(herramientas: list[dict]) -> str:
        def formato(h: dict) -> str:
            args = ", ".join(
                f"{k}={v!r}" for k, v in h["args"].items() if k != "conclusion"
            )
            return f"• {h['name']}({args})"

        r = preguntar(
            "El agente quiere ejecutar", [formato(h) for h in herramientas], True
        )
        return "aprobar" if r == "si" else "rechazar"

    def pedir_si_no(pregunta: str) -> str:
        return preguntar("Pregunta", [pregunta], False)

    def _mapea_error_ubi(err: Exception) -> dict:
        """Denegado/apagado/timeout según el error (mejor esfuerzo)."""
        texto = f"{type(err).__name__}: {err}".lower()
        if "denied" in texto or "permission" in texto or "denegad" in texto:
            return {"estado": "denegado"}
        if "disabled" in texto or "service" in texto or "off" in texto:
            return {"estado": "apagado"}
        return {"estado": "timeout"}

    def _gps_directo() -> dict:
        """Una fix de Windows; mapea todo a estados. Sin diálogo aquí."""
        caja, evento = {}, threading.Event()

        async def _localizar():
            try:
                from winsdk.windows.devices.geolocation import (
                    GeolocationAccessStatus as Acceso,
                )
                from winsdk.windows.devices.geolocation import (
                    Geolocator as WinGeo,
                )

                acceso = await WinGeo.request_access_async()
                if acceso != Acceso.ALLOWED:
                    # PositionStatus: 3=Disabled, 5=NotAvailable -> apagado;
                    # lo demás (denegado por el usuario) -> denegado.
                    try:
                        estado_os = int(WinGeo().location_status)
                    except Exception:  # noqa: BLE001 - sin estado, se asume denegado
                        estado_os = -1
                    caja["r"] = {
                        "estado": "apagado" if estado_os in (3, 5) else "denegado"
                    }
                    return
                pos = await WinGeo().get_geoposition_async(
                    datetime.timedelta(minutes=5),
                    datetime.timedelta(seconds=10),
                )
                if pos is None or pos.coordinate is None:
                    caja["r"] = {"estado": "timeout"}
                    return
                caja["r"] = {
                    "estado": "ok",
                    "lat": pos.coordinate.latitude,
                    "lon": pos.coordinate.longitude,
                }
            except ImportError:
                caja["r"] = {"estado": "no_disponible"}
            except Exception as e:  # noqa: BLE001 - se mapea a estado
                caja["r"] = _mapea_error_ubi(e)
            finally:
                evento.set()

        page.run_task(_localizar)
        if not evento.wait(timeout=25):
            return {"estado": "timeout"}
        return caja.get("r", {"estado": "timeout"})

    def pedir_ubicacion() -> dict:
        """Resuelve el interrupt de ubicación: toggle, diálogo o GPS de Windows.

        Sin control Flet de por medio (daba "Unknown control" en dev): se usa
        la API de ubicación de Windows directo desde Python. Sin GPS solo
        resuelve por WiFi si el servicio está encendido; si no, cae al fijo.
        """
        if not ubi["on"] and preguntar(
            "Ubicación precisa",
            ["El agente quiere tu ubicación para buscar estaciones cercanas."],
            False,
        ) != "si":
            return {"estado": "denegado"}
        # El "sí" (aquí o el toggle ya encendido) es consentimiento: el botón
        # lo refleja para no mentir sobre el estado.
        fijar_ubi(True)
        return _gps_directo()

    def pedir_respuesta_ubicacion() -> dict:
        """Diálogo de 3 vías para "dónde estoy": precisa, IP o rechazo.

        Con toggle encendido va directo a precisa (para eso es el toggle).
        Si el GPS falla tras elegir precisa, cae a IP: la salida siempre dice
        la fuente, así que no hay engaño.
        """
        if ubi["on"]:
            eleccion = "precisa"
        else:
            eleccion = _dialogo_tres(
                "Tu ubicación",
                ["¿Cómo respondo dónde estás?"],
                [("Precisa (GPS)", "precisa"), ("Aproximada (IP)", "ip")],
            )
            if eleccion is None:
                return {"estado": "rechazado"}
        if eleccion == "precisa":
            gps = _gps_directo()
            if gps.get("estado") == "ok":
                return {
                    "estado": "precisa",
                    "lat": gps["lat"],
                    "lon": gps["lon"],
                }
            return {"estado": "ip"}
        return {"estado": "ip"}

    def _dialogo_tres(
        titulo: str, lineas: list[str], opciones: list[tuple[str, str]]
    ) -> str | None:
        """Como `preguntar` pero con N botones; None si cierra sin elegir."""
        evento, caja = threading.Event(), {}

        def responde(valor: str | None):
            def _cerrar(_):
                caja["r"] = valor
                dlg.open = False
                page.update()
                evento.set()

            return _cerrar

        dlg = ft.AlertDialog(
            modal=True,
            bgcolor=ft.Colors.with_opacity(0.92, FONDO),
            title=ft.Text(titulo, font_family=FUENTE_TITULO, color=CREMA),
            content=ft.Column(
                [ft.Text(linea, selectable=True, color=CREMA) for linea in lineas],
                tight=True,
            ),
            actions=[
                ft.FilledButton(
                    texto, on_click=responde(valor), bgcolor=NARANJA, color=FONDO
                )
                for texto, valor in opciones
            ]
            + [ft.TextButton("Rechazar", on_click=responde(None))],
        )
        page.show_dialog(dlg)
        page.update()
        with contextlib.suppress(Exception):
            page.run_task(page.window.to_front)
        evento.wait()
        return caja.get("r")

    def turno(texto: str) -> None:
        try:
            vistos = len(agent.get_state(CONFIG).values.get("messages", []))
            payload: dict = {"messages": [HumanMessage(content=texto)]}
            while True:
                agent.invoke(payload, config=CONFIG)
                vistos = mostrar_nuevos(vistos)
                snap = agent.get_state(CONFIG)
                if not snap.next:
                    break
                valor = [i for t in snap.tasks for i in t.interrupts][0].value
                if "herramientas" in valor:
                    payload = Command(resume=pedir_aprobacion(valor["herramientas"]))
                elif valor.get("tipo") == "ubicacion":
                    payload = Command(resume=pedir_ubicacion())
                elif valor.get("tipo") == "ubicacion_respuesta":
                    payload = Command(resume=pedir_respuesta_ubicacion())
                else:
                    payload = Command(resume=pedir_si_no(valor["pregunta"]))
        except Exception as e:  # noqa: BLE001 - el error se muestra en el chat
            chat.controls.append(ft.Text(f"❌ Error: {e}", color=NARANJA))
        finally:
            enviar.disabled = False
            ocupado.visible = False
            page.update()
            page.run_task(entrada.focus)

    def al_enviar(_) -> None:
        texto = (entrada.value or "").strip()
        if not texto or enviar.disabled:
            return
        entrada.value = ""
        enviar.disabled = True
        ocupado.visible = True
        chat.controls.append(fila_mensaje(texto, mia=True))
        page.run_task(entrada.focus)
        page.update()
        page.run_thread(lambda: turno(texto))

    enviar.on_click = al_enviar
    entrada.on_submit = al_enviar
    page.add(
        ft.Stack(
            [
                fondo_ambiental(),
                # Superficie principal: panel esmerilado con bordes
                # redondeados y offset respecto a los bordes de la ventana.
                ft.Container(
                    content=ft.Column(
                        [
                            cristal(
                                ft.Text(
                                    "Asistente del día: Ecobici, clima y algo más…",
                                    font_family=FUENTE_TITULO,
                                    weight=ft.FontWeight.BOLD,
                                    color=NARANJA,
                                )
                            ),
                            ft.Container(
                            content=chat,
                            expand=True,
                            padding=ft.Padding.symmetric(horizontal=4),
                        ),
                            ft.Row([entrada, ubi_btn, enviar, ocupado]),
                        ],
                        expand=True,
                    ),
                    expand=True,
                    margin=18,
                    padding=14,
                    bgcolor=ft.Colors.with_opacity(0.06, ft.Colors.WHITE),
                    blur=ft.Blur(22, 22),
                    border=ft.Border.all(
                        1, ft.Colors.with_opacity(0.16, ft.Colors.WHITE)
                    ),
                    border_radius=28,
                ),
            ],
            expand=True,
        )
    )


if __name__ == "__main__":
    ft.run(main)
