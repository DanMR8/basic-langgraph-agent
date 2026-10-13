"""Detecta APIs de Flet inexistentes en la versión instalada.

Recorre el AST de `front/app.py` y resuelve cada ruta `ft.…` contra el
módulo real y cada `page.…` contra la clase `Page`. Habría cazado los tres
renombres de la 1.0 (`ft.alignment`, `ft.ElevatedButton`, `page.open`) sin
abrir la ventana.
"""

import ast
from pathlib import Path

import flet as ft


def _rutas(arbol: ast.AST) -> set[tuple[str, tuple[str, ...]]]:
    """Rutas punteadas con raíz `ft` o `page` usadas en el código."""
    rutas = set()
    for n in ast.walk(arbol):
        if not isinstance(n, ast.Attribute):
            continue
        partes = []
        m = n
        while isinstance(m, ast.Attribute):
            partes.append(m.attr)
            m = m.value
        if isinstance(m, ast.Name) and m.id in ("ft", "page"):
            rutas.add((m.id, tuple(reversed(partes))))
    return rutas


def test_apis_flet_existen_en_instalada():
    ruta_app = Path(__file__).parent.parent / "front" / "app.py"
    arbol = ast.parse(ruta_app.read_text(encoding="utf-8"))
    faltantes = []
    for raiz, partes in sorted(_rutas(arbol)):
        if raiz == "ft":
            obj, viva = ft, True
            for p in partes:
                try:
                    obj = getattr(obj, p)
                except AttributeError:
                    viva = False
                    break
            if not viva:
                faltantes.append("ft." + ".".join(partes))
        elif partes[0] == "window":
            # `page.window` es atributo de instancia: se resuelve el resto
            # contra la clase `Window`.
            obj, viva = ft.Window, True
            for p in partes[1:]:
                try:
                    obj = getattr(obj, p)
                except AttributeError:
                    viva = False
                    break
            if not viva:
                faltantes.append("page." + ".".join(partes))
        elif not hasattr(ft.Page, partes[0]):
            faltantes.append("page." + partes[0])
    assert not faltantes, f"APIs inexistentes en flet instalado: {faltantes}"
