"""Tests de las herramientas del agente.

Ninguno requiere un LLM ni conexión a la red, salvo donde se mockea
explícitamente.
"""

import pytest

from agent.tools import calculator, search_files

# ─────────────────────────────────────────────────────────────────────────────
# Calculadora
# ─────────────────────────────────────────────────────────────────────────────


class TestCalculator:
    def test_operacion_basica(self):
        assert calculator.invoke({"expression": "2 + 2"}) == "Resultado: 4"

    def test_precedencia_de_operadores(self):
        assert calculator.invoke({"expression": "(4 + 5) * 3 / 2"}) == "Resultado: 13.5"

    def test_potencia(self):
        assert calculator.invoke({"expression": "2 ** 8"}) == "Resultado: 256"

    def test_division_por_cero(self):
        assert "división entre cero" in calculator.invoke({"expression": "1 / 0"})

    def test_rechaza_potencia_encadenada(self):
        """`9**9**9` colgaba el proceso antes; ahora se rechaza."""
        resultado = calculator.invoke({"expression": "9 ** 9 ** 9"})
        assert "exponente" in resultado

    def test_rechaza_resultado_enorme(self):
        assert "demasiado grande" in calculator.invoke({"expression": "(10**100) ** 100"})

    def test_rechaza_desbordamiento_a_infinito(self):
        assert "demasiado grande" in calculator.invoke({"expression": "1e308 * 10"})

    def test_acepta_operadores_completos(self):
        assert calculator.invoke({"expression": "10 % 3"}) == "Resultado: 1"
        assert calculator.invoke({"expression": "7 // 2"}) == "Resultado: 3"
        assert calculator.invoke({"expression": "-5 + 3"}) == "Resultado: -2"
        assert calculator.invoke({"expression": "2 ** 0.5"}).startswith("Resultado: 1.41")

    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os')",
            "open('secreto.txt')",
            "(1).__class__",
            "[x for x in range(3)]",
            "lambda: 1",
            "a + 1",
            "1; 2",
            "",
        ],
    )
    def test_rechaza_entrada_no_segura(self, expression):
        resultado = calculator.invoke({"expression": expression})
        assert resultado.startswith("Error"), resultado

    def test_no_puede_leer_el_disco(self):
        """El sandbox no debe permitir I/O aunque la expresión lo pida."""
        resultado = calculator.invoke({"expression": "open('C:/Windows/win.ini').read()"})
        assert resultado.startswith("Error")


# ─────────────────────────────────────────────────────────────────────────────
# Búsqueda en archivos
# ─────────────────────────────────────────────────────────────────────────────


class TestSearchFiles:
    @pytest.fixture
    def data_dir(self, tmp_path, monkeypatch):
        """DATA_DIR temporal con .txt, .md en subcarpeta y un binario."""
        (tmp_path / "notas.txt").write_text(
            "LangGraph es un orquestador de grafos\nOllama corre modelos locales",
            encoding="utf-8",
        )
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "guia.md").write_text(
            "# Guia\nUn conditional edge decide la ruta", encoding="utf-8"
        )
        (tmp_path / "ignorado.bin").write_text("LangGraph", encoding="utf-8")
        monkeypatch.setattr("agent.tools.DATA_DIR", tmp_path)
        return tmp_path

    def test_encuentra_en_txt(self, data_dir):
        assert "notas.txt:1" in search_files.invoke({"query": "LangGraph"})

    def test_recorre_subdirectorios(self, data_dir):
        assert "guia.md" in search_files.invoke({"query": "conditional edge"})

    def test_ignora_extensiones_no_texto(self, data_dir):
        assert "ignorado.bin" not in search_files.invoke({"query": "LangGraph"})

    def test_case_insensitive(self, data_dir):
        assert "coincidencia" in search_files.invoke({"query": "langgraph"})

    def test_sin_coincidencias(self, data_dir):
        assert "No se encontró" in search_files.invoke({"query": "inexistente"})

    def test_directorio_inexistente(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.tools.DATA_DIR", tmp_path / "no-existe")
        assert "No hay archivos" in search_files.invoke({"query": "x"})
