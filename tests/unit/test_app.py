"""Aplikacja Streamlit: każda zakładka i każdy atak renderuje się bez wyjątku."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from qkdsim.scenarios import ATTACKS, TIME_SHIFT

APP = str(Path(__file__).parents[2] / "app" / "streamlit_app.py")


@pytest.mark.parametrize("attack", ATTACKS)
def test_app_renders_every_attack(attack: str) -> None:
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    app.sidebar.selectbox[0].set_value(attack).run()
    app.sidebar.select_slider[0].set_value(300_000 if attack != TIME_SHIFT else 1_000_000).run()
    assert not app.exception, app.exception
    assert len(app.metric) >= 6  # trzy w zakładce ataku i trzy w zakładce klucza
    assert app.title[0].value == "Symulator BB84 z atakami"
