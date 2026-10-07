from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
DEMO = APP.parent / "demo_data"
pytestmark = pytest.mark.skipif(not (DEMO / "overview.json").exists(), reason="run `make demo` first")


@pytest.mark.parametrize("page", ["Vue d'ensemble", "Détail d'une machine", "Surveillance", "Modèle", "Coût"])
def test_every_page_renders_without_error(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title
