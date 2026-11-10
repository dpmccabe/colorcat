import pytest

from colorcat import _energy


@pytest.fixture(autouse=True)
def _few_hops(monkeypatch):
    """Keep the suite fast: the library default is 16 hops per restart."""
    monkeypatch.setattr(_energy, "HOP_COUNT", 2)
