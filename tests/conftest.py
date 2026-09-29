import pytest

from career_kg import dataset, graph_store
from career_kg.recommender import Recommender


@pytest.fixture(scope="session")
def ds():
    return dataset.parse()


@pytest.fixture(scope="session")
def rec():
    try:
        driver = graph_store.connect()
        stats = graph_store.stats(driver)[0]
    except Exception as exc:
        pytest.skip(f"Neo4j not available: {exc}")
    if not stats.get("Occupation"):
        pytest.skip("Neo4j is empty; run `make ingest`")
    return Recommender(driver)
