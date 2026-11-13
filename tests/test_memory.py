import pytest

from realestate_agent.memory import LongTermMemory


def test_memory_survives_new_instance():
    LongTermMemory().save("u1", "budget_pkr", "27000000", "constraint")
    assert LongTermMemory().load("u1")[0]["value"] == "27000000"


def test_same_key_replaces_value():
    m = LongTermMemory()
    m.save("u1", "budget_pkr", "27000000")
    m.save("u1", "budget_pkr", "30000000")
    rows = m.load("u1")
    assert len(rows) == 1 and rows[0]["value"] == "30000000"


def test_users_are_isolated():
    m = LongTermMemory()
    m.save("u1", "area", "Allama Iqbal Town")
    assert m.load("u2") == []


def test_forget():
    m = LongTermMemory()
    m.save("u1", "area", "Allama Iqbal Town")
    assert m.forget("u1", "area") is True
    assert m.forget("u1", "area") is False
    assert m.load("u1") == []


@pytest.mark.parametrize("key,value,cat", [("", "x", "note"), ("k", " ", "note"), ("k", "v", "bad")])
def test_rejects_invalid(key, value, cat):
    with pytest.raises(ValueError):
        LongTermMemory().save("u1", key, value, cat)


def test_factory_uses_sqlite_without_database_url(monkeypatch):
    from realestate_agent.memory import LongTermMemory, make_memory

    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert isinstance(make_memory(), LongTermMemory)


def test_postgres_memory_round_trip():
    """Runs only when TEST_DATABASE_URL points at a disposable Postgres database."""
    import os
    import uuid

    import pytest

    from realestate_agent.memory import PostgresMemory

    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    m, user = PostgresMemory(url), f"test-{uuid.uuid4().hex}"
    m.save(user, "budget_pkr", "27000000", "constraint")
    m.save(user, "budget_pkr", "30000000", "constraint")
    assert [r["value"] for r in m.load(user)] == ["30000000"]
    assert m.forget(user, "budget_pkr") and m.load(user) == []
