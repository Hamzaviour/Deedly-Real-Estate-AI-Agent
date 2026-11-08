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
