"""Groq adapter tests. No network: they check the format translation and provider choice."""
import json
from types import SimpleNamespace as NS

from realestate_agent import llm


def test_history_translates_to_openai_messages():
    system = [{"type": "text", "text": "rules"}, {"type": "text", "text": "memory"}]
    history = [
        {"role": "user", "content": "find homes"},
        {"role": "assistant", "content": [
            NS(type="text", text="Searching."),
            NS(type="tool_use", id="c1", name="search_properties", input={"max_price": 5}),
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "[]", "is_error": False},
        ]},
    ]
    out = llm.to_openai_messages(system, history)
    assert out[0] == {"role": "system", "content": "rules\n\nmemory"}
    assert out[1] == {"role": "user", "content": "find homes"}
    call = out[2]["tool_calls"][0]
    assert out[2]["content"] == "Searching."
    assert call["id"] == "c1" and json.loads(call["function"]["arguments"]) == {"max_price": 5}
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "[]"}


def test_tool_schema_and_response_translate_back():
    tool = {"name": "t", "description": "d", "input_schema": {"type": "object"}}
    assert llm.to_openai_tool(tool)["function"]["parameters"] == {"type": "object"}

    call = NS(id="c9", function=NS(name="get_property", arguments='{"property_id": "P-103"}'))
    response = NS(choices=[NS(finish_reason="tool_calls", message=NS(content=None, tool_calls=[call]))])
    out = llm.from_openai_response(response)
    assert out.stop_reason == "tool_use"
    assert out.content[0].input == {"property_id": "P-103"}

    text = NS(choices=[NS(finish_reason="stop", message=NS(content="Done.", tool_calls=None))])
    out = llm.from_openai_response(text)
    assert out.stop_reason == "end_turn" and out.content[0].text == "Done."


def test_provider_choice(monkeypatch):
    for k in ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GROQ_API_KEY", "RE_AGENT_PROVIDER"]:
        monkeypatch.delenv(k, raising=False)
    assert llm.provider() is None
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert llm.provider() == "groq"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "y")
    assert llm.provider() == "anthropic"
    monkeypatch.setenv("RE_AGENT_PROVIDER", "groq")
    assert llm.provider() == "groq"
