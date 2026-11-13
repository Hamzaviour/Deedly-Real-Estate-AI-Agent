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


def test_model_discovery_prefers_tool_capable_models():
    assert llm.pick_groq_model(["whisper-large-v3", "llama-3.1-8b-instant", "openai/gpt-oss-120b"]) == "openai/gpt-oss-120b"
    # Unknown future names: skip audio and guard models, prefer the largest chat model.
    assert llm.pick_groq_model(["whisper-x", "llama-guard-9-12b", "newco/chat-30b", "newco/chat-200b"]) == "newco/chat-200b"
    assert llm.pick_groq_model(["whisper-large-v3"]) is None


def test_retired_model_is_rediscovered_and_retried():
    import asyncio

    import httpx
    import openai

    class FakeCompletions:
        def __init__(self):
            self.models = []

        async def create(self, model, **kw):
            self.models.append(model)
            if model == "retired-model":
                req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
                raise openai.NotFoundError("model not found", response=httpx.Response(404, request=req), body=None)
            return NS(choices=[NS(finish_reason="stop", message=NS(content="ok", tool_calls=None))])

    class FakeModels:
        async def list(self):
            return NS(data=[NS(id="openai/gpt-oss-120b"), NS(id="whisper-large-v3")])

    g = llm.GroqLLM(api_key="x", model="placeholder")
    g._resolved, g.model = True, "retired-model"
    completions = FakeCompletions()
    g._client = NS(chat=NS(completions=completions), models=FakeModels())
    out = asyncio.run(g._create(system="s", tools=[], messages=[{"role": "user", "content": "hi"}], max_tokens=100))
    assert out.content[0].text == "ok"
    assert completions.models == ["retired-model", "openai/gpt-oss-120b"]


def test_stream_collector_rebuilds_text_and_tool_calls():
    import asyncio
    from types import SimpleNamespace as NS

    from realestate_agent.llm import collect_openai_stream

    def chunk(content=None, calls=None, finish=None):
        return NS(choices=[NS(delta=NS(content=content, tool_calls=calls), finish_reason=finish)])

    def call(index, id=None, name=None, args=None):
        return NS(index=index, id=id, function=NS(name=name, arguments=args))

    async def stream():
        for c in [chunk("Let me "), chunk("check."),
                  chunk(calls=[call(0, "c1", "get_property", '{"property_')]),
                  chunk(calls=[call(0, args='id": "P-103"}')], finish="tool_calls")]:
            yield c

    seen = []
    out = asyncio.run(collect_openai_stream(stream(), seen.append))
    assert seen == ["Let me ", "check."]
    assert out.stop_reason == "tool_use"
    assert out.content[0].text == "Let me check."
    assert out.content[1].name == "get_property" and out.content[1].input == {"property_id": "P-103"}


def test_agent_streams_text_when_the_provider_can():
    import asyncio
    from types import SimpleNamespace as NS

    from realestate_agent.agent import PropertyAgent
    from realestate_agent.memory import LongTermMemory

    class StreamingLLM:
        streams_text = True

        async def stream_create(self, *, on_text, **_):
            for piece in ("Hello ", "there."):
                on_text(piece)
            return NS(stop_reason="end_turn", content=[NS(type="text", text="Hello there.")])

    agent = PropertyAgent(StreamingLLM(), None, None, LongTermMemory(), "u", lambda *a: True, [])
    pieces = []
    agent.on_text = pieces.append
    result = asyncio.run(agent.chat("hi"))
    assert pieces == ["Hello ", "there."] and result.text == "Hello there."
