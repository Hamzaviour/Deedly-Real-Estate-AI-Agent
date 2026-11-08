"""Agent loop tests. The LLM is scripted, so these test the plumbing, not model judgement.

Model behaviour is checked separately by running demo.py with a real API key.
"""
import asyncio
import json
from types import SimpleNamespace as NS

import anthropic
import httpx

from realestate_agent import properties
from realestate_agent.agent import PropertyAgent
from realestate_agent.mcp_client import ListingsMCPClient
from realestate_agent.memory import LongTermMemory


def text(body):
    return NS(stop_reason="end_turn", content=[NS(type="text", text=body)])


def tool(name, tool_id="t1", **args):
    return NS(
        stop_reason="tool_use",
        content=[NS(type="tool_use", id=tool_id, name=name, input=args)],
    )


class ScriptedLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = NS(messages=NS(create=self._create))

    async def _create(self, **kwargs):
        # Snapshot now: the agent keeps appending to the same list afterwards.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        next_item = self.responses.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item


class FakeKnowledge:
    def retrieve(self, query):
        if "tungsten" in query:
            return []
        return [{"citation": "01_buying_process.md#2", "source": "01_buying_process.md",
                 "section": "Deposits", "similarity": 0.8, "text": "No more than 10 percent."}]


def make_agent(llm, mcp, tools, approve=lambda n, a: True, user="u1"):
    return PropertyAgent(llm, mcp, FakeKnowledge(), LongTermMemory(), user, approve, tools)


def with_mcp(body):
    async def wrapper():
        async with ListingsMCPClient() as mcp:
            return await body(mcp, await mcp.list_tools())

    return asyncio.run(wrapper())


def tool_results(call):
    return call["messages"][-1]["content"]


def test_mcp_tool_call_runs_and_result_returns_to_model():
    llm = ScriptedLLM(tool("search_properties", max_price=27000000, min_beds=3), text("P-103 fits."))

    async def body(mcp, tools):
        return await make_agent(llm, mcp, tools).chat("what fits?")

    result = with_mcp(body)
    assert result.text == "P-103 fits."
    assert result.trace[0]["via"] == "mcp" and result.trace[0]["is_error"] is False
    assert "P-103" in tool_results(llm.calls[1])[0]["content"]


def test_request_uses_expected_model_settings():
    llm = ScriptedLLM(text("hi"))

    async def body(mcp, tools):
        await make_agent(llm, mcp, tools).chat("hello")

    with_mcp(body)
    call = llm.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["fallbacks"] == "default"
    names = [t["name"] for t in call["tools"]]
    assert names == sorted(names) and "retrieve_knowledge" in names and "get_property" in names


def test_short_term_memory_sends_earlier_turns():
    llm = ScriptedLLM(text("first"), text("second"))

    async def body(mcp, tools):
        agent = make_agent(llm, mcp, tools)
        await agent.chat("I like Allama Iqbal Town")
        await agent.chat("what did I say?")

    with_mcp(body)
    assert llm.calls[1]["messages"][0]["content"] == "I like Allama Iqbal Town"
    assert len(llm.calls[1]["messages"]) == 3


def test_long_term_memory_reaches_a_new_session_but_short_term_does_not():
    first = ScriptedLLM(
        tool("save_preference", key="budget_pkr", value="27000000", category="constraint"),
        text("saved"),
    )
    second = ScriptedLLM(text("your budget is 27M"))

    async def body(mcp, tools):
        await make_agent(first, mcp, tools).chat("my budget is 27M")
        await make_agent(second, mcp, tools).chat("what is my budget?")

    with_mcp(body)
    call = second.calls[0]
    assert "27000000" in call["system"][1]["text"]
    assert len(call["messages"]) == 1


def test_rag_citations_are_recorded_and_empty_result_is_explicit():
    llm = ScriptedLLM(tool("retrieve_knowledge", query="deposit"),
                      tool("retrieve_knowledge", query="tungsten"), text("done"))

    async def body(mcp, tools):
        return await make_agent(llm, mcp, tools).chat("deposit?")

    result = with_mcp(body)
    assert result.citations == ["01_buying_process.md#2"]
    assert json.loads(result.trace[1]["output"])["passages"] == []


def test_declined_viewing_never_reaches_the_server():
    llm = ScriptedLLM(
        tool("request_viewing", property_id="P-103", buyer_name="H", preferred_date="2026-11-20"),
        text("ok, not sent"),
    )

    async def body(mcp, tools):
        return await make_agent(llm, mcp, tools, approve=lambda n, a: False).chat("book it")

    result = with_mcp(body)
    assert "declined" in result.trace[0]["output"]
    assert "reference" not in result.trace[0]["output"]
    assert result.trace[0]["via"] == "approval"


def test_approved_viewing_reaches_the_server():
    seen = []
    llm = ScriptedLLM(
        tool("request_viewing", property_id="P-103", buyer_name="H", preferred_date="2026-11-20"),
        text("sent"),
    )

    async def body(mcp, tools):
        def approve(name, args):
            seen.append(name)
            return True

        return await make_agent(llm, mcp, tools, approve=approve).chat("book it")

    result = with_mcp(body)
    assert seen == ["request_viewing"]
    assert "VR-" in result.trace[0]["output"]


def test_change_event_is_visible_to_the_agent_through_mcp():
    llm = ScriptedLLM(tool("get_property", property_id="P-103"), text("it was sold"))

    async def body(mcp, tools):
        properties.set_status("P-103", "sold")
        return await make_agent(llm, mcp, tools).chat("is P-103 ok?")

    result = with_mcp(body)
    assert json.loads(result.trace[0]["output"])["status"] == "sold"


def test_mcp_error_is_sent_to_model_flagged():
    llm = ScriptedLLM(tool("get_property", property_id="P-999"), text("could not find it"))

    async def body(mcp, tools):
        return await make_agent(llm, mcp, tools).chat("P-999?")

    result = with_mcp(body)
    assert result.ok and result.trace[0]["is_error"] is True
    assert tool_results(llm.calls[1])[0]["is_error"] is True


def test_local_tool_error_does_not_crash_the_turn():
    llm = ScriptedLLM(tool("save_preference", key="", value="x", category="note"), text("sorry"))

    async def body(mcp, tools):
        return await make_agent(llm, mcp, tools).chat("save")

    result = with_mcp(body)
    assert result.ok and result.trace[0]["is_error"] is True


def test_api_error_reports_failure_and_leaves_history_valid():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    llm = ScriptedLLM(text("first"), anthropic.APIConnectionError(request=request), text("third"))

    async def body(mcp, tools):
        agent = make_agent(llm, mcp, tools)
        await agent.chat("one")
        failed = await agent.chat("two")
        assert not failed.ok and "reach" in failed.text
        assert len(agent.messages) == 2
        return await agent.chat("three")

    assert with_mcp(body).text == "third"


def test_refusal_drops_the_turn():
    llm = ScriptedLLM(NS(stop_reason="refusal", content=[]), text("fine"))

    async def body(mcp, tools):
        agent = make_agent(llm, mcp, tools)
        result = await agent.chat("bad")
        assert not result.ok and agent.messages == []
        return await agent.chat("good")

    assert with_mcp(body).text == "fine"


def test_runaway_tool_loop_stops():
    llm = ScriptedLLM(*[tool("get_property", property_id="P-103") for _ in range(20)])

    async def body(mcp, tools):
        agent = make_agent(llm, mcp, tools)
        result = await agent.chat("loop")
        return result, agent

    result, agent = with_mcp(body)
    assert not result.ok and "too many" in result.text
    assert agent.messages == []
