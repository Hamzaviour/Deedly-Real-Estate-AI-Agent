"""These tests start the real MCP server subprocess and call it through the MCP client."""
import asyncio
import json

from realestate_agent.mcp_client import ListingsMCPClient


def run(coro_fn):
    async def wrapper():
        async with ListingsMCPClient() as client:
            return await coro_fn(client)

    return asyncio.run(wrapper())


def test_lists_expected_tools_in_stable_order():
    async def body(c):
        return await c.list_tools()

    tools = run(body)
    names = [t["name"] for t in tools]
    assert names == sorted(names)
    assert set(names) == {
        "affordability_check",
        "area_price_stats",
        "compare_properties",
        "estimate_monthly_payment",
        "get_property",
        "request_viewing",
        "search_properties",
    }
    assert all(t["input_schema"]["type"] == "object" for t in tools)


def test_real_tool_invocation_returns_data():
    async def body(c):
        return await c.call_tool("search_properties", {"max_price": 27000000, "min_beds": 3})

    text, is_error = run(body)
    assert not is_error
    assert "P-103" in {h["id"] for h in json.loads(text)}


def test_tool_failure_is_flagged_as_error():
    async def body(c):
        return await c.call_tool("get_property", {"property_id": "P-999"})

    text, is_error = run(body)
    assert is_error
    assert "P-999" in text


def test_area_price_stats_and_affordability_over_mcp():
    async def body(c):
        stats = await c.call_tool("area_price_stats", {"area": "DHA Phase 6"})
        afford = await c.call_tool("affordability_check",
                                   {"monthly_income_pkr": 500000, "down_payment_pkr": 8000000})
        return stats, afford

    (stats, err1), (afford, err2) = run(body)
    assert not err1 and not err2
    row = json.loads(stats)[0]
    assert row["area"] == "DHA Phase 6" and row["median_price_per_marla"] > 0
    assert json.loads(afford)["max_price_pkr"] > 8000000
