"""Web layer tests with a scripted LLM. Approval uses ChatSession directly because the test
client buffers streaming responses, which would deadlock while waiting for a click."""
import asyncio

from starlette.testclient import TestClient

from realestate_agent import properties
from realestate_agent.mcp_client import ListingsMCPClient
from realestate_agent.web import ChatSession, create_app

from test_agent import ScriptedLLM, make_agent, text, tool

VIEWING = dict(property_id="P-103", buyer_name="H", preferred_date="2026-11-20")


def client(llm=None):
    return TestClient(create_app(llm=llm))


def test_page_and_state_load_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with client() as c:
        assert "Deedly" in c.get("/").text
        assert c.get("/static/brand/deedly-mark.svg").status_code == 200
        state = c.get("/api/state").json()
        assert state["llm_ready"] is False
        assert len(state["properties"]) == 180
        assert "search_properties" in state["mcp_tools"]
        r = c.post("/api/chat", json={"message": "hi"})
        assert r.status_code == 503 and "API key" in r.json()["error"]


def test_chat_returns_answer_and_memory_shows_in_state():
    llm = ScriptedLLM(
        tool("save_preference", key="budget_pkr", value="27000000", category="constraint"),
        text("Saved."),
    )
    with client(llm) as c:
        lines = c.post("/api/chat", json={"user": "u9", "message": "budget 27M"}).text
        assert '"Saved."' in lines
        assert c.get("/api/state?user=u9").json()["memories"][0]["value"] == "27000000"


def test_new_session_clears_conversation():
    llm = ScriptedLLM(text("one"), text("two"))
    with client(llm) as c:
        c.post("/api/chat", json={"message": "first"})
        assert c.post("/api/reset", json={}).status_code == 200
        c.post("/api/chat", json={"message": "second"})
    assert len(llm.calls[1]["messages"]) == 1


def test_status_change_event_and_validation():
    with client(ScriptedLLM()) as c:
        assert c.post("/api/status", json={"id": "P-103", "status": "sold"}).status_code == 200
        assert properties.get("P-103")["status"] == "sold"
        assert c.post("/api/status", json={"id": "P-103", "status": "bad"}).status_code == 400
        assert c.post("/api/status", json={"id": "P-999", "status": "sold"}).status_code == 404


def test_rejects_non_json_posts_foreign_hosts_and_bad_messages():
    with client(ScriptedLLM()) as c:
        assert c.post("/api/chat", content="message=hi",
                      headers={"content-type": "text/plain"}).status_code == 400
        assert c.post("/api/chat", json={"message": "  "}).status_code == 400
        assert c.post("/api/chat", json={"message": "x" * 4001}).status_code == 400
        assert c.get("/", headers={"host": "evil.example"}).status_code == 400


def run_viewing_turn(approved: bool | None):
    llm = ScriptedLLM(tool("request_viewing", **VIEWING), text("finished"))

    async def body():
        async with ListingsMCPClient() as mcp:
            session = ChatSession(make_agent(llm, mcp, await mcp.list_tools()))
            events = []
            async for event in session.stream("book it"):
                events.append(event)
                if event["type"] == "approval":
                    assert session.resolve(event["id"], approved)
            return events

    return asyncio.run(body())


def test_browser_approval_sends_request():
    events = run_viewing_turn(True)
    assert events[0]["type"] == "approval" and events[0]["input"] == VIEWING
    assert events[1]["type"] == "tool" and "VR-" in events[1]["step"]["output"]
    assert "VR-" in events[-1]["trace"][0]["output"]


def test_browser_decline_blocks_request():
    assert "declined" in run_viewing_turn(False)[-1]["trace"][0]["output"]


def test_unknown_approval_id_is_rejected():
    async def body():
        return ChatSession(agent=None).resolve("missing", True)

    assert asyncio.run(body()) is False


def test_every_home_has_served_photos(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with client() as c:
        manifest = c.get("/static/homes/photos.json").json()
        ids = {p["id"] for p in c.get("/api/state").json()["properties"]}
        assert ids <= set(manifest)
        for home in ids:
            assert manifest[home], home
            for photo in manifest[home]:
                assert photo["alt"]
                for key in ("src", "thumb"):
                    r = c.get(photo[key])
                    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


def test_shortlist_hearts_write_long_term_memory():
    with client(ScriptedLLM()) as c:
        add = lambda pid, on: c.post("/api/shortlist", json={"user": "sl1", "id": pid, "on": on})
        assert add("P-103", True).json()["shortlist"] == ["P-103"]
        assert add("P-110", True).json()["shortlist"] == ["P-103", "P-110"]
        assert add("P-103", False).json()["shortlist"] == ["P-110"]
        state = c.get("/api/state?user=sl1").json()
        assert state["shortlist"] == ["P-110"]
        assert any(m["key"] == "shortlist" and m["category"] == "shortlist" for m in state["memories"])
        assert add("P-999", True).status_code == 404
        assert c.post("/api/shortlist", json={"id": "P-103"}).status_code == 400
        add("P-110", False)
        assert c.get("/api/state?user=sl1").json()["shortlist"] == []


def test_passage_endpoint_returns_cited_text():
    with client(ScriptedLLM()) as c:
        r = c.get("/api/passage", params={"id": "09_measurements_marla_kanal.md#2"})
        assert r.status_code == 200 and "4,500 square feet" in r.json()["text"]
        assert c.get("/api/passage", params={"id": "missing.md#0"}).status_code == 404


def test_hosted_mode_scopes_buyer_names_to_a_signed_cookie(monkeypatch):
    from realestate_agent import web

    monkeypatch.setattr(web, "SCOPE_USERS", True)
    llm = ScriptedLLM(
        tool("save_preference", key="budget_pkr", value="27000000", category="constraint"),
        text("Saved."),
    )
    with client(llm) as a, client(ScriptedLLM()) as b:
        assert a.post("/api/chat", json={"user": "hamza", "message": "x"}).status_code == 401
        a.get("/api/state?user=hamza")  # Issues the signed cookie.
        assert web.COOKIE in a.cookies
        a.post("/api/chat", json={"user": "hamza", "message": "budget 27M"})
        assert a.get("/api/state?user=hamza").json()["memories"][0]["value"] == "27000000"
        # Another browser typing the same name sees nothing, and a forged cookie is rejected.
        assert b.get("/api/state?user=hamza").json()["memories"] == []
        b.cookies.clear()  # Drop b's own valid cookie so only the forged one is sent.
        b.cookies.set(web.COOKIE, "0123456789abcdef.forged")
        assert b.post("/api/reset", json={"user": "hamza"}).status_code == 401


def test_sessions_are_capped_and_idle_ones_evicted(monkeypatch):
    from realestate_agent import web

    monkeypatch.setattr(web, "MAX_SESSIONS", 3)
    llm = ScriptedLLM(*[text(str(i)) for i in range(6)])
    with client(llm) as c:
        for i in range(5):
            c.post("/api/chat", json={"user": f"u{i}", "message": "hi"})
        sessions = c.app.state.deedly["sessions"]
        assert set(sessions) == {"u2", "u3", "u4"}  # Least recently used went first.
        for s in sessions.values():
            s.last_used -= web.SESSION_IDLE_SECONDS + 1
        c.post("/api/chat", json={"user": "new", "message": "hi"})
        assert set(sessions) == {"new"}


def test_urdu_choice_adds_reply_language_to_the_system_prompt():
    llm = ScriptedLLM(text("جی"), text("ok"))
    with client(llm) as c:
        c.post("/api/chat", json={"user": "ur1", "message": "salaam", "lang": "ur"})
        c.post("/api/chat", json={"user": "ur1", "message": "hello"})
    first, second = (" ".join(b["text"] for b in call["system"]) for call in llm.calls)
    assert "Urdu" in first and "Urdu" not in second
