"""Local web app. Run: python -m realestate_agent.web, then open http://127.0.0.1:8011

Set RE_AGENT_PORT to use another port. Binds to localhost only. There is no login, so do not expose it to a network.
"""
import asyncio
import contextlib
import json
import os
import re
import uuid
from pathlib import Path


import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import properties
from .agent import PropertyAgent
from .config import MODEL
from .llm import make_llm, model_name
from .mcp_client import ListingsMCPClient
from .memory import LongTermMemory
from .rag import KnowledgeBase

PAGE = Path(__file__).with_name("static") / "index.html"
HOST, PORT = "127.0.0.1", int(os.getenv("RE_AGENT_PORT", "8011"))
APPROVAL_TIMEOUT_SECONDS = 300
MAX_MESSAGE_CHARS = 4000
STATUSES = {"available", "under_offer", "sold"}
NO_KEY_MESSAGE = (
    "No API key is set, so the advisor cannot answer yet. Copy .env.example to .env, "
    "set ANTHROPIC_API_KEY or GROQ_API_KEY, and restart the server."
)


def shortlist_ids(memories: list[dict]) -> set[str]:
    """Home ids in any shortlist memory, whether saved by the agent or by a heart click."""
    return {i for m in memories if m["category"] == "shortlist" for i in re.findall(r"P-\d{3}", m["value"])}


class ChatSession:
    """One browser conversation. Replacing it starts a new short-term memory."""

    def __init__(self, agent: PropertyAgent):
        self.agent = agent
        self.lock = asyncio.Lock()
        self.pending: dict[str, asyncio.Future] = {}

    def resolve(self, approval_id: str, approved: bool) -> bool:
        future = self.pending.get(approval_id)
        if future is None or future.done():
            return False
        future.set_result(approved)
        return True

    async def stream(self, text: str):
        """Yield approval requests as they occur, then the finished turn."""
        queue: asyncio.Queue = asyncio.Queue()

        async def approve(tool: str, args: dict) -> bool:
            approval_id = uuid.uuid4().hex
            future = asyncio.get_running_loop().create_future()
            self.pending[approval_id] = future
            await queue.put({"type": "approval", "id": approval_id, "tool": tool, "input": args})
            try:
                return await asyncio.wait_for(future, APPROVAL_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                return False  # No answer counts as no approval.
            finally:
                self.pending.pop(approval_id, None)

        def on_tool(step: dict) -> None:
            queue.put_nowait({"type": "tool", "step": step})

        async def run():
            async with self.lock:
                self.agent.approve = approve
                self.agent.on_tool = on_tool
                return await self.agent.chat(text)

        task = asyncio.create_task(run())
        task.add_done_callback(lambda _: queue.put_nowait(None))
        while (event := await queue.get()) is not None:
            yield event
        try:
            yield {"type": "done", **task.result().to_dict()}
        except Exception as e:
            yield {"type": "done", "text": f"Unexpected error: {type(e).__name__}", "ok": False,
                   "trace": [], "citations": []}


def create_app(llm=None) -> Starlette:
    """Build the app. Tests can pass a scripted llm."""
    state: dict = {}

    def new_session(user: str) -> ChatSession:
        agent = PropertyAgent(
            state["llm"], state["mcp"], state["knowledge"], state["memory"], user,
            lambda tool, args: False, state["tools"],
        )
        return ChatSession(agent)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        if llm is not None:
            state["llm"] = llm
        else:
            state["llm"] = make_llm()
        state["knowledge"] = KnowledgeBase()
        state["memory"] = LongTermMemory()
        state["sessions"] = {}
        async with ListingsMCPClient() as mcp:
            state["mcp"] = mcp
            state["tools"] = await mcp.list_tools()
            yield

    def session_for(user: str) -> ChatSession:
        if user not in state["sessions"]:
            state["sessions"][user] = new_session(user)
        return state["sessions"][user]

    async def read_json(request: Request) -> dict | None:
        # Requiring JSON makes browsers send a CORS preflight for cross-site requests, which this
        # app never approves, so other websites cannot post to it.
        if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
            return None
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return None
        return body if isinstance(body, dict) else None

    def clean_user(value) -> str:
        return value.strip()[:64] if isinstance(value, str) and value.strip() else "demo_buyer"

    async def index(request: Request):
        return HTMLResponse(PAGE.read_text(encoding="utf-8"))

    async def get_state(request: Request):
        user = clean_user(request.query_params.get("user"))
        return JSONResponse(
            {
                "llm_ready": state["llm"] is not None,
                "model": model_name(state["llm"], MODEL),
                "mcp_tools": [t["name"] for t in state["tools"]],
                "memories": (memories := state["memory"].load(user)),
                "shortlist": sorted(shortlist_ids(memories)),
                "properties": properties.search(include_unavailable=True),
            }
        )

    async def chat(request: Request):
        body = await read_json(request)
        text = body.get("message") if body else None
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_MESSAGE_CHARS:
            return JSONResponse({"error": "Message must be 1 to 4000 characters."}, status_code=400)
        if state["llm"] is None:
            return JSONResponse({"error": NO_KEY_MESSAGE}, status_code=503)
        session = session_for(clean_user(body.get("user")))
        if session.lock.locked():
            return JSONResponse({"error": "Still working on your last message."}, status_code=409)

        async def lines():
            async for event in session.stream(text.strip()):
                yield json.dumps(event) + "\n"

        return StreamingResponse(lines(), media_type="application/x-ndjson")

    async def approve(request: Request):
        body = await read_json(request)
        if not body or not isinstance(body.get("approved"), bool):
            return JSONResponse({"error": "Invalid request."}, status_code=400)
        session = state["sessions"].get(clean_user(body.get("user")))
        if session is None or not session.resolve(str(body.get("id")), body["approved"]):
            return JSONResponse({"error": "That approval has expired."}, status_code=404)
        return JSONResponse({"ok": True})

    async def reset_session(request: Request):
        body = await read_json(request)
        if body is None:
            return JSONResponse({"error": "Invalid request."}, status_code=400)
        user = clean_user(body.get("user"))
        current = state["sessions"].get(user)
        if current and current.lock.locked():
            return JSONResponse({"error": "Still working on your last message."}, status_code=409)
        state["sessions"].pop(user, None)
        return JSONResponse({"ok": True})

    async def shortlist(request: Request):
        """Heart a home. Writes the buyer's shortlist to long-term memory, the same store the
        agent reads, so a shortlist made by clicking is known to the agent next session."""
        body = await read_json(request)
        if not body or not isinstance(body.get("on"), bool):
            return JSONResponse({"error": "Invalid request."}, status_code=400)
        home_id = str(body.get("id"))
        try:
            properties.get(home_id)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        user = clean_user(body.get("user"))
        ids = shortlist_ids(state["memory"].load(user))
        ids = ids | {home_id} if body["on"] else ids - {home_id}
        memory = state["memory"]
        for m in memory.load(user):  # Fold id-only shortlist entries into one key.
            if m["category"] == "shortlist" and not re.sub(r"P-\d{3}|[\s,;]|and", "", m["value"]):
                memory.forget(user, m["key"])
        if ids:
            memory.save(user, "shortlist", ", ".join(sorted(ids)), "shortlist")
        return JSONResponse({"shortlist": sorted(shortlist_ids(memory.load(user)))})

    async def set_status(request: Request):
        """Demo control for the change event. The LLM has no tool for this."""
        body = await read_json(request)
        if not body or body.get("status") not in STATUSES:
            return JSONResponse({"error": "Invalid status."}, status_code=400)
        try:
            properties.set_status(str(body.get("id")), body["status"])
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse({"ok": True})

    return Starlette(
        routes=[
            Route("/", index),
            Route("/api/state", get_state),
            Route("/api/chat", chat, methods=["POST"]),
            Route("/api/approve", approve, methods=["POST"]),
            Route("/api/reset", reset_session, methods=["POST"]),
            Route("/api/status", set_status, methods=["POST"]),
            Route("/api/shortlist", shortlist, methods=["POST"]),
            Mount("/static", StaticFiles(directory=PAGE.parent)),
        ],
        lifespan=lifespan,
        middleware=[
            # Blocks DNS rebinding: only requests addressed to localhost are served.
            Middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
        ],
    )


if __name__ == "__main__":
    uvicorn.run(create_app(), host=HOST, port=PORT)
