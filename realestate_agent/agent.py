"""The agent loop: Claude picks tools, local tools and MCP tools run, results go back to Claude.

A manual loop is used so every decision can be traced and viewing requests can be gated.
"""
import inspect
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Callable

import anthropic

from .config import MODEL
from .llm import LLMServiceError
from .prompts import SYSTEM_PROMPT, memory_block

GATED_TOOLS = {"request_viewing"}
LANGUAGE_NOTES = {
    "ur": ("# Reply language\nThe buyer chose Urdu. Write your reply in Urdu script. Keep listing "
           "ids (P-103), PKR amounts, numbers and citation values exactly as they are, in Latin "
           "characters. Tool inputs stay in English."),
}
CITATION = re.compile(r"\[([\w.-]+\.md#\d+)\]")
MAX_STEPS = 10

LOCAL_TOOLS = [
    {
        "name": "area_guide",
        "description": (
            "Get Demo Realty's neighbourhood guide passages for one Lahore area: commute, "
            "schools, amenities, flooding and character. Returns passages with citations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"area": {"type": "string", "description": "Area name, such as DHA Phase 6."}},
            "required": ["area"],
            "additionalProperties": False,
        },
    },
    {
        "name": "forget_preference",
        "description": "Delete one saved buyer memory by key, when the buyer asks.",
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
    },
    {
        "name": "retrieve_knowledge",
        "description": (
            "Search Demo Realty's documents on buying steps, deposits, title checks, financing "
            "rules, inspection, neighbourhoods and advisory policy. Returns passages with "
            "citations. Does not contain listings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look up."}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "save_preference",
        "description": (
            "Save a lasting buyer fact to long-term memory for future sessions. "
            "Saving under an existing key replaces the old value."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Short name, such as budget_pkr."},
                "value": {"type": "string"},
                "category": {
                    "type": "string",
                    "enum": ["preference", "constraint", "shortlist", "note"],
                },
            },
            "required": ["key", "value", "category"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class TurnResult:
    text: str
    trace: list[dict] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    grounding: dict = field(default_factory=dict)
    ok: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class PropertyAgent:
    """One session. Its message list is short-term memory; a new instance starts empty."""

    def __init__(
        self,
        llm,
        mcp_client,
        knowledge,
        memory,
        user_id: str,
        approve: Callable[[str, dict], bool],
        mcp_tools: list[dict],
    ):
        self.llm = llm
        self.mcp = mcp_client
        self.knowledge = knowledge
        self.memory = memory
        self.user_id = user_id
        self.approve = approve
        self.messages: list = []
        self.retrieved: set[str] = set()  # Every citation retrieved this session.
        self.on_tool: Callable[[dict], None] | None = None  # Live trace for the web app.
        self.on_text: Callable[[str], None] | None = None  # Live answer text for the web app.
        self.language = "en"  # Reply language; "ur" for Urdu.
        self.mcp_tool_names = {t["name"] for t in mcp_tools}
        self.tools = sorted(LOCAL_TOOLS + mcp_tools, key=lambda t: t["name"])
        # Long-term memory is read once per session and kept fixed, so the prompt prefix stays
        # stable. Items saved during the session are already visible in the conversation.
        self.loaded_memories = memory.load(user_id)
        self.system = [
            {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": memory_block(self.loaded_memories)},
        ]

    async def chat(self, user_text: str) -> TurnResult:
        turn_start = len(self.messages)
        self.messages.append({"role": "user", "content": user_text})
        result = TurnResult(text="")

        for _ in range(MAX_STEPS):
            try:
                response = await self._call_llm(
                    model=MODEL,
                    max_tokens=16000,
                    system=self.system + (
                        [{"type": "text", "text": LANGUAGE_NOTES[self.language]}]
                        if self.language in LANGUAGE_NOTES else []),
                    tools=self.tools,
                    messages=self.messages,
                    output_config={"effort": "medium"},
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                )
            except LLMServiceError as e:
                return self._fail(result, turn_start, str(e))
            except anthropic.AuthenticationError:
                return self._fail(result, turn_start, "The API key was rejected. Check .env.")
            except anthropic.RateLimitError:
                return self._fail(result, turn_start, "Rate limited. Please retry shortly.")
            except anthropic.APIStatusError as e:
                return self._fail(result, turn_start, f"LLM service error {e.status_code}.")
            except anthropic.APIConnectionError:
                return self._fail(result, turn_start, "Could not reach the LLM service.")

            if response.stop_reason == "refusal":
                return self._fail(result, turn_start, "The model declined this request.")
            if response.stop_reason == "max_tokens":
                return self._fail(result, turn_start, "The response was too long to finish.")

            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "pause_turn":
                continue
            if response.stop_reason == "tool_use":
                calls = [b for b in response.content if b.type == "tool_use"]
                outputs = [await self._run_tool(c, result) for c in calls]
                self.messages.append({"role": "user", "content": outputs})
                continue

            text = "\n".join(b.text for b in response.content if b.type == "text").strip()
            result.text, result.grounding = self._check_grounding(text)
            return result

        return self._fail(result, turn_start, "Stopped after too many tool steps.")

    async def _call_llm(self, **request):
        """One model call. Streams text to on_text when a listener is set and the provider can
        stream; the returned response has the same shape either way."""
        if self.on_text and getattr(self.llm, "streams_text", False):  # Groq adapter.
            return await self.llm.stream_create(on_text=self.on_text, **request)
        if self.on_text and isinstance(self.llm, anthropic.AsyncAnthropic):
            async with self.llm.beta.messages.stream(**request) as stream:
                async for chunk in stream.text_stream:
                    self.on_text(chunk)
                return await stream.get_final_message()
        return await self.llm.beta.messages.create(**request)

    def _check_grounding(self, text: str) -> tuple[str, dict]:
        """Keep only citations to passages that were actually retrieved in this session.

        A citation the model made up, or remembered from training, has no retrieved passage
        behind it, so the sentence that carries it is removed along with the citation and the
        removal is reported. Every citation the buyer sees can then be opened and checked."""
        cited = list(dict.fromkeys(CITATION.findall(text)))
        unsupported = [c for c in cited if c not in self.retrieved]
        if unsupported:
            bad = tuple(f"[{c}]" for c in unsupported)
            lines = []
            for line in text.split("\n"):
                # Sentence ends are punctuation then whitespace, so ".md#2]" never splits.
                kept = [s for s in re.split(r"(?<=[.!?])\s+", line) if not any(b in s for b in bad)]
                if kept or not line.strip():
                    lines.append(" ".join(kept))
            text = "\n".join(lines)
        return text.strip(), {
            "cited": [c for c in cited if c in self.retrieved],
            "unsupported_removed": unsupported,
        }

    def _fail(self, result: TurnResult, turn_start: int, message: str) -> TurnResult:
        # Drop the unfinished turn so the next request starts from valid history.
        del self.messages[turn_start:]
        result.text = message
        result.ok = False
        return result

    async def _run_tool(self, call, result: TurnResult) -> dict:
        name, args = call.name, dict(call.input)
        via = "mcp" if name in self.mcp_tool_names else "local"
        try:
            approved = True
            if name in GATED_TOOLS:
                approved = self.approve(name, args)
                if inspect.isawaitable(approved):  # The web app waits for a browser click.
                    approved = await approved
            if not approved:
                via = "approval"  # The server was never called, so this is not an MCP call.
                text, is_error = json.dumps({"declined": "The buyer did not approve this."}), False
            elif via == "mcp":
                text, is_error = await self.mcp.call_tool(name, args)
            else:
                text, is_error = self._run_local(name, args, result), False
        except Exception as e:  # A tool failure goes back to the model, which must report it.
            text, is_error = f"{type(e).__name__}: {e}", True

        step = {"tool": name, "via": via, "input": args, "is_error": is_error, "output": text}
        result.trace.append(step)
        if self.on_tool:
            self.on_tool(step)
        return {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": text,
            "is_error": is_error,
        }

    def _run_local(self, name: str, args: dict, result: TurnResult) -> str:
        if name in ("retrieve_knowledge", "area_guide"):
            if name == "area_guide":
                area = args["area"].strip()
                found = self.knowledge.retrieve(
                    f"{area} neighbourhood commute schools amenities flooding", k=8)
                passages = [p for p in found if area.lower() in p["text"].lower()][:4]
            else:
                passages = self.knowledge.retrieve(args["query"])
            for p in passages:
                self.retrieved.add(p["citation"])
                if p["citation"] not in result.citations:
                    result.citations.append(p["citation"])
            if not passages:
                return json.dumps({"passages": [], "note": (
                    "No relevant passage found. Tell the buyer the documents do not cover this "
                    "and do not answer from general knowledge.")})
            return json.dumps({"passages": passages})
        if name == "save_preference":
            self.memory.save(self.user_id, args["key"], args["value"], args["category"])
            return json.dumps({"saved": args["key"]})
        if name == "forget_preference":
            return json.dumps({"removed": self.memory.forget(self.user_id, args["key"])})
        raise ValueError(f"Unknown tool {name}")
