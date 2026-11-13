"""LLM provider selection, plus a Groq adapter that speaks the shape the agent loop expects.

The agent loop is written against the Anthropic Messages API. GroqLLM exposes the same
`beta.messages.create(...)` call and returns the same response shape (stop_reason plus text and
tool_use blocks), translating to and from Groq's OpenAI-compatible chat completions API. That
keeps the agent, tools, memory, RAG, MCP and approval gate identical across providers.

Choose the provider with RE_AGENT_PROVIDER=anthropic|groq. If unset, Anthropic is used when its
key is present, otherwise Groq when GROQ_API_KEY is present.
"""
import json
import os
import re
from types import SimpleNamespace as NS

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"  # Replaced by discovery when retired.
GROQ_MAX_TOKENS = 8000


class LLMServiceError(Exception):
    """A provider failure, with a message that is safe to show the buyer."""


def provider() -> str | None:
    chosen = os.getenv("RE_AGENT_PROVIDER", "").strip().lower()
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
    has_groq = bool(os.getenv("GROQ_API_KEY"))
    if chosen == "groq":
        return "groq" if has_groq else None
    if chosen == "anthropic":
        return "anthropic" if has_anthropic else None
    if has_anthropic:
        return "anthropic"
    if has_groq:
        return "groq"
    return None


def make_llm():
    """Return a client for the configured provider, or None when no key is set."""
    which = provider()
    if which == "anthropic":
        import anthropic

        return anthropic.AsyncAnthropic()
    if which == "groq":
        return GroqLLM()
    return None


def model_name(llm, default: str) -> str:
    return getattr(llm, "model", None) or default


# Tool-capable chat models in order of preference. Groq retires models over time, so the
# adapter asks the API which ones this key can use and picks the first available.
GROQ_PREFERRED_MODELS = [
    "openai/gpt-oss-120b",
    "moonshotai/kimi-k2-instruct-0905",
    "moonshotai/kimi-k2-instruct",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]
NOT_CHAT = re.compile(r"whisper|tts|guard|orpheus|playai|distil|compound|embed|vision|safeguard", re.I)


def pick_groq_model(available: list[str]) -> str | None:
    """Choose the best chat model from the ids the account can use."""
    for name in GROQ_PREFERRED_MODELS:
        if name in available:
            return name
    chat = [m for m in available if not NOT_CHAT.search(m)]
    size = lambda m: max((float(x) for x in re.findall(r"(\d+(?:\.\d+)?)b\b", m.lower())), default=0)
    return max(chat, key=size) if chat else None


class GroqLLM:
    streams_text = True

    def __init__(self, api_key: str | None = None, model: str | None = None):
        from openai import AsyncOpenAI

        explicit = model or os.getenv("GROQ_MODEL", "").strip()
        self.model = explicit or GROQ_DEFAULT_MODEL
        self._resolved = bool(explicit)  # An explicitly chosen model is used as given.
        self._client = AsyncOpenAI(api_key=api_key or os.getenv("GROQ_API_KEY"), base_url=GROQ_BASE_URL)
        self.beta = NS(messages=NS(create=self._create))

    async def _resolve_model(self) -> None:
        """Ask Groq which models exist, so a retired default never breaks the advisor."""
        self._resolved = True
        try:
            listing = await self._client.models.list()
            available = [m.id for m in listing.data]
        except Exception:
            return  # Keep the current model; the chat call reports any real problem.
        chosen = pick_groq_model(available)
        if chosen:
            self.model = chosen

    async def stream_create(self, *, on_text, **kwargs):
        """Like create, but calls on_text with each piece of answer text as it arrives."""
        return await self._create(on_text=on_text, **kwargs)

    async def _create(self, *, system, tools, messages, max_tokens, on_text=None, **_ignored):
        import openai

        if not self._resolved:
            await self._resolve_model()
        request = dict(
            max_tokens=min(max_tokens, GROQ_MAX_TOKENS),
            messages=to_openai_messages(system, messages),
            tools=[to_openai_tool(t) for t in tools],
            tool_choice="auto",
            parallel_tool_calls=False,
        )
        for attempt in range(2):
            try:
                if on_text is None:
                    response = await self._client.chat.completions.create(model=self.model, **request)
                    return from_openai_response(response)
                stream = await self._client.chat.completions.create(
                    model=self.model, stream=True, **request)
                return await collect_openai_stream(stream, on_text)
            except openai.AuthenticationError:
                raise LLMServiceError("The Groq API key was rejected. Check GROQ_API_KEY.")
            except openai.RateLimitError:
                raise LLMServiceError("Groq rate limit reached. Please retry shortly.")
            except openai.NotFoundError:
                if attempt == 0:  # Model retired or not enabled for this key: rediscover once.
                    old = self.model
                    await self._resolve_model()
                    if self.model != old:
                        continue
                raise LLMServiceError(
                    f"Groq model '{self.model}' is not available for this key. "
                    "Set GROQ_MODEL to a model listed at console.groq.com/docs/models.")
            except openai.BadRequestError as e:
                # Groq rejects a malformed tool call with 400 tool_use_failed. One retry
                # usually succeeds because generation is sampled.
                if attempt == 0 and "tool_use_failed" in str(e):
                    continue
                raise LLMServiceError(f"Groq rejected the request: {_brief(e)}")
            except openai.APIStatusError as e:
                raise LLMServiceError(f"Groq service error {e.status_code}: {_brief(e)}")
            except openai.APIConnectionError:
                raise LLMServiceError("Could not reach Groq. Check the internet connection.")
        raise LLMServiceError("Groq could not produce a valid tool call. Please retry.")


def _brief(error) -> str:
    body = getattr(error, "body", None)
    message = body.get("message") if isinstance(body, dict) else None
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        message = body["error"].get("message", message)
    return (message or str(error))[:200]


def to_openai_tool(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool["input_schema"],
        },
    }


def _field(block, name):
    return block.get(name) if isinstance(block, dict) else getattr(block, name, None)


def to_openai_messages(system, messages: list) -> list[dict]:
    """Translate Anthropic-format history into OpenAI chat messages."""
    system_text = system if isinstance(system, str) else "\n\n".join(
        _field(b, "text") for b in system if _field(b, "text"))
    out = [{"role": "system", "content": system_text}]
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        if m["role"] == "assistant":
            texts, calls = [], []
            for b in content:
                kind = _field(b, "type")
                if kind == "text" and _field(b, "text"):
                    texts.append(_field(b, "text"))
                elif kind == "tool_use":
                    calls.append({
                        "id": _field(b, "id"),
                        "type": "function",
                        "function": {"name": _field(b, "name"),
                                     "arguments": json.dumps(dict(_field(b, "input") or {}))},
                    })
            msg = {"role": "assistant", "content": "\n".join(texts) or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:
            for b in content:
                if _field(b, "type") == "tool_result":
                    text = _field(b, "content")
                    if _field(b, "is_error"):
                        text = f"ERROR: {text}"
                    out.append({"role": "tool", "tool_call_id": _field(b, "tool_use_id"), "content": text})
                elif _field(b, "type") == "text":
                    out.append({"role": "user", "content": _field(b, "text")})
    return out


async def collect_openai_stream(stream, on_text):
    """Rebuild a full completion from streamed chunks, forwarding text as it arrives."""
    text, calls, finish = [], {}, None
    async for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        if delta.content:
            text.append(delta.content)
            on_text(delta.content)
        for tc in delta.tool_calls or []:
            slot = calls.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
            if tc.id:
                slot["id"] = tc.id
            if tc.function and tc.function.name:
                slot["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                slot["arguments"] += tc.function.arguments
        finish = choice.finish_reason or finish
    message = NS(content="".join(text) or None, tool_calls=[
        NS(id=c["id"], function=NS(name=c["name"], arguments=c["arguments"]))
        for _, c in sorted(calls.items())] or None)
    return from_openai_response(NS(choices=[NS(message=message, finish_reason=finish)]))


def from_openai_response(response):
    """Translate an OpenAI chat completion into the Anthropic response shape."""
    choice = response.choices[0]
    message = choice.message
    blocks = []
    if message.content:
        blocks.append(NS(type="text", text=message.content))
    for call in message.tool_calls or []:
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        blocks.append(NS(type="tool_use", id=call.id, name=call.function.name,
                         input=args if isinstance(args, dict) else {}))
    if any(b.type == "tool_use" for b in blocks):
        stop = "tool_use"
    elif choice.finish_reason == "length":
        stop = "max_tokens"
    else:
        stop = "end_turn"
    return NS(stop_reason=stop, content=blocks)
