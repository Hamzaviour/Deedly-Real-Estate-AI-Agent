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
from types import SimpleNamespace as NS

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
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


class GroqLLM:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        from openai import AsyncOpenAI

        self.model = model or os.getenv("GROQ_MODEL", GROQ_DEFAULT_MODEL)
        self._client = AsyncOpenAI(api_key=api_key or os.getenv("GROQ_API_KEY"), base_url=GROQ_BASE_URL)
        self.beta = NS(messages=NS(create=self._create))

    async def _create(self, *, system, tools, messages, max_tokens, **_ignored):
        import openai

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                max_tokens=min(max_tokens, GROQ_MAX_TOKENS),
                messages=to_openai_messages(system, messages),
                tools=[to_openai_tool(t) for t in tools],
                tool_choice="auto",
                parallel_tool_calls=False,
            )
        except openai.AuthenticationError:
            raise LLMServiceError("The Groq API key was rejected. Check GROQ_API_KEY in .env.")
        except openai.RateLimitError:
            raise LLMServiceError("Groq rate limit reached. Please retry shortly.")
        except openai.APIStatusError as e:
            raise LLMServiceError(f"Groq service error {e.status_code}. Please retry.")
        except openai.APIConnectionError:
            raise LLMServiceError("Could not reach Groq. Check the internet connection.")
        return from_openai_response(response)


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
