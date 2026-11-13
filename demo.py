"""Reproducible end-to-end demo with the real LLM. Needs ANTHROPIC_API_KEY or GROQ_API_KEY in .env.

Covers every required component and writes evidence/demo_transcript.md. Run: python demo.py
"""
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMO_RUNTIME = ROOT / "runtime" / "demo"
shutil.rmtree(DEMO_RUNTIME, ignore_errors=True)
os.environ["RE_AGENT_RUNTIME"] = str(DEMO_RUNTIME)



from realestate_agent import properties  # noqa: E402
from realestate_agent.agent import PropertyAgent  # noqa: E402
from realestate_agent.config import MODEL, require_api_key  # noqa: E402
from realestate_agent.llm import make_llm  # noqa: E402
from realestate_agent.mcp_client import ListingsMCPClient  # noqa: E402
from realestate_agent.memory import make_memory  # noqa: E402
from realestate_agent.rag import make_knowledge  # noqa: E402

USER = "demo_buyer"
OUT = ROOT / "evidence" / "demo_transcript.md"
lines: list[str] = []


def log(text: str = "") -> None:
    print(text)
    lines.append(text)


async def say(agent: PropertyAgent, label: str, text: str) -> None:
    log(f"\n### {label}\n\n**Buyer:** {text}\n")
    result = await agent.chat(text)
    for step in result.trace:
        log(f"- Tool `{step['tool']}` via {step['via']}, input `{json.dumps(step['input'])}`, "
            f"error: {step['is_error']}")
        log(f"  - Output: `{step['output'][:300]}`")
    if result.citations:
        log(f"- Retrieved: {', '.join(dict.fromkeys(result.citations))}")
    log(f"\n**Advisor:** {result.text}")


def auto_approve(tool: str, args: dict) -> bool:
    log(f"- Human approval requested for `{tool}` `{json.dumps(args)}`. Demo answers: yes")
    return True


async def main() -> None:
    require_api_key()
    llm = make_llm()
    log(f"# Demo transcript\n\nModel: `{getattr(llm, 'model', None) or MODEL}`. "
        "All listings and documents are synthetic.")
    knowledge = make_knowledge()
    log(f"\nRAG index: {knowledge.ingest()}")
    memory = make_memory()

    async with ListingsMCPClient() as mcp:
        tools = await mcp.list_tools()
        log(f"MCP tools: {[t['name'] for t in tools]}")

        log("\n## Session 1")
        a = PropertyAgent(llm, mcp, knowledge, memory, USER, auto_approve, tools)
        await say(a, "Tool selection and long-term save",
                  "Hi, I'm Hamza. My budget is 27 million PKR, I need at least 3 bedrooms, "
                  "and I want parking. My commute to the centre must be under 20 minutes. "
                  "What fits?")
        await say(a, "Short-term memory follow-up", "Which of those is the cheaper one per square foot?")
        await say(a, "Comparison plus MCP calculation",
                  "Compare P-103 and P-110 for me and recommend one. What would the monthly "
                  "payment be, using your planning assumptions?")
        await say(a, "RAG grounding", "How much deposit should I pay, and what should a lawyer check?")

        log("\n### Change event\n\nSimulated outside the agent: P-103 is now sold.")
        recommended_gone = "P-103"
        properties.set_status(recommended_gone, "sold")
        await say(a, "Re-planning", "I've decided on P-103. Please re-check it, and tell me what to do next.")
        await say(a, "Human oversight on a consequential tool",
                  "Request a viewing of P-110 for me on 2026-11-20, under the name Hamza.")
        await say(a, "Uncertainty behaviour", "What is the capital gains tax on selling a home here?")

        log("\n## Session 2: new agent, empty conversation\n")
        log(f"Memories in database: `{json.dumps(memory.load(USER))}`")
        b = PropertyAgent(llm, mcp, knowledge, memory, USER, auto_approve, tools)
        await say(b, "Long-term memory recall", "What do you remember about what I'm looking for?")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
