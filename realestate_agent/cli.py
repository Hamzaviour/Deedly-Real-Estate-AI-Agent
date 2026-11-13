"""Interactive chat. Run: python -m realestate_agent.cli --user hamza

Commands: /quit to exit. /sold P-103 (or /status P-103 under_offer|available) changes a
listing outside the agent, to demonstrate re-planning on the next question."""
import argparse
import asyncio
import json
import sys



from .agent import PropertyAgent
from .config import require_api_key
from .llm import make_llm
from .mcp_client import ListingsMCPClient
from .memory import make_memory
from . import properties
from .rag import make_knowledge


def print_trace(result) -> None:
    for step in result.trace:
        status = "ERROR" if step["is_error"] else "ok"
        print(f"  [{step['via']}] {step['tool']}({json.dumps(step['input'])}) -> {status}")
    if result.citations:
        print(f"  [retrieved] {', '.join(dict.fromkeys(result.citations))}")


def ask_approval(tool: str, args: dict) -> bool:
    answer = input(f"\nApprove {tool} {json.dumps(args)}? [y/N] ")
    return answer.strip().lower() == "y"


async def main(user_id: str) -> None:
    require_api_key()
    llm = make_llm()
    knowledge = make_knowledge()
    memory = make_memory()
    async with ListingsMCPClient() as mcp:
        agent = PropertyAgent(
            llm, mcp, knowledge, memory, user_id, ask_approval, await mcp.list_tools()
        )
        print(f"Session for '{user_id}'. Saved memories: {len(agent.loaded_memories)}. "
              "Type /quit to exit, /sold P-103 to simulate a sale.")
        while True:
            text = input("\nYou: ").strip()
            if text == "/quit":
                break
            if text.startswith(("/sold", "/status")):
                parts = text.split()
                status = "sold" if parts[0] == "/sold" else (parts[2] if len(parts) > 2 else "")
                if len(parts) < 2 or status not in ("available", "under_offer", "sold"):
                    print("Usage: /sold P-103  or  /status P-103 available|under_offer|sold")
                    continue
                try:
                    properties.set_status(parts[1].upper(), status)
                    print(f"  [change event] {parts[1].upper()} is now {status}. Ask about it.")
                except ValueError as e:
                    print(f"  {e}")
                continue
            if text:
                result = await agent.chat(text)
                print_trace(result)
                print(f"\nAdvisor: {result.text}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="demo_buyer")
    asyncio.run(main(parser.parse_args().user))
