"""Interactive chat. Run: python -m realestate_agent.cli --user hamza"""
import argparse
import asyncio
import json
import sys



from .agent import PropertyAgent
from .config import require_api_key
from .llm import make_llm
from .mcp_client import ListingsMCPClient
from .memory import LongTermMemory
from .rag import KnowledgeBase


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
    knowledge = KnowledgeBase()
    memory = LongTermMemory()
    async with ListingsMCPClient() as mcp:
        agent = PropertyAgent(
            llm, mcp, knowledge, memory, user_id, ask_approval, await mcp.list_tools()
        )
        print(f"Session for '{user_id}'. Saved memories: {len(agent.loaded_memories)}. "
              "Type /quit to exit.")
        while True:
            text = input("\nYou: ").strip()
            if text == "/quit":
                break
            if text:
                result = await agent.chat(text)
                print_trace(result)
                print(f"\nAdvisor: {result.text}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="demo_buyer")
    asyncio.run(main(parser.parse_args().user))
