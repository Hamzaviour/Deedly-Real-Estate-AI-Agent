"""Evaluation harness. Writes evidence/eval_report.md and evidence/eval_results.json.

Two parts:

1. Retrieval (offline, no API key): for each question in retrieval_cases.json, does the
   retriever return the expected document (hit@1, hit@4, MRR), and does it return nothing for
   questions the documents do not cover (off-topic rejection)?
2. Agent behaviour (needs ANTHROPIC_API_KEY or GROQ_API_KEY): each scenario in
   agent_cases.json runs against the real model, real MCP server and real retriever in a fresh
   runtime, and is scored by automatic checks on tool calls, citations, answer text and saved
   memory.

Run: python -m evals.run_evals            (both parts; agent part skipped without a key)
     python -m evals.run_evals --retrieval-only
     python -m evals.run_evals --case rag-deposit --case replan-sold
"""
import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RE_AGENT_RUNTIME", str(Path(tempfile.gettempdir()) / "deedly-evals"))

from realestate_agent import properties  # noqa: E402
from realestate_agent.agent import PropertyAgent  # noqa: E402
from realestate_agent.config import MODEL  # noqa: E402
from realestate_agent.llm import make_llm, model_name, provider  # noqa: E402
from realestate_agent.mcp_client import ListingsMCPClient  # noqa: E402
from realestate_agent.memory import LongTermMemory  # noqa: E402
from realestate_agent.rag import (HybridKnowledgeBase, KnowledgeBase,  # noqa: E402
                                  LexicalKnowledgeBase, make_knowledge)

HERE = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
USER = "eval_buyer"


# ---------- Retrieval ----------

def run_retrieval(knowledge) -> dict:
    cases = json.loads((HERE / "retrieval_cases.json").read_text(encoding="utf-8"))
    rows, hit1, hit4, rr, covered, off_ok, off_total = [], 0, 0, 0.0, 0, 0, 0
    for case in cases:
        hits = knowledge.retrieve(case["q"], k=4)
        sources = [h["source"] for h in hits]
        expected = case["doc"]
        if expected is None:
            off_total += 1
            ok = not hits
            off_ok += ok
            rows.append({"q": case["q"], "expected": None, "got": sources, "pass": ok})
            continue
        expected = [expected] if isinstance(expected, str) else expected
        covered += 1
        rank = next((i + 1 for i, s in enumerate(sources) if s in expected), None)
        hit1 += rank == 1
        hit4 += rank is not None
        rr += 1 / rank if rank else 0
        rows.append({"q": case["q"], "expected": expected, "got": sources, "rank": rank,
                     "pass": rank == 1})
    return {
        "retriever": type(knowledge).__name__,
        "covered_questions": covered,
        "off_topic_questions": off_total,
        "hit_at_1": round(hit1 / covered, 3),
        "hit_at_4": round(hit4 / covered, 3),
        "mrr": round(rr / covered, 3),
        "off_topic_rejection": round(off_ok / off_total, 3) if off_total else None,
        "rows": rows,
    }


# ---------- Agent behaviour ----------

def _match(patterns: list[str], text: str) -> list[str]:
    """Patterns that do NOT match the text."""
    return [p for p in patterns if not re.search(p, text, re.I)]


def score(case: dict, turns: list, memories: list[dict], approvals: list) -> list[str]:
    """Return the list of failed checks. Tool checks cover every turn; text checks the last."""
    c = case["checks"]
    called = {s["tool"] for t in turns for s in t.trace}
    last = turns[-1]
    memory_text = " ".join(f"{m['key']} {m['value']}" for m in memories)
    failures = []
    if not last.ok:
        failures.append(f"turn failed: {last.text}")
    if c.get("tools_any") and not called & set(c["tools_any"]):
        failures.append(f"none of {c['tools_any']} called (called {sorted(called)})")
    for t in c.get("tools_all", []):
        if t not in called:
            failures.append(f"{t} not called")
    for t in c.get("tools_none", []):
        if t in called:
            failures.append(f"{t} should not be called")
    if c.get("must_cite") and not last.grounding.get("cited"):
        failures.append("no verified citation in the answer")
    removed = [g for t in turns for g in t.grounding.get("unsupported_removed", [])]
    if removed:
        failures.append(f"cited passages it never retrieved: {removed}")
    failures += [f"answer lacks /{p}/" for p in _match(c.get("includes", []), last.text)]
    failures += [f"answer contains /{p}/" for p in c.get("excludes", [])
                 if re.search(p, last.text, re.I)]
    failures += [f"memory lacks /{p}/" for p in _match(c.get("memory_includes", []), memory_text)]
    failures += [f"memory contains /{p}/" for p in c.get("memory_excludes", [])
                 if re.search(p, memory_text, re.I)]
    if "approval_requested" in c and bool(approvals) != c["approval_requested"]:
        failures.append("approval was not requested" if c["approval_requested"]
                        else "approval was requested")
    return failures


async def run_case(case: dict, llm, knowledge, pause: float) -> dict:
    runtime = Path(tempfile.mkdtemp(prefix=f"eval-{case['id']}-"))
    os.environ["RE_AGENT_RUNTIME"] = str(runtime)  # Fresh listings, statuses and memory.
    approvals: list = []

    def approve(tool: str, args: dict) -> bool:
        approvals.append({"tool": tool, "input": args})
        return case.get("approve", True)

    started = time.monotonic()
    try:
        memory = LongTermMemory()
        async with ListingsMCPClient() as mcp:
            tools = await mcp.list_tools()
            agent = PropertyAgent(llm, mcp, knowledge, memory, USER, approve, tools)
            turns = []
            for turn in case["turns"]:
                for pid, status in turn.get("before", {}).get("status", {}).items():
                    properties.set_status(pid, status)
                turns.append(await agent.chat(turn["say"]))
                if pause:
                    await asyncio.sleep(pause)
        memories = memory.load(USER)
        failures = score(case, turns, memories, approvals)
    except Exception as e:  # A crash is a failed case, not a crashed run.
        turns, memories, failures = [], [], [f"crashed: {type(e).__name__}: {e}"]
    finally:
        shutil.rmtree(runtime, ignore_errors=True)
    return {
        "id": case["id"],
        "category": case["category"],
        "pass": not failures,
        "failures": failures,
        "seconds": round(time.monotonic() - started, 1),
        "tools": [s["tool"] for t in turns for s in t.trace],
        "cited": [c for t in turns for c in t.grounding.get("cited", [])],
        "answer": turns[-1].text if turns else "",
    }


async def run_agent(knowledge, only: list[str], pause: float) -> dict | None:
    if provider() is None:
        return None
    llm = make_llm()
    cases = json.loads((HERE / "agent_cases.json").read_text(encoding="utf-8"))
    if only:
        cases = [c for c in cases if c["id"] in only]
    results = []
    for case in cases:
        r = await run_case(case, llm, knowledge, pause)
        print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['id']}" +
              ("" if r["pass"] else f"  <- {'; '.join(r['failures'])}"))
        results.append(r)
    by_cat: dict[str, list[bool]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r["pass"])
    return {
        "provider": provider(),
        "model": model_name(llm, MODEL),
        "passed": sum(r["pass"] for r in results),
        "total": len(results),
        "by_category": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_cat.items())},
        "cases": results,
    }


# ---------- Report ----------

def compare_retrievers() -> list[dict]:
    """Score each retriever on the same questions, to show what the hybrid adds."""
    rows = []
    for cls in (LexicalKnowledgeBase, KnowledgeBase, HybridKnowledgeBase):
        try:
            r = run_retrieval(cls())
        except ImportError:
            continue
        rows.append({k: r[k] for k in ("retriever", "hit_at_1", "hit_at_4", "mrr",
                                        "off_topic_rejection")})
    return rows


def write_report(retrieval: dict, agent: dict | None, comparison: list[dict]) -> None:
    EVIDENCE.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out = [f"# Evaluation report\n\nGenerated {now} by `python -m evals.run_evals`. "
           "All listings and documents are synthetic.\n",
           "## Retrieval (RAG)\n",
           f"Retriever: `{retrieval['retriever']}`. {retrieval['covered_questions']} questions the "
           f"documents cover, {retrieval['off_topic_questions']} they do not.\n",
           "| Metric | Score |", "|---|---|",
           f"| Hit@1 (expected document ranked first) | {retrieval['hit_at_1']:.0%} |",
           f"| Hit@4 (expected document in top 4) | {retrieval['hit_at_4']:.0%} |",
           f"| Mean reciprocal rank | {retrieval['mrr']:.3f} |",
           f"| Off-topic questions correctly given no passages | {retrieval['off_topic_rejection']:.0%} |",
           ""]
    if comparison:
        out += ["Same questions, each retriever on its own:\n",
                "| Retriever | Hit@1 | Hit@4 | MRR | Off-topic rejection |", "|---|---|---|---|---|"]
        out += [f"| {c['retriever']} | {c['hit_at_1']:.0%} | {c['hit_at_4']:.0%} | {c['mrr']:.3f} | "
                f"{c['off_topic_rejection']:.0%} |" for c in comparison]
        out += ["", "LexicalKnowledgeBase is the hosted (serverless) fallback. It matches words, so a "
                "question that shares one word with the documents can retrieve a passage; the agent's "
                "instructions then require it to say the passages do not answer the question.\n"]
    misses = [r for r in retrieval["rows"] if not r["pass"]]
    if misses:
        out += ["Misses:\n"] + [f"- \"{r['q']}\": expected {r['expected']}, got {r['got']}"
                                for r in misses] + [""]
    out.append("## Agent behaviour\n")
    if agent is None:
        out.append("Skipped: no ANTHROPIC_API_KEY or GROQ_API_KEY set.\n")
    else:
        out += [f"Model: `{agent['model']}` via {agent['provider']}. "
                f"**{agent['passed']}/{agent['total']} scenarios passed.**\n",
                "| Category | Passed |", "|---|---|"]
        out += [f"| {k} | {v} |" for k, v in agent["by_category"].items()]
        out += ["", "| Scenario | Result | Tools called | Notes |", "|---|---|---|---|"]
        for r in agent["cases"]:
            notes = "; ".join(r["failures"]).replace("|", "\\|") or (
                f"cited {', '.join(dict.fromkeys(r['cited']))}" if r["cited"] else "")
            out.append(f"| {r['id']} | {'pass' if r['pass'] else '**fail**'} | "
                       f"{', '.join(dict.fromkeys(r['tools']))} | {notes} |")
        out.append("")
    (EVIDENCE / "eval_report.md").write_text("\n".join(out), encoding="utf-8")
    (EVIDENCE / "eval_results.json").write_text(
        json.dumps({"generated": now, "retrieval": retrieval, "agent": agent}, indent=1),
        encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--case", action="append", default=[], help="Run only these scenario ids")
    parser.add_argument("--pause", type=float, default=0.0,
                        help="Seconds to wait between turns, for rate-limited keys")
    parser.add_argument("--compare-retrievers", action="store_true",
                        help="Also score lexical, embedding and hybrid retrievers separately")
    parser.add_argument("--min-hit1", type=float, default=0.85,
                        help="Exit non-zero if retrieval hit@1 falls below this")
    args = parser.parse_args()

    knowledge = make_knowledge()
    retrieval = run_retrieval(knowledge)
    print(f"Retrieval ({retrieval['retriever']}): hit@1 {retrieval['hit_at_1']:.0%}, "
          f"hit@4 {retrieval['hit_at_4']:.0%}, MRR {retrieval['mrr']}, "
          f"off-topic rejection {retrieval['off_topic_rejection']:.0%}")
    agent = None
    if not args.retrieval_only:
        print("Agent scenarios:")
        agent = asyncio.run(run_agent(knowledge, args.case, args.pause))
        if agent is None:
            print("  skipped: no API key")
        else:
            print(f"Agent: {agent['passed']}/{agent['total']} passed")
    comparison = compare_retrievers() if args.compare_retrievers else []
    write_report(retrieval, agent, comparison)
    print(f"Saved {EVIDENCE / 'eval_report.md'}")
    ok = retrieval["hit_at_1"] >= args.min_hit1 and retrieval["off_topic_rejection"] == 1.0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
