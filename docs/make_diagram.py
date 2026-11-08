"""Draws docs/architecture.png. Run: python docs/make_diagram.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1f2933"
AGENT = "#dbeafe"
MEMORY = "#fef3c7"
RAG = "#dcfce7"
MCP = "#ede9fe"
HUMAN = "#fee2e2"
LLM = "#e0f2fe"


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                       facecolor=color, edgecolor=INK, linewidth=1.2)
    )
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=11,
            fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.56, body, ha="center", va="top", fontsize=8.5, color=INK,
            linespacing=1.35)


def arrow(ax, start, end, label="", offset=(0, 0.12), both=False):
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="<|-|>" if both else "-|>", mutation_scale=13,
                        color=INK, linewidth=1.2, shrinkA=0, shrinkB=0)
    )
    if label:
        ax.text((start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1], label,
                ha="center", va="bottom", fontsize=8, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=0.9))


fig, ax = plt.subplots(figsize=(13, 8.2))
ax.set_xlim(0, 13)
ax.set_ylim(0, 8.2)
ax.axis("off")
ax.text(6.5, 7.95, "Deedly, Real Estate Property Advisory Agent: Architecture", ha="center",
        fontsize=14, fontweight="bold", color=INK)

box(ax, 0.3, 5.3, 2.5, 1.7, "Buyer (human)", "Asks questions\nApproves viewing\nrequests", HUMAN)
box(ax, 4.0, 4.7, 4.6, 2.5, "Python AI Agent (agent.py)",
    "Manual tool loop with decision trace\n\nSystem prompt: role, scope, tool rules,\n"
    "evidence, memory, re-planning, failure\n\nApproval gate on request_viewing", AGENT)
box(ax, 9.8, 5.3, 2.9, 1.7, "LLM", "Claude API\nclaude-opus-5-5\nChooses tools, writes answer", LLM)

box(ax, 0.3, 2.5, 2.8, 1.6, "Short-term memory",
    "In-session message list\nFollow-ups like \"that one\"\nEmpty in a new session", MEMORY)
box(ax, 0.3, 0.4, 2.8, 1.6, "Long-term memory",
    "SQLite memory.db\nBudget, areas, shortlist\nLoaded in each new session", MEMORY)

box(ax, 3.9, 0.4, 3.5, 3.1, "RAG pipeline (rag.py)",
    "6 Markdown documents\n\nChunk by section and size\n\nEmbed: bge-small-en-v1.5\n\n"
    "Store and search: Chroma\n\nPassages with citations,\nempty if below relevance floor", RAG)

box(ax, 8.2, 2.1, 2.2, 1.4, "MCP client", "mcp_client.py\nClientSession\nover stdio", MCP)
box(ax, 8.2, 0.4, 2.2, 1.2, "MCP server", "mcp_server.py\nFastMCP subprocess", MCP)
box(ax, 10.8, 0.4, 2.0, 3.1, "MCP tools",
    "search_properties\n\nget_property\n\ncompare_properties\n\nestimate_monthly_\npayment\n\n"
    "request_viewing\n(simulated)", MCP)

arrow(ax, (2.8, 6.15), (4.0, 6.15), "chat", both=True)
arrow(ax, (8.6, 6.15), (9.8, 6.15), "tools, history", both=True)
# Short-term memory: up from the box, then right into the agent.
ax.plot([1.7, 1.7], [4.3, 4.9], color=INK, linewidth=1.2)
arrow(ax, (1.7, 4.4), (1.7, 4.1))
arrow(ax, (1.7, 4.9), (4.0, 4.9), "every turn")
# Long-term memory: along the gutter between the memory boxes and the RAG box.
ax.plot([3.5, 3.5, 4.0], [1.2, 4.4, 4.4], color=INK, linewidth=1.2)
arrow(ax, (3.5, 1.2), (3.1, 1.2))
arrow(ax, (4.0, 4.4), (4.3, 4.7))
ax.text(3.5, 2.8, "save / forget / load", fontsize=8, color=INK, ha="center", va="center",
        rotation=90, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
arrow(ax, (5.65, 4.7), (5.65, 3.5), "retrieve_knowledge", offset=(0, -0.05), both=True)
arrow(ax, (8.0, 4.7), (9.1, 3.5), "tool calls", offset=(0.45, 0), both=True)
arrow(ax, (9.3, 2.1), (9.3, 1.6), "", both=True)
arrow(ax, (10.4, 1.0), (10.8, 1.0), "", both=True)

out = Path(__file__).with_name("architecture.png")
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
print(out)
