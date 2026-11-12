<div align="center">

<img src="realestate_agent/static/brand/deedly-mark.svg" width="72" alt="Deedly logo">

# Deedly

**An AI property advisor for Lahore that searches homes, remembers buyers, compares options and cites its sources.**

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Starlette](https://img.shields.io/badge/Starlette-ASGI-1A1D21)
![MCP](https://img.shields.io/badge/MCP-stdio%20server-17403C)
![RAG](https://img.shields.io/badge/RAG-Chroma%20%2B%20fastembed-2E8A5C)
![LLM](https://img.shields.io/badge/LLM-Groq%20%7C%20Anthropic-B4472C)
![Tests](https://img.shields.io/badge/tests-68%20passing-2E8A5C)

**[🌐 Try the live demo](https://deedly-property-advisor.vercel.app/)**

Assignment 2, Agent Based Modeling (AIDE3313), scenario 38: *retrieve property details, remember buyer preferences and compare options.*

</div>

> All listings, prices and documents are synthetic. The areas are real parts of Lahore, but no home here is for sale. Viewing requests are simulated. Deedly gives general information, not legal or financial advice.

---

## Contents

- [Overview](#overview)
- [Screenshots](#screenshots)
- [Features](#features)
- [How it works](#how-it-works)
- [Assignment requirements](#assignment-requirements)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [HTTP API](#http-api)
- [MCP tools](#mcp-tools)
- [Testing](#testing)
- [Deployment](#deployment)
- [Data and credits](#data-and-credits)
- [Limitations and roadmap](#limitations-and-roadmap)
- [Author](#author)

---

## Overview

Deedly is a property portal with an AI agent built in. A buyer can search and filter homes across Johar Town, Allama Iqbal Town, Park View City, Gulberg and Model Town like on any listings site, then open the advisor, **Dee**, and ask in plain language:

> *"My budget is 2.7 crore. I need at least 3 bedrooms, parking and a commute under 20 minutes. What fits?"*

Behind that answer the agent:

1. saves the budget to **long-term memory** (SQLite),
2. searches live listings through a real **MCP server**,
3. reads the buying guides through **RAG** and cites them,
4. keeps the conversation as **short-term memory** for follow-ups like *"which of those is cheaper per square foot?"*,
5. **re-plans** if a home is sold mid-conversation,
6. and stops for **human approval** before sending any viewing request.

Every one of those steps is shown on screen as it happens, so the agent's work can be checked rather than taken on trust.

---

## Screenshots

| Home and search | Browse by area |
|---|---|
| ![Home page with hero search](docs/screenshots/01-home.png) | ![Area cards](docs/screenshots/02-areas.png) |

| Listings with shortlist and compare | Side-by-side comparison |
|---|---|
| ![Listing grid with compare tray](docs/screenshots/03-listings.png) | ![Compare dialog](docs/screenshots/04-compare.png) |

| Home sheet with mortgage calculator | Advisor with Dee |
|---|---|
| ![Home detail sheet](docs/screenshots/05-home-sheet.png) | ![Advisor drawer](docs/screenshots/06-advisor.png) |

---

## Features

### For the buyer

- **Portal-style search.** Type tabs (houses, apartments, townhouses), location, max price, bedrooms and keyword, plus quick area chips.
- **Browse by area.** Five Lahore neighbourhoods with home counts and starting prices.
- **Listing grid.** Sort by price, value per square foot, newest or shortest commute. Filter to available homes only. Every active filter shows as a removable chip.
- **Shortlist.** A heart on every home. The shortlist is stored in the buyer's long-term memory, so the agent knows about it in later sessions.
- **Compare.** Pick two or three homes for a side-by-side table that ticks the best price, value, size, commute and estimated monthly payment, then ask Dee to recommend one.
- **Home sheet.** Photo gallery (arrow keys work), facts, features and a **mortgage calculator** with deposit, rate and term sliders. It uses the same formula as the agent's payment tool, so the numbers always agree.
- **Advisor drawer.** Chat with Dee. Homes named in an answer appear as cards and are highlighted in the grid as *Advisor picks*.

### For the marker

- **Agent bar.** One thin line across the top: LLM, short-term memory, long-term memory, RAG, MCP, approval and re-planning. A dot turns green only when that component really runs. Hovering explains what happened.
- **Live tool steps.** While Dee works, each step appears in the chat, tagged `MCP`, `RAG` or `MEM`.
- **Record dropdown.** Saved memory, raw tool input and output, and retrieved sources with similarity scores.
- **Re-planning control.** Each card has a status control. Mark a home sold and ask about it to watch the agent re-check.
- **Approval gate.** Viewing requests show exactly what will be sent and wait for Approve or Decline. Declining means the MCP server is never called.

### Dee, the live mascot

An original animated house character drawn in SVG. Dee's eyes follow the pointer, blink and bob. Dee waves on hover, thinks while a message is sent, puffs chimney smoke while tools run, puts on reading glasses while searching the guides, cheers when an answer lands and floats a heart when a home is shortlisted. Dee also gives short tips in a speech bubble. All motion stops under *reduce motion*.

---

## How it works

### Agent workflow

```mermaid
flowchart LR
    B([Buyer]) -->|message| UI[Web app<br/>Starlette]
    UI --> A{{PropertyAgent<br/>agent loop}}
    A <-->|messages + tools| L[(LLM<br/>Groq or Anthropic)]
    A -->|listing tools| C[MCP client] <-->|stdio| S[MCP server<br/>FastMCP]
    S <--> P[(Listings<br/>SQLite)]
    A -->|retrieve_knowledge| R[RAG<br/>fastembed + Chroma]
    R <--> K[/6 Markdown guides/]
    A -->|save / forget| M[(Long-term memory<br/>SQLite)]
    A -->|request_viewing| H{Buyer approval}
    H -->|approved| C
    A -->|answer + trace + citations| UI
```

### One turn, step by step

1. The buyer's message is added to the session's message list (short-term memory).
2. The agent sends the system prompt, the saved buyer profile, the tool list and the conversation to the LLM.
3. The LLM either answers or asks for tools.
4. The agent runs each tool. Listing tools go through the **MCP client** to the server subprocess. Knowledge and memory tools run locally. Every call is recorded in a trace and streamed to the browser.
5. Results go back to the LLM. Steps 3 to 5 repeat until a final answer, up to 10 steps.
6. `request_viewing` is intercepted by an approval check **in code**, not only in the prompt, and waits for the buyer's click.

The server streams events to the browser as NDJSON (`tool`, `approval`, `done`), which is what drives the agent bar, the live steps and Dee's reactions.

![Architecture diagram](docs/architecture.png)

More detail, including the system prompt justification, is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Assignment requirements

| Requirement | Where | How to see it |
|---|---|---|
| Python agent + LLM | `realestate_agent/agent.py`, `llm.py` | Agent bar *LLM* dot, any answer |
| Short-term memory | `PropertyAgent.messages` | Ask a follow-up: *"which is cheaper per sq ft?"* |
| Long-term memory | `realestate_agent/memory.py` (SQLite) | State a budget or heart a home, start a new session, ask again |
| RAG with citations, 6 documents | `knowledge/`, `realestate_agent/rag.py` | Ask about deposits or title checks, see green citations and `RAG` steps |
| MCP client and server | `mcp_client.py`, `mcp_server.py` | `MCP` steps in chat, `tests/test_mcp.py` |
| Prompt engineering | `realestate_agent/prompts.py` | Justified in `docs/ARCHITECTURE.md` |
| Tests and error handling | `tests/` | `python -m pytest tests -v` |
| Human oversight | Approval gate in `agent.py` and `web.py` | Ask for a viewing |
| Re-planning | Prompt rules + live status via MCP | Change a home's status, then ask about it |

---

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ (developed on 3.14) |
| Web server | Starlette (ASGI) on Uvicorn, NDJSON streaming |
| Frontend | One HTML page, vanilla JavaScript and CSS, no build step, inline SVG icons and mascot |
| LLM | Groq (best available tool-capable model, picked automatically, OpenAI-compatible API through the `openai` SDK) or Anthropic (`claude-opus-5-5` through the `anthropic` SDK) |
| Agent loop | Hand-written tool-use loop with tracing and an approval gate |
| Tools protocol | Model Context Protocol: `FastMCP` server over stdio, official `ClientSession` client |
| RAG | `fastembed` (`BAAI/bge-small-en-v1.5`, runs locally) + `chromadb` (cosine), section-aware chunking. Hosted builds use a pure-Python BM25 retriever over the same chunks and citations |
| Storage | SQLite for listings, viewing requests and buyer memory |
| Config | `python-dotenv` |
| Testing | `pytest`, Starlette `TestClient`, a scripted LLM for deterministic agent tests |
| Fonts and images | Self-hosted Archivo variable font, local photos from Unsplash and Wikimedia Commons |

---

## Project structure

```
real-estate-agent/
├── realestate_agent/            # The application package
│   ├── agent.py                 # Agent loop: LLM ↔ tools, trace, approval gate, error handling
│   ├── llm.py                   # Provider choice + Groq adapter (OpenAI format ↔ agent format)
│   ├── prompts.py               # System prompt and saved-profile block
│   ├── mcp_server.py            # FastMCP server exposing the listing tools
│   ├── mcp_client.py            # MCP stdio client: list_tools / call_tool
│   ├── properties.py            # Listing store, comparison and payment logic (SQLite)
│   ├── rag.py                   # Chunking, embeddings, Chroma index, retrieval with citations
│   ├── memory.py                # Long-term buyer memory (SQLite)
│   ├── web.py                   # Starlette app: pages, chat stream, approval, shortlist, status
│   ├── cli.py                   # Terminal chat
│   ├── config.py                # Paths and settings
│   ├── ingest.py                # Rebuild the RAG index
│   └── static/
│       ├── index.html           # The whole frontend: portal, advisor drawer, Dee the mascot
│       ├── tokens.css           # Colour, type and spacing tokens
│       ├── fonts/               # Archivo variable font
│       ├── brand/               # Logo, favicon, brand board
│       └── homes/               # Listing photos, photos.json, CREDITS.md
├── data/properties.json         # 12 synthetic listings (seed data)
├── knowledge/                   # 6 Markdown guides used by RAG
├── tests/                       # 68 tests: agent, MCP, RAG, memory, properties, web, Groq adapter
├── docs/
│   ├── ARCHITECTURE.md          # Design, components, prompt justification, limitations
│   ├── architecture.png         # Architecture diagram
│   ├── make_diagram.py          # Script that draws the diagram
│   └── screenshots/             # Images used in this README
├── evidence/test_results.txt    # Latest test run summary
├── app.py                       # Hosted entrypoint (top-level ASGI app)
├── pyproject.toml               # Hosted dependency set and Vercel entrypoint
├── vercel.json                  # Function duration and bundle excludes
├── demo.py                      # Scripted end-to-end demo with the real LLM
├── DESIGN.md                    # Design system: colours, components, mascot states
├── PRODUCT.md                   # Product brief
├── requirements.txt             # Full local install, including embeddings and Chroma
└── .env.example                 # Copy to .env and add one API key
```

Runtime data (`runtime/`: SQLite databases and the Chroma index) is created on first run and is not committed.

---

## Getting started

### Prerequisites

- Python 3.11 or newer
- An API key from **Groq** ([console.groq.com/keys](https://console.groq.com/keys), free tier available) **or** **Anthropic** ([console.anthropic.com](https://console.anthropic.com))
- About 100 MB of disk for the embedding model, downloaded on first run

### Install

```bash
git clone https://github.com/Hamzaviour/deedly-property-advisor.git
cd deedly-property-advisor
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Configure

```bash
# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

Open `.env` and set **one** key:

```ini
GROQ_API_KEY=gsk_...
# or
ANTHROPIC_API_KEY=sk-ant-...
```

Search, listings, shortlist, compare and the calculator all work without a key. Only the advisor needs one.

---

## Usage

### Web app

```bash
python -m realestate_agent.web
```

Open **http://127.0.0.1:8011**. The server binds to localhost only and has no login, so keep it on your own machine. Set `RE_AGENT_PORT` to use another port.

### Demo script for a live presentation

Open the advisor and run the five prompts in order. They are also listed in the drawer:

| # | Shows | Prompt |
|---|---|---|
| 1 | Search, MCP, long-term memory | *My budget is 2.7 crore PKR. I need at least 3 bedrooms, parking, and a commute under 20 minutes. What fits?* |
| 2 | Short-term memory | *Which of those is cheaper per square foot?* |
| 3 | Compare + payment tool | *Compare P-103 and P-110, recommend one, and estimate the monthly payment using your planning assumptions.* |
| 4 | RAG with citations | *How much deposit should I pay, and what should a lawyer check?* |
| 5 | Human approval | *Request a viewing of P-110 on 2026-11-20 under the name Hamza.* |

Then:

- **Re-planning:** set P-110 to *Sold* on its card and ask *"Is P-110 still my best option?"*
- **Long-term memory across sessions:** click the refresh icon next to the buyer name (new session) and ask *"What's my budget?"*

### Terminal chat

```bash
python -m realestate_agent.cli --user hamza
```

Viewing requests ask `Approve ...? [y/N]` in the terminal. Type `/quit` to leave. Start again with the same `--user` to see long-term memory.

### Scripted end-to-end demo

```bash
python demo.py
```

This runs a full scripted session against the real LLM in isolated storage and writes a transcript to `evidence/demo_transcript.md`.

### Rebuild the RAG index

After editing anything in `knowledge/`:

```bash
python -m realestate_agent.ingest
```

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | | Use Groq as the LLM provider |
| `GROQ_MODEL` | auto | Groq model. Unset, Deedly asks Groq which models your key can use and picks the best tool-capable one, so retired models never break it |
| `ANTHROPIC_API_KEY` | | Use Anthropic as the LLM provider |
| `RE_AGENT_MODEL` | `claude-opus-5-5` | Anthropic model |
| `RE_AGENT_PROVIDER` | auto | `groq` or `anthropic`. Auto picks Anthropic if its key is set, otherwise Groq |
| `RE_AGENT_PORT` | `8011` | Web server port |
| `RE_AGENT_RUNTIME` | `./runtime` (`/tmp/deedly` on Vercel) | Where SQLite databases and the Chroma index are stored |
| `RE_AGENT_RAG` | auto | `embeddings` or `lexical` (BM25). Auto uses embeddings when installed, BM25 on Vercel |
| `RE_AGENT_CHAT_LIMIT` | `25` | Hosted only: chat messages per visitor per hour, to protect the API key |

---

## HTTP API

All `POST` endpoints require `Content-Type: application/json`. Requests addressed to any host other than localhost are rejected.

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/` | | The web app |
| `GET` | `/api/state?user=` | | Listings, saved memory, shortlist, model name, MCP tools, whether the LLM is ready |
| `POST` | `/api/chat` | `{user, message}` | NDJSON stream of `tool`, `approval` and `done` events |
| `POST` | `/api/approve` | `{user, id, approved}` | Resolves a pending viewing request |
| `POST` | `/api/reset` | `{user}` | Starts a new session. Long-term memory is kept |
| `POST` | `/api/shortlist` | `{user, id, on}` | Adds or removes a home in the buyer's shortlist memory |
| `POST` | `/api/status` | `{id, status}` | Demo control: `available`, `under_offer` or `sold` |

---

## MCP tools

Served by `realestate_agent/mcp_server.py` and discovered by the client at startup.

| Tool | Purpose |
|---|---|
| `search_properties` | Filter listings by price, bedrooms, area and type |
| `get_property` | Live details and status of one home |
| `compare_properties` | Check homes against the buyer's criteria and rank them |
| `estimate_monthly_payment` | Amortised loan payment |
| `request_viewing` | Simulated viewing request. Gated by buyer approval |

Local tools: `retrieve_knowledge` (RAG), `save_preference` and `forget_preference` (long-term memory).

Comparison and payment maths are deterministic Python, so the model never does arithmetic or ranking from memory.

---

## Testing

```bash
python -m pytest tests -v
```

No API key is needed. The tests use the real MCP server, real SQLite, real embeddings and real Chroma. The LLM is scripted, so they check the agent's plumbing (tool routing, approval, memory, error recovery, streaming), not the model's judgement. `demo.py` exercises the real model.

| File | Covers |
|---|---|
| `test_agent.py` | Tool loop, tracing, approval gate, error and refusal handling, history repair |
| `test_mcp.py` | MCP discovery and calls over stdio |
| `test_rag.py` | Chunking, retrieval quality and relevance floor for both embeddings and BM25 |
| `test_memory.py` | Save, replace, forget, isolation between buyers |
| `test_properties.py` | Search, compare ranking, payment maths, status changes |
| `test_web.py` | Pages, streaming chat, approval, sessions, shortlist, security checks |
| `test_llm.py` | Groq adapter translation and provider selection |

---

## Deployment

### Vercel

The repository deploys to Vercel as a Python (ASGI) function with no extra setup:

- `app.py` exposes the Starlette `app`, and `pyproject.toml` names it as the entrypoint.
- `pyproject.toml` holds a **light dependency set**. RAG switches to the built-in **BM25** retriever, so no embedding model download or Chroma is needed. Citations and the relevance floor work the same way.
- Runtime data goes to `/tmp/deedly`. Listings are re-seeded from `data/properties.json` on each cold start.
- The MCP server still runs as a real stdio subprocess inside the function.
- Each visitor gets a small hourly chat budget (`RE_AGENT_CHAT_LIMIT`), because a public deployment spends the owner's API key.

Steps:

1. Import the GitHub repository in Vercel, or run `vercel deploy --prod` from a clean clone.
2. In **Project → Settings → Environment Variables**, add `GROQ_API_KEY` (or `ANTHROPIC_API_KEY`), then redeploy.

**Known trade-offs on serverless:** conversations, shortlists and status changes live in instance memory and `/tmp`, so a cold start resets them, and a viewing approval can expire if the platform moves the request to a different instance. For a demo that must behave exactly like local, run it locally or on a host with a long-running process and a persistent disk.

### Local or a long-running host

```bash
pip install -r requirements.txt
python -m realestate_agent.web
```

---

## Data and credits

- **Listings and documents** are synthetic, written for this coursework. Area names are real parts of Lahore.
- **Photos** are illustrative, not the listed homes:
  - Lahore photos from **Wikimedia Commons**, under CC BY and CC BY-SA licences. Each one shows its credit on the home sheet, and all are listed in [`realestate_agent/static/homes/CREDITS.md`](realestate_agent/static/homes/CREDITS.md).
  - Other photos from **Unsplash**, listed in the same file.
- **Font:** Archivo (SIL Open Font License).
- **Mascot and logo:** original artwork for this project.

---

## Limitations and roadmap

**Limitations**

- Twelve synthetic listings and six short documents, so retrieval quality on a real corpus is untested.
- Sessions and pending approvals are held in memory, so a restart clears conversations (long-term memory survives).
- Smaller open models on Groq choose tools less reliably than larger ones.

**Ideas for next steps**

- Stream answer text word by word
- Map view with area boundaries
- Area pages built from the neighbourhood guide
- A "My viewings" panel for simulated requests
- More listings and areas, such as DHA, Bahria Town, Wapda Town and Garden Town

---

## Author

**Hamza**: design, development and documentation.

Built for Agent Based Modeling (AIDE3313), Assignment 2.
