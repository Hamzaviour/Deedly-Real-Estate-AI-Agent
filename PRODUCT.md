# Deedly

## Purpose

Deedly is a Python AI property advisory agent built for Assignment 2 of Agent Based Modeling (AIDE3313), scenario 38. It retrieves property details, remembers buyer preferences and compares options.

## Primary user

An instructor marking a live demo on a laptop or projected screen. They need to see, within seconds, that the agent uses an LLM, short-term memory, long-term memory, RAG with citations, a real MCP tool and human approval. The on-screen conversation is that of a home buyer, so the product must still read as a credible property adviser.

## Capabilities

- Chat with an agent that chooses tools and cites retrieved documents.
- Search, inspect and compare listings over MCP. Estimate loan payments.
- Remember buyer facts across sessions. A new session clears the conversation only.
- Re-plan when a listing's status changes. The demo lets the presenter change a status.
- Send simulated viewing requests only after explicit approval.

## Constraints and truth

- All listings, neighbourhoods and documents are synthetic. Viewing requests are simulated. This must stay visible.
- No claims of real customers, licensing, accuracy figures or market data.
- The agent gives general information, not legal or financial advice.
- Runs locally from Python with no build step. No login, localhost only.
- Must work on a projector and with keyboard only.

## Terminology

Buyer, listing, home, viewing request, saved memory, source, tool. "Deedly" is the brand. "Demo Realty" remains the fictional agency named in the source documents.

## Platform

web

## Stack

Delegated to the builder: Starlette serving one static HTML page, with no framework or build step.
