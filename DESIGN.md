# Deedly design system

Deedly looks and works like a property portal for Lahore: search first, homes as cards, and an advisor that slides in when you want it. The agent's work stays visible, but quietly, in a thin bar across the top.

## Logo and mascot

The mark is a letter D with a home in its counter (`static/brand/deedly-mark.svg`), drawn in brick on white or white on brick.

Dee is the advisor's mascot: a small house with a brick roof, a chimney and a face. Dee is drawn inline as SVG and reacts to real events:

| State | When | What changes |
|---|---|---|
| Idle | Always | Gentle bob, random blinks, eyes follow the pointer |
| Wave | Page load, hover | Hand waves |
| Think | A message is sent | Thinking dots, small "o" mouth |
| Work | A listing tool runs | Chimney smoke |
| Read | The guides are searched (RAG) | Reading glasses |
| Happy | An answer arrives | Closed smiling eyes, a hop |
| Sad | A turn fails | Frown, droop |
| Love | A home is shortlisted | A heart floats up |

## Colour

Tokens live in `static/tokens.css`.

| Token | Value | Use |
|---|---|---|
| Brick | `#B4472C` | Primary actions, prices, shortlist, the logo |
| Teal | `#17403C` | Dark sections, selected tabs, buyer messages |
| Teal deep | `#0E2A27` | Agent bar, compare tray |
| Sand | `#F6F3EE` | Page background |
| Ink | `#1A1D21` | Text |
| Gold | `#E2A43A` | Under offer, approval, focus rings |
| Green | `#2E8A5C` | Evidence ticks, citations, best values in comparisons |
| Red | `#C0392B` | Sold stamp and errors |

## Type

Archivo variable, self-hosted in `static/fonts`. Headings use weight 800 at 82 to 88 percent width. Numbers are tabular.

## Components

- **Agent bar:** 28 px tall, one line: LLM, short-term, long-term, RAG, MCP, approval and re-planning. A dot turns green only on a real event. Counters for memory, tools and sources open a dropdown with the details.
- **Hero search:** type tabs, location, max price, bedrooms, keyword, popular areas.
- **Area cards:** photo, area name, number of homes, lowest price.
- **Home card:** photo with type and status badges, heart (shortlist) and compare buttons, price with price per square foot, beds, baths, size, commute, a demo status control and an Ask button.
- **Compare tray and dialog:** up to three homes side by side, best value in each row ticked in green.
- **Home sheet:** gallery with photo credit, facts, features, mortgage calculator, ask and viewing actions.
- **Advisor drawer:** Dee in the header with a RAG badge, demo prompts, live tool steps (MCP, RAG, MEM), answers with citations and home cards, and the approval block.

## Motion

Hero image settles, cards rise in a stagger, photos zoom on hover, the drawer slides, dialogs scale in, agent dots pop when they turn on, and the SOLD stamp lands. Everything is switched off under reduced motion.

## Voice

Plain and specific. Use home, buyer, viewing request and source. Always keep the synthetic data notice visible.
