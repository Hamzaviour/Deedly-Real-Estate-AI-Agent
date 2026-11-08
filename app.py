"""Deployment entrypoint. Hosts that load a top-level ASGI `app` (such as Vercel) start here.

Locally, run `python -m realestate_agent.web` instead.
"""
from realestate_agent.web import create_app

app = create_app()
