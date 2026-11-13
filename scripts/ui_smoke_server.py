"""Manual UI check without an API key: serves the app with a scripted model that streams.

Run: python scripts/ui_smoke_server.py, then open http://127.0.0.1:8013 and send any message.
The scripted model reads the guides, searches Gulberg listings, then streams an answer with
one real citation and one invented citation, which the grounding check should remove.
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402

from realestate_agent.web import create_app  # noqa: E402

ANSWER = ("For Gulberg flats, **P-109** is the cheapest available option. Pay a deposit of no "
          "more than 10 percent, only against a signed sale agreement [01_buying_process.md#3]. "
          "Stamp duty is exactly 4 percent [07_made_up.md#9].")


class ScriptedStreamingLLM:
    streams_text = True
    model = "scripted-demo-model"

    def __init__(self):
        self.step = 0
        self.beta = NS(messages=NS(create=self._create))

    async def _create(self, **request):
        return await self.stream_create(on_text=lambda _: None, **request)

    async def stream_create(self, *, on_text, messages, **_):
        last = messages[-1]["content"]
        if isinstance(last, str):  # A new buyer message starts the script again.
            self.step = 0
        self.step += 1
        if self.step == 1:
            return NS(stop_reason="tool_use", content=[NS(type="tool_use", id="t1",
                      name="retrieve_knowledge", input={"query": "how much deposit"})])
        if self.step == 2:
            return NS(stop_reason="tool_use", content=[NS(type="tool_use", id="t2",
                      name="search_properties", input={"area": "Gulberg", "property_type": "apartment"})])
        for word in ANSWER.split(" "):
            on_text(word + " ")
            await asyncio.sleep(0.04)
        return NS(stop_reason="end_turn", content=[NS(type="text", text=ANSWER)])


if __name__ == "__main__":
    uvicorn.run(create_app(llm=ScriptedStreamingLLM()), host="127.0.0.1", port=8013)
