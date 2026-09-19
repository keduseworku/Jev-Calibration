"""Send 5 examples and print raw responses. Run first, once TYPESAFE_API_KEY is set."""
import asyncio, json, sys
from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient
from jev_calibration.dataset import load_dataset
from jev_calibration.jev_client import serialize
from jev_calibration.questions import QUESTIONS

async def main():
    load_dotenv()
    rows = load_dataset()
    picks = [next(r for r in rows if r["category"] == c) for c in
             ["strong_positive", "strong_negative", "weak_positive", "weak_negative", "neutral_positive"]]
    async with AsyncTypeSafeClient() as client:
        for r in picks:
            resp = await client.system_one(state={"text": r["text"]}, questions=QUESTIONS)
            print(r["category"], "|", r["text"][:90])
            print(json.dumps(serialize(resp), indent=1)); print("model:", resp.model, "usage:", resp.usage); print()

asyncio.run(main())
