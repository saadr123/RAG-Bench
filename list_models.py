"""
Asks your API key which models it can actually use.

Run with:   python list_models.py

Model names get retired and renamed constantly, so this beats guessing.
Pick a Flash or Flash-Lite model from the output and put it in .env as
LLM_MODEL. Pro models are not on the free tier.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_BASE_URL"],
)

try:
    models = sorted(m.id for m in client.models.list())
except Exception as e:
    print(f"Could not list models: {type(e).__name__}: {e}")
    raise SystemExit(1)

# Strip the "models/" prefix Google returns - .env wants the bare name
cleaned = [m.split("/")[-1] for m in models]

flash = [m for m in cleaned if "flash" in m.lower()]
other = [m for m in cleaned if m not in flash]

print(f"{len(cleaned)} models available to your key.\n")
print("=== FLASH / FLASH-LITE (free tier - pick one of these) ===")
for m in flash:
    print(" ", m)

print(f"\n=== EVERYTHING ELSE ({len(other)} models, mostly paid or non-chat) ===")
for m in other[:15]:
    print(" ", m)
if len(other) > 15:
    print(f"  ... and {len(other) - 15} more")

print("\nPut your choice in .env as:  LLM_MODEL=<name from the Flash list>")
