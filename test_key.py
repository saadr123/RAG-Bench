"""
Quick diagnostic: checks your .env, then makes ONE tiny API call.

Run it with:   python test_key.py

Cheaper than debugging through the full sweep - it isolates the API call
from chunking, embedding, and retrieval so you know exactly what's broken.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

key = os.environ.get("LLM_API_KEY")
base = os.environ.get("LLM_BASE_URL")
model = os.environ.get("LLM_MODEL")

print("=" * 60)
print("Checking .env values")
print("=" * 60)

if not key:
    print("LLM_API_KEY: MISSING - is your file named exactly .env ?")
    raise SystemExit(1)

# Never print the whole key. Show enough to spot formatting problems.
print(f"LLM_API_KEY:  {key[:6]}...{key[-4:]}   (length {len(key)})")
print(f"LLM_BASE_URL: {base}")
print(f"LLM_MODEL:    {model}")

problems = []
if key.startswith(("'", '"')) or key.endswith(("'", '"')):
    problems.append("key has quote marks around it - remove them")
if key != key.strip():
    problems.append("key has leading/trailing whitespace")
if " " in key:
    problems.append("key contains a space")
if key.startswith("your-key"):
    problems.append("key is still the placeholder from .env.example")
if not base:
    problems.append("LLM_BASE_URL is missing")
if not model:
    problems.append("LLM_MODEL is missing")

if problems:
    print("\nProblems found in .env:")
    for p in problems:
        print(f"  - {p}")
    raise SystemExit(1)

print("\nFormat looks fine. Making one test call...\n")

client = OpenAI(api_key=key, base_url=base)

try:
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=10,
    )
    print("=" * 60)
    print("SUCCESS")
    print("=" * 60)
    print("Model replied:", r.choices[0].message.content)
    if r.usage:
        print(f"Tokens: {r.usage.prompt_tokens} in / {r.usage.completion_tokens} out")
    print("\nYour setup works. Run: python -m src.runner --sample-size 2")
except Exception as e:
    print("=" * 60)
    print(f"FAILED: {type(e).__name__}")
    print("=" * 60)
    print(e)
    print("\nRead the message above - the provider usually says exactly what's wrong.")
    print("If it mentions the model name, try changing LLM_MODEL in .env.")
    print("If it mentions the API key, generate a fresh one at aistudio.google.com.")
