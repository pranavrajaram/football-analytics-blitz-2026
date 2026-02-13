from v2.play_generator import PlayGenerator
from v2.config import OPENAI_API_KEY
import os

print(f"API Key present: {bool(OPENAI_API_KEY)}")
# print(f"Key start: {OPENAI_API_KEY[:10]}")

try:
    gen = PlayGenerator()
    print(f"Model: {gen.model}")
    print("Attempting generation...")
    play = gen.generate({"down": 1, "distance": 10}, [])
    print(f"Result: {play}")
except Exception as e:
    print(f"CRASH: {e}")
