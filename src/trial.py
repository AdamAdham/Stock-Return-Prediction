from src.config import settings
import json
import os

aapl_path = os.path.join(settings.RAW_DIR, "AAPL.json")
with open(aapl_path, "r", encoding="utf-8") as file:
    aapl = json.load(file)

print(aapl_path)
