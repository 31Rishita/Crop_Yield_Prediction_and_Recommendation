import json
import os
from datetime import datetime

CACHE_FILE = "cache/user_history.json"

def save_cache(input_data, yield_pred, crops):
    os.makedirs("cache", exist_ok=True)

    record = {
        "time": datetime.now().isoformat(),
        "input": input_data,
        "yield": yield_pred,
        "recommendations": crops
    }

    data = []
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)

    data.append(record)

    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)
