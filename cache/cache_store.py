import json
import os
from datetime import datetime

CACHE_DIR = "cache"
CACHE_FILE = os.path.join(CACHE_DIR, "user_history.json")


def save_cache(input_data, yield_pred, crops):
    """
    Store user input + model outputs in cache for future learning
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(),
        "input_features": input_data,
        "predicted_yield": float(yield_pred) if yield_pred is not None else None,
        "recommended_crops": crops
    }

    data = []

    # Load existing cache safely
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []

    data.append(record)

    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_cache():
    """
    Load cached user history
    """
    if not os.path.exists(CACHE_FILE):
        return []

    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []
