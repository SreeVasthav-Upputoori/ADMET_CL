import os
import json
import pickle
import joblib
import lightgbm as lgb

def load_model(path):
    """
    Robust loader for sklearn, joblib, and native LightGBM models
    """
    # Try joblib
    try:
        return joblib.load(path)
    except Exception:
        pass

    # Try pickle
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass

    # Try native LightGBM Booster
    try:
        return lgb.Booster(model_file=path)
    except Exception:
        pass

    raise ValueError(f"Unable to load model: {path}")


def load_all_models(model_dir="models"):
    """
    Loads all models defined in model_registry.json
    """
    registry_path = os.path.join(model_dir, "model_registry.json")

    with open(registry_path, "r") as f:
        registry = json.load(f)

    models = {}

    for name, meta in registry.items():
        model_path = os.path.join(model_dir, meta["file"])
        models[name] = {
            "model": load_model(model_path),
            "meta": meta
        }

    return models
