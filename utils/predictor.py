import numpy as np
import pandas as pd
import lightgbm as lgb

def predict_all(X, models):
    predictions = {}

    for name, bundle in models.items():
        model = bundle["model"]
        meta = bundle["meta"]

        # Native LightGBM Booster
        if isinstance(model, lgb.Booster):
            y_pred = model.predict(X)

        # Sklearn-style models
        elif meta["type"] == "classification":
            y_pred = model.predict_proba(X)[:, 1]

        else:
            y_pred = model.predict(X)

        predictions[name] = np.round(y_pred, 3)

    return pd.DataFrame(predictions)
