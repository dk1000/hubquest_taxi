import hashlib
import logging
import os
import shutil
from datetime import datetime

import joblib
from sklearn.metrics import mean_absolute_error, r2_score


def save_model(model, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    logging.info(f"Saving model to the file")

    joblib.dump(model, path / name)
    logging.info(f"Saved model as {name}")


def eval_model(model, x, y, type: str):
    pred_values = model.predict(x)
    mae = mean_absolute_error(y, pred_values)
    r2 = r2_score(y, pred_values)

    logging.info(f"{type} MAE: {round(mae, 3)}\n{type} R2: {round(r2, 3)}")
    return mae, r2


def hash_model_name(params):
    params = str(params)
    hashed_params = hashlib.md5(params.encode()).hexdigest()
    today = datetime.today().strftime("%Y%m%d%H%M%S")
    return hashed_params + today


def load_current_active_model(path):
    if not os.path.exists(path / "active"):
        os.makedirs(path / "active")
    for root, dirs, files in os.walk(path / "active"):
        if len(files) == 0:
            logging.info(f"No active models found")
            return None
        else:
            return joblib.load(path / "active" / files[0])


def set_model_as_active(model, path, name):
    if not os.path.exists(path / "active"):
        os.makedirs(path / "active")
    for root, dirs, files in os.walk(path / "active"):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    logging.info(f"Set current model as active")
    joblib.dump(model, path / "active" / name)


def compare_current_model_with_active(x, y, current_model, active_model, path, name):
    current_mae = eval_model(current_model, x, y, "Current model Test")[0]
    active_mae = eval_model(active_model, x, y, "Active model Test")[0]
    if current_mae < active_mae:
        set_model_as_active(current_model, path, name)
    else:
        logging.info(f"Current active model is better than new one. No changes on Active Model")
