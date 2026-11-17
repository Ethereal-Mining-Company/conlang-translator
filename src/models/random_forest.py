import json
import re
import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from src.features import extract_token_features
from src.utils import append_csv_row, make_run_uuid, resolve_path, setup_logger


def _flatten(sentences: list[list[dict]], feat_cfg: dict | None = None):
    X: list[dict] = list()
    y: list[str] = list()
    for sent in sentences:
        for i, tok in enumerate(sent):
            X.append(extract_token_features(sent, i, feat_cfg))
            y.append(tok["label"])
    return X, y


def _build_rf(run: dict, seed: int) -> Pipeline:
    return Pipeline([
        ("vec", DictVectorizer(sparse=True)),
        ("clf", RandomForestClassifier(
            n_estimators=run.get("n_estimators", 200),
            max_depth=run.get("max_depth", None),
            min_samples_split=run.get("min_samples_split", 2),
            max_features=run.get("max_features", "sqrt"),
            class_weight=run.get("class_weight", None),
            random_state=seed,
            n_jobs=-1,
        )),
    ])


def train_and_evaluate(
    train_s,
    dev_s,
    test_s,
    cfg: dict,
    runs_override: list[dict] | None = None,
    cv_folds: int | None = None,
    save_artifacts: bool = True,
    champion_val_acc: float = -1.0,
) -> dict:
    mcfg = cfg["models"]["random_forest"]
    feat_cfg = cfg.get("features")
    seed = cfg["runtime"]["random_seed"]
    folds = cv_folds if cv_folds is not None else cfg["runtime"].get("cv_folds", 1)
    runs = runs_override if runs_override is not None else mcfg["runs"]
    if not runs:
        raise ValueError("No training runs provided for random_forest.")

    logger = setup_logger("rf_trainer", cfg["paths"]["logs"], mcfg["log_file"])

    X_train, y_train = _flatten(train_s, feat_cfg)
    X_dev, y_dev = _flatten(dev_s, feat_cfg)
    X_test, y_test = _flatten(test_s, feat_cfg)

    best_pipe: Pipeline | None = None
    best_acc = -1.0
    best_run: dict | None = None
    best_uuid = ""
    sweep: list[dict] = list()

    for idx, run in enumerate(runs, 1):
        run_uuid = make_run_uuid()
        run_name = run.get("name", f"RF_Run_{idx}")
        t0 = time.perf_counter()
        fold_scores: list[float] = list()

        if folds > 1:
            kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
            for f_idx, (tr_idx, va_idx) in enumerate(kf.split(train_s), 1):
                f_t0 = time.perf_counter()
                X_tr, y_tr = _flatten([train_s[i] for i in tr_idx], feat_cfg)
                X_va, y_va = _flatten([train_s[i] for i in va_idx], feat_cfg)
                f_pipe = _build_rf(run, seed)
                f_pipe.fit(X_tr, y_tr)
                f_acc = float(accuracy_score(y_va, f_pipe.predict(X_va)))
                fold_scores.append(f_acc)
                append_csv_row(cfg["paths"]["epochs_csv"], {
                    "run_uuid": run_uuid,
                    "engine_type": "random_forest",
                    "run_name": run_name,
                    "fold": f_idx,
                    "epoch": 1,
                    "train_loss": "",
                    "val_accuracy": f_acc,
                    "elapsed_sec": round(time.perf_counter() - f_t0, 2),
                })
            pipe = _build_rf(run, seed)
            pipe.fit(X_train, y_train)
            val_acc = float(sum(fold_scores) / len(fold_scores))
        else:
            pipe = _build_rf(run, seed)
            pipe.fit(X_train, y_train)
            val_acc = float(accuracy_score(y_dev, pipe.predict(X_dev)))
            fold_scores = [val_acc]
            append_csv_row(cfg["paths"]["epochs_csv"], {
                "run_uuid": run_uuid,
                "engine_type": "random_forest",
                "run_name": run_name,
                "fold": 1,
                "epoch": 1,
                "train_loss": "",
                "val_accuracy": val_acc,
                "elapsed_sec": round(time.perf_counter() - t0, 2),
            })

        elapsed = round(time.perf_counter() - t0, 2)
        tuned = {k: v for k, v in run.items() if k != "name"}
        logger.info(f"[{run_uuid}] {run_name} | folds={folds} | val_acc={val_acc:.4f} | time={elapsed}s")

        sweep.append({
            "run_uuid": run_uuid,
            "name": run_name,
            "tuned_params": tuned,
            "val_accuracy": val_acc,
            "fold_scores": fold_scores,
            "time_s": elapsed,
            "all_params": {k: str(v) for k, v in pipe.named_steps["clf"].get_params().items()},
        })
        if val_acc > best_acc:
            best_acc, best_pipe, best_run, best_uuid = val_acc, pipe, {**run, "name": run_name}, run_uuid

    if best_pipe is None or best_run is None:
        raise RuntimeError("Training completed without selecting a valid model.")

    t0 = time.perf_counter()
    y_pred = best_pipe.predict(X_test)
    latency_ms = ((time.perf_counter() - t0) / max(len(test_s), 1)) * 1000.0
    test_acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    labels = sorted(list(set(y_test)))
    top_labels = sorted(labels, key=lambda l: y_test.count(l), reverse=True)[: cfg["runtime"]["confusion_matrix_top_k"]]
    cm = confusion_matrix(y_test, y_pred, labels=top_labels).tolist()

    promoted = bool(save_artifacts and best_acc > champion_val_acc)
    metadata = {
        "run_uuid": best_uuid,
        "engine_type": "random_forest",
        "best_run_name": best_run["name"],
        "promoted": promoted,
        "global_seed": seed,
        "cv_folds": folds,
        "feature_cfg": feat_cfg,
        "feature_count": len(best_pipe.named_steps["vec"].feature_names_),
        "target_classes": [str(c) for c in best_pipe.named_steps["clf"].classes_],
        "full_winning_parameters": {k: str(v) for k, v in best_pipe.named_steps["clf"].get_params().items()},
    }

    for entry in sweep:
        is_sweep_winner = entry["run_uuid"] == best_uuid
        append_csv_row(cfg["paths"]["runs_csv"], {
            "run_uuid": entry["run_uuid"],
            "engine_type": "random_forest",
            "run_name": entry["name"],
            "promoted": promoted if is_sweep_winner else False,
            "tuned_params": str(entry["tuned_params"]),
            "mean_cv_accuracy": entry["val_accuracy"],
            "mean_cv_loss": round(1.0 - entry["val_accuracy"], 6),
            "fold_scores": str(entry["fold_scores"]),
            "cv_folds": folds,
            "global_seed": seed,
            "final_test_score": test_acc if is_sweep_winner else "",
            "macro_f1": macro_f1 if is_sweep_winner else "",
            "latency_ms": round(latency_ms, 4) if is_sweep_winner else "",
            "all_model_parameters": json.dumps(entry["all_params"]),
            "metadata_json": json.dumps(metadata) if is_sweep_winner else "",
        })

    if promoted:
        out_path = resolve_path(cfg["paths"]["artifacts_prod"]) / mcfg["artifact_name"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": best_pipe, "metadata": metadata, "feature_cfg": feat_cfg}, out_path)
        logger.info(f"[{best_uuid}] Promoted new champion ({best_acc:.4f} > {champion_val_acc:.4f})")
    elif save_artifacts:
        logger.info(f"[{best_uuid}] Challenger ({best_acc:.4f}) did not beat champion ({champion_val_acc:.4f}); kept existing artifact.")

    return {
        "display_name": mcfg["display_name"],
        "best_run_uuid": best_uuid,
        "best_run_name": best_run["name"],
        "best_val_accuracy": best_acc,
        "promoted": promoted,
        "sweep": sweep,
        "best_test_metrics": {"test_accuracy": test_acc, "macro_f1": macro_f1, "latency_ms": latency_ms},
        "confusion_matrix": {"labels": top_labels, "matrix": cm},
    }


def load_predictor(cfg: dict):
    art = resolve_path(cfg["paths"]["artifacts_prod"]) / cfg["models"]["random_forest"]["artifact_name"]
    loaded = joblib.load(art)
    pipe = loaded["pipeline"] if isinstance(loaded, dict) else loaded
    feat_cfg = loaded.get("feature_cfg", cfg.get("features")) if isinstance(loaded, dict) else cfg.get("features")

    def predict_fn(text: str) -> list[dict]:
        words = re.findall(r"\w+|[^\w\s]", text)
        if not words:
            return list()
        sent = [{"word": w} for w in words]
        feats = [extract_token_features(sent, i, feat_cfg) for i in range(len(sent))]
        preds = pipe.predict(feats)
        out: list[dict] = list()
        for w, lbl in zip(words, preds):
            parts = str(lbl).split("|", 1)
            out.append({"word": w, "upos": parts[0], "morph": parts[1] if len(parts) > 1 else "", "label": str(lbl)})
        return out

    return predict_fn
