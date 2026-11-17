import json
import os
import re
import resource
import signal
import sys
import threading
import time
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.features import extract_token_features
from src.utils import append_csv_row, make_run_uuid, resolve_path, setup_logger

os.environ.setdefault("OMP_NUM_THREADS", "6")


class _ProgressBar:
    def __init__(self, items: list, desc: str = ""):
        self.items = list(items)
        self.total = max(len(self.items), 1)
        self.desc = desc

    def __iter__(self):
        for idx, item in enumerate(self.items, 1):
            yield idx, item

    def update(self, idx: int, **metrics) -> None:
        filled = int(20 * idx / self.total)
        bar = "█" * filled + "░" * (20 - filled)
        pct = int(100 * idx / self.total)
        info = " | ".join(f"{k}={v}" for k, v in metrics.items())
        print(f"\r{self.desc} [{bar}] {pct:3d}% | {info}", end="", flush=True)

    def close(self) -> None:
        print()


def _get_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = (1024 ** 3) if sys.platform == "darwin" else (1024 ** 2)
    return rss / divisor


class _MemoryWatchdog:
    def __init__(self, max_ram_gb: float = 14.0):
        self.max_ram_gb = max_ram_gb
        self._stop = threading.Event()
        self.tripped_reason: str | None = None
        self._thread = threading.Thread(target=self._monitor, daemon=True)

    def _monitor(self) -> None:
        while not self._stop.wait(1.0):
            ram = _get_rss_gb()
            if ram > self.max_ram_gb:
                self.tripped_reason = f"RAM limit exceeded ({ram:.2f} GB > {self.max_ram_gb:.2f} GB)"
                signal.raise_signal(signal.SIGINT)
                return

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        if self.tripped_reason and exc_type is KeyboardInterrupt:
            raise RuntimeError(f"Memory boundary triggered: {self.tripped_reason}") from exc_val


def _flatten(sentences: list[list[dict]], feat_cfg: dict | None = None):
    X: list[dict] = list()
    y: list[str] = list()
    for sent in sentences:
        for i, tok in enumerate(sent):
            X.append(extract_token_features(sent, i, feat_cfg))
            y.append(tok["label"])
    return X, y


def _to_dense_f32(X):
    est_gb = (X.shape[0] * X.shape[1] * 4) / (1024 ** 3)
    if est_gb > 3.5:
        raise MemoryError(f"Dense matrix too large ({est_gb:.2f} GB > 3.50 GB safety cap).")
    return X.toarray().astype(np.float32)


def _fit_with_progress(
    X_tr_dict,
    y_tr,
    X_va_dict,
    y_va,
    run: dict,
    run_uuid: str,
    run_name: str,
    fold: int,
    cfg: dict,
    seed: int,
    logger,
) -> tuple[Pipeline, float]:
    n_hash = run.get("n_hash_features", 2048)
    total_iter = run.get("max_iter", 200)
    step = run.get("step_iter", 25)

    hasher = FeatureHasher(n_features=n_hash, input_type="dict", dtype=np.dtype(np.float32))
    transformer = FunctionTransformer(_to_dense_f32, accept_sparse=True)

    X_tr_dense = np.asarray(transformer.transform(hasher.transform(X_tr_dict)), dtype=np.float32)
    X_va_dense = np.asarray(transformer.transform(hasher.transform(X_va_dict)), dtype=np.float32)

    clf = HistGradientBoostingClassifier(
        max_iter=step,
        warm_start=True,
        learning_rate=run.get("learning_rate", 0.08),
        max_leaf_nodes=run.get("max_leaf_nodes", 63),
        max_bins=run.get("max_bins", 128),
        l2_regularization=run.get("l2_regularization", 0.1),
        early_stopping=False,
        random_state=seed,
    )

    best_val_acc = -1.0
    stale_steps = 0
    steps = list(range(step, total_iter + 1, step))
    pbar = _ProgressBar(steps, desc=f"{run_name} [{run_uuid}]")

    for idx, cur_iter in pbar:
        st_t0 = time.perf_counter()
        clf.set_params(max_iter=cur_iter)
        clf.fit(X_tr_dense, y_tr)
        val_acc = float(accuracy_score(y_va, clf.predict(X_va_dense)))
        st_sec = round(time.perf_counter() - st_t0, 2)
        pbar.update(idx, iter=f"{cur_iter}/{total_iter}", val_acc=f"{val_acc:.4f}", ram=f"{_get_rss_gb():.1f}GB")

        append_csv_row(cfg["paths"]["epochs_csv"], {
            "run_uuid": run_uuid,
            "engine_type": "hist_gbdt",
            "run_name": run_name,
            "fold": fold,
            "epoch": cur_iter,
            "train_loss": "",
            "val_accuracy": val_acc,
            "elapsed_sec": st_sec,
        })

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            stale_steps = 0
        else:
            stale_steps += 1
            if stale_steps >= 2:
                pbar.close()
                logger.info(f"[{run_uuid}] {run_name} early-stopped at iter={cur_iter} (val_acc={best_val_acc:.4f})")
                break
    else:
        pbar.close()

    pipe = Pipeline([("hash", hasher), ("dense", transformer), ("clf", clf)])
    return pipe, best_val_acc


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
    mcfg = cfg["models"]["hist_gbdt"]
    feat_cfg = cfg.get("features")
    seed = cfg["runtime"]["random_seed"]
    folds = cv_folds if cv_folds is not None else cfg["runtime"].get("cv_folds", 1)
    runs = runs_override if runs_override is not None else mcfg["runs"]
    if not runs:
        raise ValueError("No training runs provided for hist_gbdt.")

    logger = setup_logger("hgb_trainer", cfg["paths"]["logs"], mcfg["log_file"])

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
        run_name = run.get("name", f"HGB_Run_{idx}")
        t0 = time.perf_counter()
        fold_scores: list[float] = list()

        with _MemoryWatchdog(max_ram_gb=14.0):
            if folds > 1:
                kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
                for f_idx, (tr_idx, va_idx) in enumerate(kf.split(train_s), 1):
                    X_tr, y_tr = _flatten([train_s[i] for i in tr_idx], feat_cfg)
                    X_va, y_va = _flatten([train_s[i] for i in va_idx], feat_cfg)
                    _, f_acc = _fit_with_progress(X_tr, y_tr, X_va, y_va, run, run_uuid, run_name, f_idx, cfg, seed, logger)
                    fold_scores.append(f_acc)
                pipe, _ = _fit_with_progress(X_train, y_train, X_dev, y_dev, run, run_uuid, run_name, 0, cfg, seed, logger)
                val_acc = float(sum(fold_scores) / len(fold_scores))
            else:
                pipe, val_acc = _fit_with_progress(X_train, y_train, X_dev, y_dev, run, run_uuid, run_name, 1, cfg, seed, logger)
                fold_scores = [val_acc]

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
        "engine_type": "hist_gbdt",
        "best_run_name": best_run["name"],
        "promoted": promoted,
        "global_seed": seed,
        "cv_folds": folds,
        "feature_cfg": feat_cfg,
        "n_hash_features": best_pipe.named_steps["hash"].n_features,
        "target_classes": [str(c) for c in best_pipe.named_steps["clf"].classes_],
        "full_winning_parameters": {k: str(v) for k, v in best_pipe.named_steps["clf"].get_params().items()},
    }

    for entry in sweep:
        is_sweep_winner = entry["run_uuid"] == best_uuid
        append_csv_row(cfg["paths"]["runs_csv"], {
            "run_uuid": entry["run_uuid"],
            "engine_type": "hist_gbdt",
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
    art = resolve_path(cfg["paths"]["artifacts_prod"]) / cfg["models"]["hist_gbdt"]["artifact_name"]
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
            upos_tag = parts[0]
            morph_tag = parts[1] if len(parts) > 1 else ""
            out.append({"word": w, "upos": upos_tag, "morph": morph_tag, "label": str(lbl)})
        return out


    return predict_fn
