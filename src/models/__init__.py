import importlib
import itertools
import json
import pkgutil
import time
from pathlib import Path

from src.features import load_dataset_splits
from src.utils import export_ud_lemmas, load_config, resolve_path, set_seed

MODEL_REGISTRY = {}
for _, module_name, _ in pkgutil.iter_modules([str(Path(__file__).parent)]):
    if not module_name.startswith("_") and module_name != "base":
        mod = importlib.import_module(f"src.models.{module_name}")
        if hasattr(mod, "train_and_evaluate") and hasattr(mod, "load_predictor"):
            MODEL_REGISTRY[module_name] = mod


def expand_param_grid(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    values = [v if isinstance(v, list) else [v] for v in grid.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def get_tagger(cfg: dict | None = None, model_key: str | None = None):
    cfg = cfg or load_config()
    key = (model_key or cfg["translator"]["active_model"]).strip().lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{key}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[key].load_predictor(cfg), cfg["models"][key]["display_name"]


def diagnose_all_models(text: str, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    results = {}
    for key, mod in sorted(MODEL_REGISTRY.items()):
        if not cfg["models"].get(key, {}).get("enabled", True):
            continue
        t0 = time.perf_counter()
        predictor = mod.load_predictor(cfg)
        tokens = predictor(text)
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        results[key] = {
            "display_name": cfg["models"][key]["display_name"],
            "latency_ms": ms,
            "tokens": tokens,
        }
    return results


def run_experiments(
    models: list[str] | None = None,
    run_names: dict[str, list[str]] | list[str] | None = None,
    param_overrides: dict[str, list[dict] | dict] | None = None,
    data_source: str | Path | dict[str, str | Path] | None = None,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    cv_folds: int | None = None,
    seed: int | None = None,
    save_artifacts: bool = True,
    force_overwrite: bool = False,
    config_path: str | None = None,
) -> dict:
    cfg = load_config(config_path)
    run_seed = seed if seed is not None else cfg["runtime"]["random_seed"]
    set_seed(run_seed, cfg["runtime"]["deterministic"])

    train_s, dev_s, test_s, source_tag = load_dataset_splits(
        cfg, data_source=data_source, split_ratios=split_ratios, seed=run_seed
    )
    export_ud_lemmas(train_s, cfg["paths"]["lemmas_file"])

    target_models = models if models is not None else sorted(MODEL_REGISTRY.keys())
    all_metrics = {}
    metrics_path = resolve_path(cfg["paths"]["artifacts_prod"]) / cfg["runtime"]["metrics_file"]
    if metrics_path.exists():
        all_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    for key in target_models:
        if key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{key}'. Available: {list(MODEL_REGISTRY.keys())}")
        if models is None and not cfg["models"].get(key, {}).get("enabled", True):
            continue

        display_name = cfg["models"][key]["display_name"]
        prev_entry = all_metrics.get(display_name, {})
        prev_val_acc = prev_entry.get("best_val_accuracy", -1.0)

        override = (param_overrides or {}).get(key)
        if isinstance(override, dict):
            runs_list = expand_param_grid(override)
        elif override is not None:
            runs_list = override
        else:
            runs_list = list(cfg["models"][key].get("runs", list()))

        selected_names = run_names.get(key) if isinstance(run_names, dict) else run_names
        if selected_names is not None:
            runs_list = [r for r in runs_list if r.get("name") in selected_names]
            if not runs_list:
                raise ValueError(f"No runs matched run_names={selected_names} for '{key}'.")

        res = MODEL_REGISTRY[key].train_and_evaluate(
            train_s, dev_s, test_s, cfg,
            runs_override=runs_list,
            cv_folds=cv_folds,
            save_artifacts=save_artifacts,
            champion_val_acc=-1.0 if force_overwrite else prev_val_acc,
        )
        res["data_source"] = source_tag
        if res["promoted"] or display_name not in all_metrics:
            all_metrics[display_name] = res

    if save_artifacts:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    return all_metrics


def run_all_training(config_path: str | None = None) -> dict:
    return run_experiments(config_path=config_path)


if __name__ == "__main__":
    run_all_training()
