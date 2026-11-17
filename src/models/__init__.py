import importlib
import pkgutil
from pathlib import Path

from src.features import download_ud_ewt, parse_conllu
from src.models.base import BaseTagger
from src.utils import load_config, resolve_path, set_seed
from src.visualizations import plot_model_head_to_head

MODEL_REGISTRY: dict[str, type[BaseTagger]] = {}


def register_model(key: str):
    def decorator(cls: type[BaseTagger]):
        MODEL_REGISTRY[key.lower()] = cls
        return cls
    return decorator


def _discover_models() -> None:
    pkg_dir = Path(__file__).resolve().parent
    for _, module_name, _ in pkgutil.iter_modules([str(pkg_dir)]):
        if module_name != "base":
            importlib.import_module(f"src.models.{module_name}")


_discover_models()


def get_tagger(cfg: dict, model_key: str | None = None) -> tuple[BaseTagger, str]:
    key = (model_key or cfg["translator"]["active_model"]).lower()
    if key not in cfg["models"] or key not in MODEL_REGISTRY:
        raise ValueError(f"Unregistered model: {key}")
    mcfg = cfg["models"][key]
    artifact = resolve_path(cfg["paths"]["artifacts_prod"]) / mcfg["artifact_name"]
    if not artifact.exists():
        raise FileNotFoundError(f"Missing {artifact}. Run 'python -m src.models' first.")
    instance = MODEL_REGISTRY[key](cfg, key).load(artifact)
    return instance, mcfg["display_name"]


def run_all_training(config_path: str | Path | None = None) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["runtime"]["random_seed"], cfg["runtime"]["deterministic"])
    paths = download_ud_ewt(cfg)
    train_s, dev_s, test_s = parse_conllu(paths["train"]), parse_conllu(paths["dev"]), parse_conllu(paths["test"])
    for key, mcfg in cfg["models"].items():
        if mcfg.get("enabled", True) and key in MODEL_REGISTRY:
            MODEL_REGISTRY[key](cfg, key).train(train_s, dev_s, test_s)
    plot_model_head_to_head(resolve_path(cfg["paths"]["artifacts_prod"]) / cfg["runtime"]["metrics_file"], cfg["paths"]["figures"])


if __name__ == "__main__":
    run_all_training()