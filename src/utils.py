import logging
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import yaml


def get_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path else get_root() / "config.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(rel_or_abs: str | Path) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else get_root() / p


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(name: str, log_dir: str | Path, filename: str) -> logging.Logger:
    out_dir = resolve_path(log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        for h in (logging.FileHandler(out_dir / filename, encoding="utf-8"), logging.StreamHandler()):
            h.setFormatter(fmt)
            logger.addHandler(h)
    return logger


def export_ud_lemmas(sentences: list[list[dict]], out_path: str | Path = "data/lemmas.yaml") -> None:
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for sent in sentences:
        for tok in sent:
            w, lem, upos = tok["word"].lower(), tok["lemma"].lower(), tok["upos"]
            if upos in ("NOUN", "VERB", "ADJ") and w.isalpha() and lem.isalpha() and w != lem:
                counts.setdefault(upos, {}).setdefault(w, {})
                counts[upos][w][lem] = counts[upos][w].get(lem, 0) + 1

    table = {
        upos: {w: max(lemmas.items(), key=lambda x: x[1])[0] for w, lemmas in sorted(words.items())}
        for upos, words in sorted(counts.items())
    }
    dest = resolve_path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(table, sort_keys=True), encoding="utf-8")


@lru_cache(maxsize=1)
def _load_lemma_table() -> dict:
    path = resolve_path("data/lemmas.yaml")
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


@lru_cache(maxsize=1)
def _load_reverse_lemma_table() -> dict:
    table = _load_lemma_table()
    rev: dict[str, dict[str, str]] = {}
    for upos, mapping in table.items():
        rev[upos] = {}
        for surface, lemma in sorted(mapping.items()):
            if upos == "VERB" and surface.endswith(("ing", "s")):
                continue
            rev[upos].setdefault(lemma, surface)
    return rev


def lemmatize_token(word: str, upos: str) -> str:
    low = word.lower()
    table = _load_lemma_table()
    if low in table.get(upos, {}):
        return table[upos][low]
    if len(low) <= 3:
        return low
    if upos == "NOUN":
        if low.endswith("ies") and len(low) > 4:
            return low[:-3] + "y"
        if low.endswith(("ses", "xes", "zes", "ches", "shes")) and len(low) > 4:
            return low[:-2]
        if low.endswith("s") and not low.endswith("ss"):
            return low[:-1]
    elif upos == "VERB":
        for sfx, min_len in (("ing", 5), ("ed", 4)):
            if low.endswith(sfx) and len(low) > min_len:
                stem = low[: -len(sfx)]
                return stem[:-1] if len(stem) >= 2 and stem[-1] == stem[-2] else stem
        if low.endswith("ied") and len(low) > 4:
            return low[:-3] + "y"
        if low.endswith("es") and len(low) > 4:
            return low[:-2]
        if low.endswith("s") and not low.endswith("ss"):
            return low[:-1]
    elif upos == "ADJ":
        for sfx, repl, min_len in (("ier", "y", 4), ("er", "", 4), ("iest", "y", 5), ("est", "", 5)):
            if low.endswith(sfx) and len(low) > min_len:
                return low[: -len(sfx)] + repl
    return low


def inflect_english(lemma: str, upos: str, feature: str) -> str:
    low = lemma.lower()
    rev = _load_reverse_lemma_table()
    if feature in ("PLUR", "PAST") and low in rev.get(upos, {}):
        return rev[upos][low]
    if feature == "PLUR":
        if low.endswith("y") and len(low) > 2 and low[-2] not in "aeiou":
            return f"{low[:-1]}ies"
        if low.endswith(("s", "x", "z", "ch", "sh")):
            return f"{low}es"
        return f"{low}s"
    if feature == "PAST":
        if low.endswith("e"):
            return f"{low}d"
        return f"{low}ed"
    if feature == "FUT":
        return f"will {low}"
    if feature == "PROG":
        if low.endswith("e") and not low.endswith("ee"):
            return f"{low[:-1]}ing"
        return f"{low}ing"
    if feature == "COMP":
        return f"more {low}"
    if feature == "SUPER":
        return f"most {low}"
    return low