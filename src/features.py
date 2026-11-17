import csv
import random
import urllib.request
from pathlib import Path

from src.utils import resolve_path


def _fetch_if_url(path_or_url: str | Path, cache_dir: Path) -> Path:
    s = str(path_or_url)
    if s.startswith(("http://", "https://")):
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest = cache_dir / Path(s.split("?")[0]).name
        if not dest.exists():
            urllib.request.urlretrieve(s, dest)
        return dest
    return resolve_path(s)


def download_ud_ewt(cfg: dict) -> dict[str, Path]:
    out_dir = resolve_path(cfg["paths"]["raw_data"])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, url in cfg["dataset"]["urls"].items():
        dest = out_dir / f"en_ewt-ud-{split}.conllu"
        if not dest.exists():
            urllib.request.urlretrieve(url, dest)
        paths[split] = dest
    return paths


def parse_conllu(file_path: str | Path) -> list[list[dict]]:
    sentences: list[list[dict]] = []
    current: list[dict] = []
    with resolve_path(file_path).open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 6 or "-" in cols[0] or "." in cols[0]:
                continue
            word, lemma, upos, feats = cols[1], cols[2], cols[3], cols[5]
            current.append({
                "word": word,
                "lemma": lemma,
                "upos": upos,
                "feats": feats,
                "label": build_target_label(upos, feats),
            })
    if current:
        sentences.append(current)
    return sentences


def parse_tabular_corpus(file_path: str | Path) -> list[list[dict]]:
    path = resolve_path(file_path)
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    sentences: list[list[dict]] = []
    current: list[dict] = []
    last_sid = None
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            sid = row.get("sentence_id", "0")
            if last_sid is not None and sid != last_sid and current:
                sentences.append(current)
                current = []
            last_sid = sid
            word = row["word"].strip()
            upos = row.get("upos", "X").strip()
            feats = row.get("feats", "_").strip()
            lemma = row.get("lemma", word.lower()).strip()
            label = row.get("label") or build_target_label(upos, feats)
            current.append({"word": word, "lemma": lemma, "upos": upos, "feats": feats, "label": label})
    if current:
        sentences.append(current)
    return sentences


def _parse_any(file_path: Path) -> list[list[dict]]:
    if file_path.suffix.lower() in (".csv", ".tsv"):
        return parse_tabular_corpus(file_path)
    return parse_conllu(file_path)


def split_sentences(
    sentences: list[list[dict]],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[list[list[dict]], list[list[dict]], list[list[dict]]]:
    data = list(sentences)
    rng = random.Random(seed)
    rng.shuffle(data)
    n = len(data)
    n_train = int(n * ratios[0])
    n_dev = int(n * ratios[1])
    return data[:n_train], data[n_train : n_train + n_dev], data[n_train + n_dev :]


def load_dataset_splits(
    cfg: dict,
    data_source: str | Path | dict[str, str | Path] | None = None,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[list[list[dict]], list[list[dict]], list[list[dict]], str]:
    cache_dir = resolve_path(cfg["paths"]["raw_data"])
    if data_source is None:
        paths = download_ud_ewt(cfg)
        return parse_conllu(paths["train"]), parse_conllu(paths["dev"]), parse_conllu(paths["test"]), "ud_english_ewt"

    if isinstance(data_source, dict):
        tr_p = _fetch_if_url(data_source["train"], cache_dir)
        dv_p = _fetch_if_url(data_source["dev"], cache_dir)
        ts_p = _fetch_if_url(data_source["test"], cache_dir)
        return _parse_any(tr_p), _parse_any(dv_p), _parse_any(ts_p), f"custom_dict:{tr_p.stem}"

    single_path = _fetch_if_url(data_source, cache_dir)
    all_sents = _parse_any(single_path)
    tr_s, dv_s, ts_s = split_sentences(all_sents, ratios=split_ratios, seed=seed)
    return tr_s, dv_s, ts_s, f"auto_split:{single_path.name}"


def build_target_label(upos: str, feats: str) -> str:
    if feats == "_" or not feats:
        return upos
    fmap = dict(item.split("=", 1) for item in feats.split("|") if "=" in item)
    tags = [upos]
    if upos == "NOUN" and fmap.get("Number") == "Plur":
        tags.append("Number=Plur")
    elif upos == "VERB":
        if fmap.get("Tense") in ("Past", "Fut"):
            tags.append(f"Tense={fmap['Tense']}")
        elif fmap.get("VerbForm") in ("Part", "Ger"):
            tags.append(f"VerbForm={fmap['VerbForm']}")
        elif fmap.get("Person") == "3" and fmap.get("Number") == "Sing":
            tags.append("Person=3|Number=Sing")
    elif upos == "ADJ" and fmap.get("Degree") in ("Cmp", "Sup"):
        tags.append(f"Degree={fmap['Degree']}")
    elif upos == "PRON" and fmap.get("Poss") == "Yes":
        tags.append("Poss=Yes")
    return "|".join(tags)


def extract_token_features(sentence: list[dict], index: int, feat_cfg: dict | None = None) -> dict:
    fcfg = feat_cfg or {"prefix_lengths": [1, 2, 3], "suffix_lengths": [1, 2, 3, 4], "context_window": 2, "include_orthography": True}
    word = sentence[index]["word"]
    low = word.lower()
    feats: dict = {"word_lower": low, "word_len": len(word)}

    for p in fcfg.get("prefix_lengths", [1, 2, 3]):
        feats[f"prefix_{p}"] = low[:p]
    for s in fcfg.get("suffix_lengths", [1, 2, 3, 4]):
        feats[f"suffix_{s}"] = low[-s:]

    if fcfg.get("include_orthography", True):
        feats["is_upper"] = word.isupper()
        feats["is_title"] = word.istitle()
        feats["is_digit"] = word.isdigit()
        feats["has_hyphen"] = "-" in word
        feats["has_apostrophe"] = "'" in word

    window = fcfg.get("context_window", 2)
    for offset in range(1, window + 1):
        prev_w = sentence[index - offset]["word"].lower() if index - offset >= 0 else f"<BOS_{offset}>"
        next_w = sentence[index + offset]["word"].lower() if index + offset < len(sentence) else f"<EOS_{offset}>"
        feats[f"prev_{offset}_word"] = prev_w
        feats[f"prev_{offset}_suf2"] = prev_w[-2:]
        feats[f"next_{offset}_word"] = next_w
        feats[f"next_{offset}_suf2"] = next_w[-2:]

    return feats
