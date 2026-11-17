import urllib.request
from pathlib import Path
from src.utils import resolve_path


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
    sentences, current = list(), list()
    with resolve_path(file_path).open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = list()
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


def extract_token_features(sentence: list[dict], index: int) -> dict:
    word = sentence[index]["word"]
    low = word.lower()
    prev_w = sentence[index - 1]["word"].lower() if index > 0 else "<BOS>"
    next_w = sentence[index + 1]["word"].lower() if index < len(sentence) - 1 else "<EOS>"
    return {
        "word_lower": low,
        "prefix_1": low[:1],
        "prefix_2": low[:2],
        "prefix_3": low[:3],
        "suffix_1": low[-1:],
        "suffix_2": low[-2:],
        "suffix_3": low[-3:],
        "is_upper": word.isupper(),
        "is_title": word.istitle(),
        "is_digit": word.isdigit(),
        "has_hyphen": "-" in word,
        "word_len": len(word),
        "prev_word": prev_w,
        "prev_suffix_2": prev_w[-2:],
        "next_word": next_w,
        "next_suffix_2": next_w[-2:],
    }