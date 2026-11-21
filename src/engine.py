import csv
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import yaml

from src.utils import inflect_english, lemmatize_token, resolve_path

POS_MAP = {
    "n.": "NOUN", "n": "NOUN", "noun": "NOUN",
    "propn.": "PROPN", "propn": "PROPN", "pn.": "PROPN", "name": "PROPN",
    "v.": "VERB", "v": "VERB", "verb": "VERB",
    "aux.": "AUX", "aux": "AUX",
    "adj.": "ADJ", "adj": "ADJ",
    "adv.": "ADV", "adv": "ADV",
    "pron.": "PRON", "pron": "PRON",
    "det.": "DET", "det": "DET", "dem.": "DET", "dem": "DET", "art.": "DET", "art": "DET",
    "prep.": "ADP", "prep": "ADP", "adp.": "ADP", "adp": "ADP",
    "conj.": "CCONJ", "conj": "CCONJ", "cconj.": "CCONJ", "cconj": "CCONJ",
    "sconj.": "SCONJ", "sconj": "SCONJ",
    "part.": "PART", "part": "PART",
    "intj.": "INTJ", "intj": "INTJ", "interj.": "INTJ", "interj": "INTJ",
    "num.": "NUM", "num": "NUM",
    "sym.": "SYM", "sym": "SYM",
    "punct.": "PUNCT", "punct": "PUNCT",
    "x": "X",
}


class _HTMLTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_row = self.in_cell = False
        self.row, self.cell, self.rows = list(), list(), list()

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.in_row, self.row = True, list()
        elif tag in ("td", "th") and self.in_row:
            self.in_cell, self.cell = True, list()

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.in_cell:
            self.in_cell = False
            self.row.append("".join(self.cell).strip())
        elif tag == "tr" and self.in_row:
            self.in_row = False
            if len(self.row) >= 3:
                self.rows.append(self.row)

    def handle_data(self, data):
        if self.in_cell:
            self.cell.append(data)


class LanguageProfile:
    def __init__(self, name: str, folder: Path):
        self.name = name
        self.folder = folder
        self.word_order = "SVO"
        self.adj_before_noun = True
        self.affixes: dict[str, str] = {}
        self.loaded_sources: list[str] = list()
        self.eng_to_con: dict[tuple[str, str], str] = {}
        self.eng_fallback: dict[str, str] = {}
        self.eng_phrases: dict[str, str] = {}
        self.con_to_eng: dict[str, tuple[str, str]] = {}
        self.con_phrases: dict[str, str] = {}
        self._load_all()

    def _load_all(self) -> None:
        for jf in sorted(self.folder.glob("*.json")):
            self._parse_json(jf)
            self.loaded_sources.append(jf.name)
        for cf in sorted(self.folder.glob("*.csv")):
            self._parse_csv(cf)
            self.loaded_sources.append(cf.name)
        for hf in sorted(self.folder.glob("*.html")):
            self._parse_html(hf)
            self.loaded_sources.append(hf.name)
        for yf in sorted(self.folder.glob("*.yaml")) + sorted(self.folder.glob("*.yml")):
            self._parse_yaml(yf)
            self.loaded_sources.append(yf.name)
        for tf in sorted(self.folder.glob("*.txt")):
            self._parse_txt(tf)
            self.loaded_sources.append(tf.name)
        if not self.eng_fallback:
            raise ValueError(f"No vocabulary found in {self.folder}")

    def _register(self, raw_eng: str, raw_pos: str, raw_con: str, override: bool = False) -> None:
        cleaned_con = re.sub(r"[ˈˌ]", "", raw_con).strip()
        if not cleaned_con:
            return
        upos = POS_MAP.get(raw_pos.strip().lower(), "X")
        con = cleaned_con if upos == "PROPN" else cleaned_con.lower()
        defs = [d.strip() for d in raw_eng.split(";") if d.strip()]
        for idx, defn in enumerate(defs):
            eng = re.sub(r"\s*\([^)]*\)", "", defn)
            eng = re.sub(r"\s*\(.*$", "", eng).strip().lower()
            if not eng:
                continue
            if " " in eng:
                if override or eng not in self.eng_phrases:
                    self.eng_phrases[eng] = con
            else:
                if override or (eng, upos) not in self.eng_to_con:
                    self.eng_to_con[(eng, upos)] = con
                if override or eng not in self.eng_fallback:
                    self.eng_fallback[eng] = con
            if " " in con:
                if override or con not in self.con_phrases:
                    self.con_phrases[con] = eng
            elif override and idx == 0:
                self.con_to_eng[con] = (eng, upos)
            elif con not in self.con_to_eng:
                self.con_to_eng[con] = (eng, upos)

    def _parse_json(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_order = data.get("sentenceWordOrder", "S-V-IO-O").replace("-", "")
        if all(c in raw_order for c in ("S", "V", "O")):
            self.word_order = "".join(c for _, c in sorted((raw_order.index(c), c) for c in ("S", "V", "O")))
        self.adj_before_noun = data.get("adjectiveOrder", {}).get("radioID") == "beforeAdjOrder"
        tokens = [op.get("insert", "").strip() for op in data.get("grammarEditor", {}).get("ops", list()) if op.get("insert", "").strip()]
        lmap = {"Past": "VERB_PAST", "Future": "VERB_FUTURE", "Plural": "NOUN_PLUR", "Comparative": "ADJ_COMP", "Superlative": "ADJ_SUPER"}
        for i, tok in enumerate(tokens):
            if tok in lmap and i + 1 < len(tokens):
                affix = tokens[i + 1].lstrip("-").strip()
                if affix and affix != "-":
                    self.affixes[lmap[tok]] = affix
        for line in data.get("words", {}).get("value", "").splitlines():
            if "=" in line and ":" in line:
                left, con = line.split("=", 1)
                eng, pos = left.split(":", 1)
                self._register(eng, pos, con, False)

    def _parse_csv(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            if not next(reader, None):
                return
            for row in sorted([r for r in reader if len(r) >= 4], key=lambda r: (r[3].strip().lower(), r[0].strip().lower())):
                self._register(row[3], row[2], row[0], True)

    def _parse_html(self, path: Path) -> None:
        parser = _HTMLTableParser()
        parser.feed(path.read_text(encoding="utf-8"))
        for row in parser.rows:
            if len(row) >= 4 and row[1].startswith("/"):
                self._register(row[3], row[2], row[0], False)
            elif len(row) == 3:
                self._register(row[2], row[1], row[0], False)

    def _parse_yaml(self, path: Path) -> None:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if "word_order" in data:
            self.word_order = str(data["word_order"]).upper()
        if "adj_before_noun" in data:
            self.adj_before_noun = bool(data["adj_before_noun"])
        self.affixes.update(data.get("affixes", {}))
        for item in data.get("vocabulary", list()):
            self._register(item["english"], item.get("pos", "n"), item["conlang"], True)

    def _parse_txt(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8").strip()
        if text.startswith("{"):
            self._parse_json(path)
            return
        for line in text.splitlines():
            s = line.strip()
            if s and not s.startswith("#") and "=" in s and ":" in s:
                left, con = s.split("=", 1)
                eng, pos = left.split(":", 1)
                self._register(eng, pos, con, True)


class DynamicTranslator:
    EXTS = ("*.csv", "*.json", "*.html", "*.yaml", "*.yml", "*.txt")

    def __init__(
        self,
        languages_dir: str | Path = "data/languages",
        english_rules_path: str | Path = "data/english_rules.yaml",
    ):
        self.languages_dir = resolve_path(languages_dir)
        rules_file = resolve_path(english_rules_path)
        rules = yaml.safe_load(rules_file.read_text(encoding="utf-8")) if rules_file.exists() else {}
        self.contractions: dict[str, str] = rules.get("contractions", {})
        self.singular_triggers: set[str] = {w.lower() for w in rules.get("singular_triggers", list())}
        self.profiles: dict[str, LanguageProfile] = {}
        self.reload_languages()

    def reload_languages(self) -> list[str]:
        self.profiles.clear()
        if self.languages_dir.exists():
            for folder in sorted(self.languages_dir.iterdir()):
                if folder.is_dir() and any(list(folder.glob(ext)) for ext in self.EXTS):
                    self.profiles[folder.name.lower()] = LanguageProfile(folder.name.lower(), folder)
        return sorted(self.profiles.keys())

    def available_languages(self) -> list[str]:
        return ["english"] + sorted(self.profiles.keys())

    def translate(self, text: str, src_lang: str, tgt_lang: str, tagger_fn) -> str:
        src = src_lang.strip().lower()
        tgt = tgt_lang.strip().lower()
        if src == tgt:
            return text
        if src != "english" and tgt != "english":
            english_bridge = self._con_to_eng(text, src)
            return str(self._eng_to_con(english_bridge, tgt, tagger_fn))
        if src == "english":
            return str(self._eng_to_con(text, tgt, tagger_fn))
        return self._con_to_eng(text, src)

    def _eng_to_con(self, text: str, target: str, tagger_fn, return_trace: bool = False):
        if target not in self.profiles:
            raise ValueError(f"Unknown language: {target}")
        prof = self.profiles[target]
        tagged = tagger_fn(self._norm_eng(text))
        if not tagged:
            return ([], "") if return_trace else ""

        rules = getattr(self, "english_rules", getattr(self, "rules", {}))
        inv_verbs = {w.lower() for w in rules.get("invariant_verbs", ["cast", "hit", "cut", "set", "hurt"])}
        guarded = set()
        for idx in range(1, len(tagged) - 1):
            if (
                tagged[idx]["word"].lower() in inv_verbs
                and tagged[idx]["upos"] == "NOUN"
                and tagged[idx - 1]["upos"] in {"NOUN", "PROPN", "PRON"}
                and tagged[idx + 1]["upos"] in {"DET", "ADJ", "NOUN", "PROPN", "PRON"}
            ):
                tagged[idx].update({"upos": "VERB", "morph": "Tense=Past", "label": "VERB|Tense=Past"})
                guarded.add(idx)

        orig_tagged = [dict(t) for t in tagged]
        if not prof.adj_before_noun:
            tagged = self._swap_adj(tagged)

        lemmas_table = yaml.safe_load(resolve_path("data/lemmas.yaml").read_text(encoding="utf-8")) if return_trace else {}
        proper_nouns = getattr(prof, "proper_nouns", getattr(prof, "propn", {}))

        def _process(stream: list[dict], record: bool) -> tuple[list[dict], list[str]]:
            out, rows, i, n = [], [], 0, len(stream)
            while i < n:
                matched = False
                for span in (4, 3, 2):
                    if i + span <= n:
                        chunk = " ".join(t["word"].lower() for t in stream[i : i + span])
                        if chunk in prof.eng_phrases:
                            w = self._match_case(stream[i]["word"], prof.eng_phrases[chunk])
                            out.append(w)
                            if record:
                                rows.append({
                                    "token": chunk, "tag": f"PHRASE({span}w)", "s0": "SKIP", "s1": "CAUGHT",
                                    "s2": "SKIP", "s3": "SKIP", "s4": "SKIP", "s5": "SKIP", "output": w.lower(),
                                })
                            i += span
                            matched = True
                            break
                if matched:
                    continue

                tok = stream[i]
                raw, low, upos, morph, label = tok["word"], tok["word"].lower(), tok["upos"], tok.get("morph", ""), tok.get("label", tok["upos"])
                if upos == "PUNCT":
                    if out:
                        out[-1] += raw
                    else:
                        out.append(raw)
                    i += 1
                    continue

                lookup_pos = "DET" if (upos == "PRON" and "Poss=Yes" in morph) else upos
                hit = prof.eng_to_con.get((low, lookup_pos)) or prof.eng_to_con.get((low, upos))
                lemma = tok.get("lemma") or lemmatize_token(raw, upos)

                if hit:
                    root = hit
                    w = self._inflect(hit, upos, morph, prof) if morph else hit
                else:
                    root = (
                        prof.eng_to_con.get((lemma.lower(), lookup_pos))
                        or prof.eng_to_con.get((lemma.lower(), upos))
                        or prof.eng_fallback.get(lemma.lower())
                        or prof.eng_fallback.get(low)
                        or raw
                    )
                    w = self._inflect(root, upos, morph, prof) if root != raw else raw

                w_cased = self._match_case(raw, w)
                out.append(w_cased)

                if record:
                    s0 = "FIXED" if i in guarded else "MISS"
                    disp_out = w_cased if (raw in proper_nouns or upos == "PROPN") else w.lower()
                    if raw in proper_nouns or upos == "PROPN":
                        s2 = "LORE MAP" if (raw in proper_nouns or root != raw) else "OOV KEPT"
                        rows.append({
                            "token": raw, "tag": label, "s0": s0, "s1": "MISS",
                            "s2": s2, "s3": "SKIP", "s4": "SKIP", "s5": "SKIP", "output": disp_out,
                        })
                    else:
                        s3 = f"UD:{lemmas_table[upos][low]}" if low in lemmas_table.get(upos, {}) else (f"STRIP:{lemma}" if lemma != low else f"ROOT:{lemma}")
                        s4 = f"'{root.lower()}'" if (root.lower() != lemma.lower() or low in {"a", "an"}) else "PASSTHRU"
                        sfx = w.lower()[len(root.lower()):] if (len(w) > len(root) and w.lower().startswith(root.lower())) else ""
                        s5 = "NONE"
                        for fk, ak in [("Number=Plur", "PLUR"), ("Tense=Past", "PAST"), ("Tense=Fut", "FUT"), ("Degree=Cmp", "COMP"), ("Degree=Sup", "SUPER")]:
                            if fk in morph:
                                s5 = f"+{ak}(-{sfx})" if sfx else f"+{ak}"
                                break
                        rows.append({
                            "token": raw, "tag": label, "s0": s0, "s1": "MISS",
                            "s2": "MISS", "s3": s3, "s4": s4, "s5": s5, "output": disp_out,
                        })
                i += 1
            return rows, out

        _, final_out = _process(tagged, record=False)
        final_str = self._apply_sentence_case(self._clean_punctuation(" ".join(final_out)))
        if return_trace:
            trace_rows, _ = _process(orig_tagged, record=True)
            return trace_rows, final_str
        return final_str

    def _con_to_eng(self, text: str, source: str) -> str:
        if source not in self.profiles:
            raise ValueError(f"Unknown language: {source}")
        prof = self.profiles[source]
        tokens = re.findall(r"\w+|[^\w\s]", text)
        out, i, n = list(), 0, len(tokens)
        prev_word, prev_propn = "", False
        while i < n:
            matched = False
            for span in (3, 2):
                if i + span <= n:
                    chunk = " ".join(t.lower() for t in tokens[i : i + span])
                    if chunk in prof.con_phrases:
                        w = prof.con_phrases[chunk]
                        out.append(self._match_case(tokens[i], w))
                        prev_word, prev_propn = w, False
                        i += span
                        matched = True
                        break
            if matched:
                continue
            tok = tokens[i]
            if not re.match(r"\w+", tok):
                if out:
                    out[-1] += tok
                else:
                    out.append(tok)
                prev_word, prev_propn = "", False
                i += 1
                continue
            low = tok.lower()
            if tok in prof.con_to_eng:
                w, pos = prof.con_to_eng[tok]
                is_propn = pos == "PROPN"
            else:
                w, pos = self._rev_token(low, prof)
                is_propn = pos == "PROPN"
            if pos == "VERB":
                w = self._agree_third_person_verb(w, prev_word, prev_propn)
            w = self._match_case(tok, w)
            out.append(w)
            prev_word, prev_propn = w, is_propn
            i += 1
        result = self._clean_punctuation(" ".join(out))
        result = re.sub(r"\b(\w+)'s\s+(\w+)\b", r"\2 of \1", result)
        return self._apply_sentence_case(result)
   
    def _agree_third_person_verb(self, verb_lemma: str, prev_eng: str, prev_is_propn: bool) -> str:
        if prev_eng.lower() in self.singular_triggers or prev_is_propn:
            if verb_lemma == "have":
                return "has"
            if verb_lemma == "be":
                return "is"
            if verb_lemma.endswith(("s", "x", "z", "ch", "sh", "o")):
                return f"{verb_lemma}es"
            if verb_lemma.endswith("y") and len(verb_lemma) > 2 and verb_lemma[-2] not in "aeiou":
                return f"{verb_lemma[:-1]}ies"
            return f"{verb_lemma}s"
        return verb_lemma

    @staticmethod
    def _rev_token(tok: str, prof: LanguageProfile) -> tuple[str, str]:
        if tok in prof.con_to_eng:
            return prof.con_to_eng[tok]
        rules = [
            ("NOUN_PLUR", "NOUN", "PLUR"),
            ("VERB_PAST", "VERB", "PAST"),
            ("VERB_FUTURE", "VERB", "FUT"),
            ("VERB_PROG", "VERB", "PROG"),
            ("ADJ_COMP", "ADJ", "COMP"),
            ("ADJ_SUPER", "ADJ", "SUPER"),
        ]
        fallback = None
        for key, req_pos, feat in rules:
            sfx = prof.affixes.get(key, "")
            if sfx and tok.endswith(sfx) and len(tok) > len(sfx):
                stem = tok[: -len(sfx)]
                if stem in prof.con_to_eng:
                    lemma, stem_pos = prof.con_to_eng[stem]
                    if stem_pos == req_pos:
                        return inflect_english(lemma, req_pos, feat), "INFLECTED"
                    fallback = (lemma, stem_pos)
        return fallback or (tok, "X")

    @staticmethod
    def _inflect(root: str, upos: str, morph: str, prof: LanguageProfile) -> str:
        checks = [
            (upos == "NOUN" and "Plur" in morph, "NOUN_PLUR"),
            (upos == "VERB" and "Past" in morph, "VERB_PAST"),
            (upos == "VERB" and "Fut" in morph, "VERB_FUTURE"),
            (upos == "VERB" and ("Part" in morph or "Ger" in morph), "VERB_PROG"),
            (upos == "ADJ" and "Cmp" in morph, "ADJ_COMP"),
            (upos == "ADJ" and "Sup" in morph, "ADJ_SUPER"),
        ]
        for cond, key in checks:
            if cond and key in prof.affixes:
                return f"{root}{prof.affixes[key]}"
        return root

    @staticmethod
    def _swap_adj(tagged: list[dict]) -> list[dict]:
        res, i = list(tagged), 0
        while i < len(res) - 1:
            if res[i]["upos"] == "ADJ" and res[i + 1]["upos"] == "NOUN":
                res[i], res[i + 1] = res[i + 1], res[i]
                i += 2
            else:
                i += 1
        return res

    @staticmethod
    def _match_case(source_word: str, target_word: str) -> str:
        if not target_word:
            return target_word
        if len(source_word) > 1 and source_word.isupper():
            return target_word.upper()
        if source_word and source_word[0].isupper():
            return " ".join(part.capitalize() for part in target_word.split(" "))
        return target_word

    @staticmethod
    def _apply_sentence_case(text: str) -> str:
        return re.sub(
            r"(^|[.!?]\s+)(['\"“‘]?)([a-z])",
            lambda m: f"{m.group(1)}{m.group(2)}{m.group(3).upper()}",
            text,
        )

    @staticmethod
    def _clean_punctuation(text: str) -> str:
        text = re.sub(r"\s*-\s*", "-", text)
        text = re.sub(r"\s*[—–]\s*", "—", text)
        text = re.sub(r"(\d),\s+(\d)", r"\1,\2", text)
        text = re.sub(r":'\s+", ": '", text)
        return text

    def _norm_eng(self, text: str) -> str:
        text = re.sub(
            r"\b([A-Za-z]+)'s\s+([A-Za-z]+)\b",
            lambda m: f"{m.group(2)} of {m.group(1)}",
            text,
        )
        for pat, rep in self.contractions.items():
            text = re.sub(pat, rep, text, flags=re.IGNORECASE)
        return text
