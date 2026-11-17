import re
import time
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from src.features import extract_token_features
from src.models import register_model
from src.models.base import BaseTagger
from src.utils import lemmatize_token, resolve_path, set_seed, setup_logger
from src.visualizations import plot_top_confusion_matrix, plot_tuning_comparison, save_metrics_record


@register_model("random_forest")
class RandomForestTagger(BaseTagger):
    def __init__(self, cfg: dict, model_key: str = "random_forest"):
        super().__init__(cfg, model_key)
        self.pipe = None

    @staticmethod
    def _xy(sentences: list[list[dict]]) -> tuple[list[dict], list[str]]:
        X, y = [], []
        for sent in sentences:
            for i in range(len(sent)):
                X.append(extract_token_features(sent, i))
                y.append(sent[i]["label"])
        return X, y

    def train(self, train_sents: list, dev_sents: list, test_sents: list) -> None:
        rt = self.cfg["runtime"]
        set_seed(rt["random_seed"], rt["deterministic"])
        logger = setup_logger("RFTrainer", self.cfg["paths"]["logs"], self.mcfg["log_file"])

        X_tr, y_tr = self._xy(train_sents)
        X_dv, y_dv = self._xy(dev_sents)
        X_te, y_te = self._xy(test_sents)

        runs, best_pipe, best_acc = [], None, -1.0
        for r in self.mcfg["runs"]:
            t0 = time.perf_counter()
            pipe = Pipeline([
                ("vec", DictVectorizer(sparse=True)),
                ("clf", RandomForestClassifier(n_estimators=r["n_estimators"], max_depth=r["max_depth"], n_jobs=-1, random_state=rt["random_seed"])),
            ])
            pipe.fit(X_tr, y_tr)
            elapsed = time.perf_counter() - t0
            acc = accuracy_score(y_dv, pipe.predict(X_dv))
            logger.info(f"{r['name']} -> Val Acc: {acc:.4f} ({elapsed:.1f}s)")
            runs.append({"Model": r["name"], "Validation Accuracy": round(acc, 4), "Train Seconds": round(elapsed, 2)})
            if acc > best_acc:
                best_acc, best_pipe = acc, pipe

        self.pipe = best_pipe
        t0 = time.perf_counter()
        preds = self.pipe.predict(X_te)
        inf_ms = ((time.perf_counter() - t0) / len(test_sents)) * 1000.0
        test_acc = accuracy_score(y_te, preds)
        macro_f1 = f1_score(y_te, preds, average="macro", zero_division=0)
        logger.info(f"Best {self.mcfg['display_name']} Test Acc: {test_acc:.4f} | Macro F1: {macro_f1:.4f} | {inf_ms:.3f} ms/sent")
        logger.info("\n" + classification_report(y_te, preds, zero_division=0))

        prod_dir = resolve_path(self.cfg["paths"]["artifacts_prod"])
        prod_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipe, prod_dir / self.mcfg["artifact_name"])
        save_metrics_record(prod_dir / rt["metrics_file"], self.mcfg["display_name"], runs, {"test_accuracy": round(test_acc, 4), "macro_f1": round(macro_f1, 4), "latency_ms": round(inf_ms, 3)})
        plot_tuning_comparison(runs, f"{self.mcfg['display_name']} Tuning", self.cfg["paths"]["figures"], f"{self.model_key}_tuning")
        plot_top_confusion_matrix(y_te, preds, f"{self.mcfg['display_name']} Confusion Matrix", self.cfg["paths"]["figures"], f"{self.model_key}_cm", rt["confusion_matrix_top_k"])

    def load(self, artifact_path: Path) -> "RandomForestTagger":
        self.pipe = joblib.load(artifact_path)
        return self

    def __call__(self, text: str) -> list[dict]:
        tokens = re.findall(r"\w+|[^\w\s]", text)
        if not tokens:
            return []
        sent = [{"word": w} for w in tokens]
        preds = self.pipe.predict([extract_token_features(sent, i) for i in range(len(sent))])
        return [
            {"word": w, "lemma": lemmatize_token(w, lbl.split("|", 1)[0]), "upos": lbl.split("|", 1)[0], "morph": lbl.split("|", 1)[1] if "|" in lbl else ""}
            for w, lbl in zip(tokens, preds)
        ]