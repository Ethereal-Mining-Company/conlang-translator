import re
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from torchcrf import CRF

from src.models import register_model
from src.models.base import BaseTagger
from src.utils import lemmatize_token, resolve_path, set_seed, setup_logger
from src.visualizations import plot_top_confusion_matrix, plot_tuning_comparison, save_metrics_record


class _SeqDataset(Dataset):
    def __init__(self, sentences: list[list[dict]], w2i: dict, t2i: dict):
        self.samples = [
            (torch.tensor([w2i.get(t["word"].lower(), w2i["<UNK>"]) for t in s]), torch.tensor([t2i[t["label"]] for t in s]))
            for s in sentences
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _collate(batch):
    words, tags = zip(*batch)
    max_len = max(len(w) for w in words)
    pw = torch.zeros(len(words), max_len, dtype=torch.long)
    pt = torch.zeros(len(tags), max_len, dtype=torch.long)
    mask = torch.zeros(len(words), max_len, dtype=torch.bool)
    for i, (w, t) in enumerate(zip(words, tags)):
        pw[i, : len(w)], pt[i, : len(w)], mask[i, : len(w)] = w, t, True
    return pw, pt, mask


class _BiLSTMCRFNet(nn.Module):
    def __init__(self, vocab_size: int, tagset_size: int, emb_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, words, tags=None, mask=None):
        emissions = self.fc(self.dropout(self.lstm(self.dropout(self.embedding(words)))[0]))
        return -self.crf(emissions, tags, mask=mask, reduction="mean") if tags is not None else self.crf.decode(emissions, mask=mask)


@register_model("bilstm_crf")
class BiLSTMCRFTagger(BaseTagger):
    def __init__(self, cfg: dict, model_key: str = "bilstm_crf"):
        super().__init__(cfg, model_key)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        self.w2i: dict = {}
        self.i2t: dict = {}

    def _eval(self, model, loader, i2t):
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for words, tags, mask in loader:
                preds = model(words.to(self.device), mask=mask.to(self.device))
                for p_seq, t_seq, m_seq in zip(preds, tags, mask):
                    ln = int(m_seq.sum().item())
                    y_pred.extend(i2t[i] for i in p_seq[:ln])
                    y_true.extend(i2t[int(i)] for i in t_seq[:ln])
        return accuracy_score(y_true, y_pred), y_true, y_pred

    def train(self, train_sents: list, dev_sents: list, test_sents: list) -> None:
        rt = self.cfg["runtime"]
        set_seed(rt["random_seed"], rt["deterministic"])
        logger = setup_logger("BiLSTMTrainer", self.cfg["paths"]["logs"], self.mcfg["log_file"])

        w2i, t2i = {"<PAD>": 0, "<UNK>": 1}, {}
        for s in train_sents:
            for tok in s:
                w2i.setdefault(tok["word"].lower(), len(w2i))
                t2i.setdefault(tok["label"], len(t2i))
        for s in dev_sents + test_sents:
            for tok in s:
                t2i.setdefault(tok["label"], len(t2i))
        i2t = {v: k for k, v in t2i.items()}

        g = torch.Generator().manual_seed(rt["random_seed"])
        tr_ld = DataLoader(_SeqDataset(train_sents, w2i, t2i), batch_size=self.mcfg["batch_size"], shuffle=True, generator=g, collate_fn=_collate)
        dv_ld = DataLoader(_SeqDataset(dev_sents, w2i, t2i), batch_size=self.mcfg["eval_batch_size"], collate_fn=_collate)
        te_ld = DataLoader(_SeqDataset(test_sents, w2i, t2i), batch_size=self.mcfg["eval_batch_size"], collate_fn=_collate)

        runs, best_state, best_hp, best_acc = [], None, None, -1.0
        for r in self.mcfg["runs"]:
            set_seed(rt["random_seed"], rt["deterministic"])
            t0 = time.perf_counter()
            net = _BiLSTMCRFNet(len(w2i), len(t2i), r["emb_dim"], r["hidden_dim"], self.mcfg["dropout"]).to(self.device)
            opt = torch.optim.Adam(net.parameters(), lr=r["lr"])
            val_acc = 0.0
            for ep in range(1, r["epochs"] + 1):
                net.train()
                tot = 0.0
                for words, tags, mask in tr_ld:
                    opt.zero_grad()
                    loss = net(words.to(self.device), tags.to(self.device), mask.to(self.device))
                    loss.backward()
                    opt.step()
                    tot += loss.item()
                val_acc, _, _ = self._eval(net, dv_ld, i2t)
                logger.info(f"[{r['name']}] Ep {ep}/{r['epochs']} | Loss: {tot/len(tr_ld):.4f} | Val Acc: {val_acc:.4f}")
            elapsed = time.perf_counter() - t0
            runs.append({"Model": r["name"], "Validation Accuracy": round(val_acc, 4), "Train Seconds": round(elapsed, 2)})
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.cpu() for k, v in net.state_dict().items()}
                best_hp = {"emb_dim": r["emb_dim"], "hidden_dim": r["hidden_dim"], "dropout": self.mcfg["dropout"]}

        self.w2i, self.i2t = w2i, i2t
        self.net = _BiLSTMCRFNet(len(w2i), len(t2i), **best_hp).to(self.device)
        self.net.load_state_dict(best_state)

        t0 = time.perf_counter()
        test_acc, y_true, y_pred = self._eval(self.net, te_ld, i2t)
        inf_ms = ((time.perf_counter() - t0) / len(test_sents)) * 1000.0
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        logger.info(f"Best {self.mcfg['display_name']} Test Acc: {test_acc:.4f} | Macro F1: {macro_f1:.4f} | {inf_ms:.3f} ms/sent")
        logger.info("\n" + classification_report(y_true, y_pred, zero_division=0))

        prod_dir = resolve_path(self.cfg["paths"]["artifacts_prod"])
        prod_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "word2idx": w2i, "idx2tag": i2t, **best_hp}, prod_dir / self.mcfg["artifact_name"])
        save_metrics_record(prod_dir / rt["metrics_file"], self.mcfg["display_name"], runs, {"test_accuracy": round(test_acc, 4), "macro_f1": round(macro_f1, 4), "latency_ms": round(inf_ms, 3)})
        plot_tuning_comparison(runs, f"{self.mcfg['display_name']} Tuning", self.cfg["paths"]["figures"], f"{self.model_key}_tuning")
        plot_top_confusion_matrix(y_true, y_pred, f"{self.mcfg['display_name']} Confusion Matrix", self.cfg["paths"]["figures"], f"{self.model_key}_cm", rt["confusion_matrix_top_k"])

    def load(self, artifact_path: Path) -> "BiLSTMCRFTagger":
        ckpt = torch.load(artifact_path, map_location=self.device)
        self.w2i, self.i2t = ckpt["word2idx"], ckpt["idx2tag"]
        self.net = _BiLSTMCRFNet(len(self.w2i), len(self.i2t), ckpt["emb_dim"], ckpt["hidden_dim"], ckpt["dropout"]).to(self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        return self

    def __call__(self, text: str) -> list[dict]:
        tokens = re.findall(r"\w+|[^\w\s]", text)
        if not tokens:
            return []
        w_ids = torch.tensor([[self.w2i.get(w.lower(), self.w2i["<UNK>"]) for w in tokens]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            pred_ids = self.net(w_ids, mask=torch.ones_like(w_ids, dtype=torch.bool))[0]
        return [
            {"word": w, "lemma": lemmatize_token(w, self.i2t[tid].split("|", 1)[0]), "upos": self.i2t[tid].split("|", 1)[0], "morph": self.i2t[tid].split("|", 1)[1] if "|" in self.i2t[tid] else ""}
            for w, tid in zip(tokens, pred_ids)
        ]