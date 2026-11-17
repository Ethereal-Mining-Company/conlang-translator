import json
import re
import time
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from torchcrf import CRF

from src.utils import append_csv_row, make_run_uuid, resolve_path, setup_logger

MAX_CHAR_LEN = 20


def _encode_chars(word: str, c2i: dict[str, int]) -> list[int]:
    ids = [c2i.get(ch, 1) for ch in word[:MAX_CHAR_LEN]]
    return ids + [0] * (MAX_CHAR_LEN - len(ids))


class _TagDataset(Dataset):
    def __init__(self, sents, w2i, c2i, t2i):
        self.data = [
            (
                [w2i.get(t["word"].lower(), 1) for t in s],
                [_encode_chars(t["word"], c2i) for t in s],
                [t2i[t["label"]] for t in s],
            )
            for s in sents if s
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _collate(batch):
    xs = [torch.tensor(x, dtype=torch.long) for x, _, _ in batch]
    cs = [torch.tensor(c, dtype=torch.long) for _, c, _ in batch]
    ys = [torch.tensor(y, dtype=torch.long) for _, _, y in batch]
    masks = [torch.ones(len(x), dtype=torch.bool) for x, _, _ in batch]
    return (
        nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0),
        nn.utils.rnn.pad_sequence(cs, batch_first=True, padding_value=0),
        nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0),
        nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=False),
    )


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size: int, char_vocab_size: int, tagset_size: int, emb_dim: int, hidden_dim: int, dropout: float, char_dim: int = 32):
        super().__init__()
        self.char_dim = char_dim
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if char_dim > 0:
            self.char_emb = nn.Embedding(char_vocab_size, char_dim, padding_idx=0)
            self.char_conv = nn.Conv1d(char_dim, char_dim, kernel_size=3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim + char_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def _emissions(self, x, chars):
        w_feat = self.emb(x)
        if self.char_dim > 0:
            B, T, C = chars.shape
            c_emb = self.char_emb(chars.view(B * T, C)).transpose(1, 2)
            c_feat = torch.relu(self.char_conv(c_emb)).max(dim=-1).values.view(B, T, -1)
            combined = self.drop(torch.cat([w_feat, c_feat], dim=-1))
        else:
            combined = self.drop(w_feat)
        lstm_out, _ = self.lstm(combined)
        return self.fc(self.drop(lstm_out))

    def forward(self, x, chars, tags, mask):
        return -self.crf(self._emissions(x, chars), tags, mask=mask, reduction="mean")

    def decode(self, x, chars, mask):
        return self.crf.decode(self._emissions(x, chars), mask=mask)


def _eval(model, loader, device, idx2tag):
    model.eval()
    yt: list[str] = list()
    yp: list[str] = list()
    with torch.no_grad():
        for x, c, y, m in loader:
            x, c, m = x.to(device), c.to(device), m.to(device)
            for pred_seq, true_seq, mask_seq in zip(model.decode(x, c, m), y.tolist(), m.tolist()):
                valid = sum(mask_seq)
                yt.extend(idx2tag[t] for t in true_seq[:valid])
                yp.extend(idx2tag[p] for p in pred_seq[:valid])
    return yt, yp


def _fit_single(tr_sents, va_sents, w2i, c2i, t2i, i2t, run: dict, mcfg: dict, cfg: dict, run_uuid: str, run_name: str, fold: int, device, logger):
    emb_dim = run.get("emb_dim", 128)
    hidden_dim = run.get("hidden_dim", 256)
    char_dim = run.get("char_dim", 32)
    epochs = run.get("epochs", 6)
    lr = run.get("lr", 0.003)
    dropout = run.get("dropout", mcfg.get("dropout", 0.3))
    batch_size = run.get("batch_size", mcfg.get("batch_size", 32))

    tr_loader = DataLoader(_TagDataset(tr_sents, w2i, c2i, t2i), batch_size=batch_size, shuffle=True, collate_fn=_collate)
    va_loader = DataLoader(_TagDataset(va_sents, w2i, c2i, t2i), batch_size=mcfg["eval_batch_size"], collate_fn=_collate)

    model = BiLSTMCRF(len(w2i), len(c2i), len(t2i), emb_dim, hidden_dim, dropout, char_dim=char_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    val_acc, avg_loss = 0.0, 0.0

    for ep in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        tot_loss, steps = 0.0, 0
        for x, c, y, m in tr_loader:
            x, c, y, m = x.to(device), c.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            loss = model(x, c, y, m)
            loss.backward()
            opt.step()
            tot_loss += float(loss.item())
            steps += 1
        avg_loss = round(tot_loss / max(steps, 1), 6)
        yt, yp = _eval(model, va_loader, device, i2t)
        val_acc = float(accuracy_score(yt, yp))
        ep_sec = round(time.perf_counter() - ep_t0, 2)
        logger.info(f"[{run_uuid}] {run_name} | fold={fold} | ep={ep}/{epochs} | loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

        append_csv_row(cfg["paths"]["epochs_csv"], {
            "run_uuid": run_uuid,
            "engine_type": "bilstm_crf",
            "run_name": run_name,
            "fold": fold,
            "epoch": ep,
            "train_loss": avg_loss,
            "val_accuracy": val_acc,
            "elapsed_sec": ep_sec,
        })

    full_params = {"emb_dim": emb_dim, "hidden_dim": hidden_dim, "char_dim": char_dim, "epochs": epochs, "lr": lr, "dropout": dropout, "batch_size": batch_size}
    return model, val_acc, avg_loss, full_params


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
    mcfg = cfg["models"]["bilstm_crf"]
    seed = cfg["runtime"]["random_seed"]
    folds = cv_folds if cv_folds is not None else cfg["runtime"].get("cv_folds", 1)
    runs = runs_override if runs_override is not None else mcfg["runs"]
    if not runs:
        raise ValueError("No training runs provided for bilstm_crf.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("bilstm_trainer", cfg["paths"]["logs"], mcfg["log_file"])

    w2i = {"<PAD>": 0, "<UNK>": 1}
    c2i = {"<PAD>": 0, "<UNK>": 1}
    t2i = {}
    for s in train_s:
        for tok in s:
            raw_w = tok["word"]
            low_w = raw_w.lower()
            if low_w not in w2i:
                w2i[low_w] = len(w2i)
            for ch in raw_w:
                if ch not in c2i:
                    c2i[ch] = len(c2i)
            if tok["label"] not in t2i:
                t2i[tok["label"]] = len(t2i)
    i2t = {v: k for k, v in t2i.items()}

    ts_loader = DataLoader(_TagDataset(test_s, w2i, c2i, t2i), batch_size=mcfg["eval_batch_size"], collate_fn=_collate)
    best_model: BiLSTMCRF | None = None
    best_acc = -1.0
    best_run: dict | None = None
    best_uuid = ""
    best_params: dict = {}
    sweep: list[dict] = list()

    for idx, run in enumerate(runs, 1):
        run_uuid = make_run_uuid()
        run_name = run.get("name", f"BiLSTM_Run_{idx}")
        t0 = time.perf_counter()
        fold_scores: list[float] = list()

        if folds > 1:
            kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
            for f_idx, (tr_idx, va_idx) in enumerate(kf.split(train_s), 1):
                _, f_acc, _, _ = _fit_single(
                    [train_s[i] for i in tr_idx], [train_s[i] for i in va_idx],
                    w2i, c2i, t2i, i2t, run, mcfg, cfg, run_uuid, run_name, f_idx, device, logger
                )
                fold_scores.append(f_acc)
            model, _, avg_loss, full_params = _fit_single(
                train_s, dev_s, w2i, c2i, t2i, i2t, run, mcfg, cfg, run_uuid, run_name, 0, device, logger
            )
            val_acc = float(sum(fold_scores) / len(fold_scores))
        else:
            model, val_acc, avg_loss, full_params = _fit_single(
                train_s, dev_s, w2i, c2i, t2i, i2t, run, mcfg, cfg, run_uuid, run_name, 1, device, logger
            )
            fold_scores = [val_acc]

        elapsed = round(time.perf_counter() - t0, 2)
        tuned = {k: v for k, v in run.items() if k != "name"}
        sweep.append({
            "run_uuid": run_uuid,
            "name": run_name,
            "tuned_params": tuned,
            "val_accuracy": val_acc,
            "val_loss": avg_loss,
            "fold_scores": fold_scores,
            "time_s": elapsed,
            "all_params": full_params,
        })
        if val_acc > best_acc:
            best_acc, best_model, best_run, best_uuid, best_params = val_acc, model, {**run, "name": run_name}, run_uuid, full_params

    if best_model is None or best_run is None:
        raise RuntimeError("Training completed without selecting a valid model.")

    t0 = time.perf_counter()
    y_test, y_pred = _eval(best_model, ts_loader, device, i2t)
    latency_ms = ((time.perf_counter() - t0) / max(len(test_s), 1)) * 1000.0
    test_acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    labels = sorted(list(set(y_test)))
    top_labels = sorted(labels, key=lambda l: y_test.count(l), reverse=True)[: cfg["runtime"]["confusion_matrix_top_k"]]
    cm = confusion_matrix(y_test, y_pred, labels=top_labels).tolist()

    promoted = bool(save_artifacts and best_acc > champion_val_acc)
    metadata = {
        "run_uuid": best_uuid,
        "engine_type": "bilstm_crf",
        "best_run_name": best_run["name"],
        "promoted": promoted,
        "global_seed": seed,
        "cv_folds": folds,
        "vocab_size": len(w2i),
        "char_vocab_size": len(c2i),
        "target_classes": list(i2t.values()),
        "full_winning_parameters": best_params,
    }

    for entry in sweep:
        is_sweep_winner = entry["run_uuid"] == best_uuid
        append_csv_row(cfg["paths"]["runs_csv"], {
            "run_uuid": entry["run_uuid"],
            "engine_type": "bilstm_crf",
            "run_name": entry["name"],
            "promoted": promoted if is_sweep_winner else False,
            "tuned_params": str(entry["tuned_params"]),
            "mean_cv_accuracy": entry["val_accuracy"],
            "mean_cv_loss": entry["val_loss"],
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
        torch.save({
            "state_dict": best_model.state_dict(),
            "word2idx": w2i,
            "char2idx": c2i,
            "idx2tag": i2t,
            "emb_dim": best_params["emb_dim"],
            "hidden_dim": best_params["hidden_dim"],
            "char_dim": best_params["char_dim"],
            "dropout": best_params["dropout"],
            "metadata": metadata,
        }, out_path)
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
    art = resolve_path(cfg["paths"]["artifacts_prod"]) / cfg["models"]["bilstm_crf"]["artifact_name"]
    ckpt = torch.load(art, map_location="cpu")
    w2i = ckpt["word2idx"]
    c2i = ckpt.get("char2idx", {"<PAD>": 0, "<UNK>": 1})
    char_dim = ckpt.get("char_dim", 0)
    i2t = {int(k): v for k, v in ckpt["idx2tag"].items()}
    model = BiLSTMCRF(len(w2i), len(c2i), len(i2t), ckpt["emb_dim"], ckpt["hidden_dim"], ckpt["dropout"], char_dim=char_dim)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    def predict_fn(text: str) -> list[dict]:
        words = re.findall(r"\w+|[^\w\s]", text)
        if not words:
            return list()
        ids = torch.tensor([[w2i.get(w.lower(), 1) for w in words]], dtype=torch.long)
        chars = torch.tensor([[_encode_chars(w, c2i) for w in words]], dtype=torch.long)
        mask = torch.ones_like(ids, dtype=torch.bool)
        with torch.no_grad():
            pred_ids = model.decode(ids, chars, mask)
        out: list[dict] = list()
        for w, pid in zip(words, pred_ids[0]):
            lbl = i2t[pid]
            parts = lbl.split("|", 1)
            out.append({"word": w, "upos": parts[0], "morph": parts[1] if len(parts) > 1 else "", "label": lbl})
        return out

    return predict_fn
