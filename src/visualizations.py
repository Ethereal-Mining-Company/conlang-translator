import json
import numpy as np
import torch
import yaml
import joblib
from pathlib import Path
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.utils import lemmatize_token, resolve_path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix


def _save_fig(fig, out_dir: str | Path, filename: str):
    out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.write_html(out / f"{filename}.html")
    try:
        fig.write_image(out / f"{filename}.png", scale=2)
    except Exception:
        pass
    return fig


def save_metrics_record(metrics_path: str | Path, family: str, runs: list[dict], test_metrics: dict) -> None:
    path = resolve_path(metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[family] = {"tuning_runs": runs, "best_test_metrics": test_metrics}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def plot_tuning_comparison(runs: list[dict], title: str, out_dir: str | Path, filename: str):
    fig = px.bar(pd.DataFrame(runs), x="Model", y="Validation Accuracy", color="Model", title=title, text_auto=True)
    fig.update_traces(texttemplate="%{y:.4f}", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_range=[0, 1.05])
    return _save_fig(fig, out_dir, filename)


def plot_model_head_to_head(metrics_path: str | Path, out_dir: str | Path):
    data = json.loads(resolve_path(metrics_path).read_text(encoding="utf-8"))
    rows = [
        {"Model Family": k, "Test Accuracy": v["best_test_metrics"]["test_accuracy"], "Macro F1": v["best_test_metrics"]["macro_f1"]}
        for k, v in data.items()
    ]
    fig = px.bar(pd.DataFrame(rows), x="Model Family", y=["Test Accuracy", "Macro F1"], barmode="group", title="Model Test Performance Comparison", text_auto=True)
    fig.update_traces(texttemplate="%{y:.4f}", textposition="outside")
    fig.update_layout(yaxis_range=[0, 1.08])
    return _save_fig(fig, out_dir, "model_comparison")


def plot_top_confusion_matrix(y_true: list[str], y_pred: list[str], title: str, out_dir: str | Path, filename: str, top_k: int = 12):
    labels = pd.Series(y_true).value_counts().head(top_k).index.tolist()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues", title=title, labels={"x": "Predicted", "y": "True", "color": "Count"})
    return _save_fig(fig, out_dir, filename)


def export_conveyor_pptx_visual(
    sentence: str,
    cfg: dict,
    translator,
    tagger,
    model_name: str,
    target_lang: str = "aelorian",
    out_file: str = "figures/conveyor_belt_pptx.png",
):
    trace_rows, final_out = translator._eng_to_con(sentence, target_lang, tagger, return_trace=True)
    badge_colors = {
        "FIXED": ("#d69e2e", "white"),
        "CAUGHT": ("#2b6cb0", "white"),
        "LORE MAP": ("#6b46c1", "white"),
        "OOV KEPT": ("#6b46c1", "white"),
        "PASSTHRU": ("#fed7d7", "#9b2c2c"),
        "NONE": ("#edf2f7", "#718096"),
        "MISS": ("#edf2f7", "#718096"),
        "SKIP": ("#f7fafc", "#cbd5e0"),
    }

    def _style(txt: str) -> tuple[str, str, str]:
        if txt in badge_colors:
            return txt, *badge_colors[txt]
        if txt.startswith(("UD:", "STRIP:")):
            return txt, "#319795", "white"
        if txt.startswith("ROOT:"):
            return txt, "#e2e8f0", "#2d3748"
        if txt.startswith("+"):
            return txt, "#c05621", "white"
        return txt, "#2f855a", "white"

    fig, ax = plt.subplots(figsize=(15, 0.72 * len(trace_rows) + 3.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, len(trace_rows) + 4.3)
    ax.axis("off")

    ax.text(0.8, len(trace_rows) + 3.75, f"Translation Conveyor Belt — Live Engine Trace ({model_name}  ->  {target_lang.title()})", fontsize=14, fontweight="bold", color="#1a202c")
    ax.text(0.8, len(trace_rows) + 3.25, f"Input:  \"{sentence}\"", fontsize=10.5, color="#4a5568")

    station_specs = [
        (33.0, "S0: Guard", "Fix invariant verbs"),
        (42.5, "S1: Idiom", "2-4 word phrases"),
        (52.0, "S2: Name", "Lore / OOV PROPN"),
        (61.5, "S3: Lemma", "UD table / suffix"),
        (71.0, "S4: Lexicon", "Lookup root word"),
        (80.5, "S5: Affix", "Attach ending"),
    ]
    for sx, stitle, ssub in station_specs:
        ax.add_patch(patches.FancyBboxPatch((sx, len(trace_rows) + 2.05), 8.6, 0.85, boxstyle="round,pad=0.04", facecolor="#2d3748", edgecolor="none"))
        ax.text(sx + 4.3, len(trace_rows) + 2.58, stitle, fontsize=8.5, fontweight="bold", color="white", ha="center")
        ax.text(sx + 4.3, len(trace_rows) + 2.20, ssub, fontsize=6.8, color="#cbd5e0", ha="center")

    ax.add_patch(patches.FancyBboxPatch((0.5, len(trace_rows) + 1.25), 99, 0.58, boxstyle="round,pad=0.04", facecolor="#edf2f7", edgecolor="none"))
    ax.text(1.0, len(trace_rows) + 1.46, "Token", fontsize=9.5, fontweight="bold", color="#2d3748")
    ax.text(13.5, len(trace_rows) + 1.46, "ML Tag", fontsize=9.5, fontweight="bold", color="#2d3748")
    ax.text(90.5, len(trace_rows) + 1.46, target_lang.title(), fontsize=9.5, fontweight="bold", color="#2d3748")

    for idx, row in enumerate(trace_rows):
        y = len(trace_rows) - idx + 0.35
        bg = "#ffffff" if idx % 2 == 0 else "#f7fafc"
        ax.add_patch(patches.Rectangle((0.5, y - 0.18), 99, 0.80, facecolor=bg, edgecolor="#e2e8f0", linewidth=0.6))
        ax.text(1.0, y + 0.12, row["token"], fontsize=9.8, fontweight="bold", color="#1a202c")
        ax.text(13.5, y + 0.12, row["tag"], fontsize=8.3, family="monospace", color="#4a5568")

        for s_idx, skey in enumerate(["s0", "s1", "s2", "s3", "s4", "s5"]):
            b_txt, b_bg, b_fg = _style(row[skey])
            bx = 33.0 + s_idx * 9.5
            ax.add_patch(patches.FancyBboxPatch((bx, y - 0.05), 8.6, 0.50, boxstyle="round,pad=0.03", facecolor=b_bg, edgecolor="none"))
            ax.text(bx + 4.3, y + 0.12, b_txt, fontsize=7.6, fontweight="bold", color=b_fg, ha="center")

        ax.text(90.5, y + 0.12, row["output"], fontsize=10, fontweight="bold", color="#2b6cb0")

    ax.add_patch(patches.FancyBboxPatch((0.5, 0.1), 99, 0.75, boxstyle="round,pad=0.05", facecolor="#ebf8ff", edgecolor="#90cdf4", linewidth=1))
    ax.text(1.5, 0.38, f"Final Translated Sentence:   {final_out}", fontsize=11, fontweight="bold", color="#2b6cb0")

    out_path = resolve_path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()


def export_crf_transitions_visual(
    prod_dir: str = "artifacts/production",
    out_file: str = "figures/crf_transitions_25x25.png",
):
    pt_path = next(resolve_path(prod_dir).glob("*bilstm*.pt"))
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)

    idx2tag = ckpt["idx2tag"]
    sorted_ids = sorted(int(k) for k in idx2tag.keys())
    labels = [idx2tag[i] if i in idx2tag else idx2tag[str(i)] for i in sorted_ids]
    trans = ckpt["state_dict"]["crf.transitions"].numpy()[np.ix_(sorted_ids, sorted_ids)]

    vlim = round(float(np.max(np.abs(trans))), 1)
    fig, ax = plt.subplots(figsize=(13.5, 11.5), dpi=300)
    im = ax.imshow(trans, cmap="RdBu", vmin=-vlim, vmax=vlim)

    ax.grid(False)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="#e2e8f0", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xlabel("Next Predicted Tag (To)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Current Predicted Tag (From)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title("BiLSTM-CRF — Learned 25 × 25 Grammar Transition Matrix (Blue = Favored, Red = Penalized)", fontsize=13, fontweight="bold", pad=14)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Learned Transition Weight", fontsize=10)

    out_path = resolve_path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()


def export_rf_tree_visual(
    prod_dir: str = "artifacts/production",
    out_file: str = "figures/rf_tree_sample.png",
    max_depth: int = 2,
):
    rf_path = next(
        p for p in resolve_path(prod_dir).iterdir()
        if ("random" in p.name.lower() or "rf" in p.name.lower()) and p.suffix in {".joblib", ".pkl"}
    )
    rf_bundle = joblib.load(rf_path)

    candidates = list(rf_bundle.values()) if isinstance(rf_bundle, dict) else [rf_bundle]
    all_objs = list()
    for item in candidates:
        all_objs.append(item)
        if hasattr(item, "named_steps"):
            all_objs.extend(item.named_steps.values())
        elif isinstance(item, (list, tuple)):
            all_objs.extend(item)

    vec = next(o for o in all_objs if isinstance(o, DictVectorizer))
    clf = next(o for o in all_objs if isinstance(o, RandomForestClassifier))
    le = next((o for o in all_objs if isinstance(o, LabelEncoder)), None)

    feature_names = list(vec.get_feature_names_out())
    if le is not None:
        class_names = [str(c) for c in le.classes_]
    elif isinstance(rf_bundle, dict) and "idx2tag" in rf_bundle:
        idx2tag = rf_bundle["idx2tag"]
        class_names = [str(idx2tag[int(c)]) for c in clf.classes_]
    else:
        class_names = [str(c) for c in clf.classes_]

    single_tree = next(iter(clf.estimators_))

    fig, ax = plt.subplots(figsize=(15, 7), dpi=300)
    plot_tree(
        single_tree,
        max_depth=max_depth,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10,
        impurity=False,
        proportion=True,
        ax=ax,
    )
    for txt in ax.texts:
        lines = [l for l in txt.get_text().splitlines() if not l.startswith(("value =", "[0.", "0."))]
        txt.set_text("\n".join(lines))

    ax.set_title(
        f"Random Forest — Partial Decision Tree (Tree #1 of {len(clf.estimators_)}, Top {max_depth} Levels)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    out_path = resolve_path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
