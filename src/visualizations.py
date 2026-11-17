import json
from pathlib import Path
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix
from src.utils import resolve_path


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
    fig = px.bar(pd.DataFrame(runs), x="Model", y="Validation Accuracy", color="Model", title=title, text_auto=".4f")
    fig.update_layout(showlegend=False, yaxis_range=[0, 1])
    return _save_fig(fig, out_dir, filename)


def plot_model_head_to_head(metrics_path: str | Path, out_dir: str | Path):
    data = json.loads(resolve_path(metrics_path).read_text(encoding="utf-8"))
    rows = [
        {"Model Family": k, "Test Accuracy": v["best_test_metrics"]["test_accuracy"], "Macro F1": v["best_test_metrics"]["macro_f1"]}
        for k, v in data.items()
    ]
    fig = px.bar(pd.DataFrame(rows), x="Model Family", y=["Test Accuracy", "Macro F1"], barmode="group", title="Model Test Performance Comparison", text_auto=".4f")
    fig.update_layout(yaxis_range=[0, 1])
    return _save_fig(fig, out_dir, "model_comparison")


def plot_top_confusion_matrix(y_true: list[str], y_pred: list[str], title: str, out_dir: str | Path, filename: str, top_k: int = 12):
    labels = pd.Series(y_true).value_counts().head(top_k).index.tolist()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues", title=title, labels={"x": "Predicted", "y": "True", "color": "Count"})
    return _save_fig(fig, out_dir, filename)