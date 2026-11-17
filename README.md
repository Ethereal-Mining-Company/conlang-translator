# Codex Arcanum
*A Multilingual Morphological POS Tagger & Conlang Translation Engine*

![Arcane Translator Cover Image](images/arcane_translator.png)

>[!CAUTION]
>This is a fictional project as part of a broader D&D universe campaign for entertainment use only. It may be integrated into educational projects as flavor text to help simulate working in real-world environments.

>[!NOTE]
>This `README.md` contains flavor text for entertainment use only to be used with an educational project.


## Preface
> *"No living voice has ever spoken these words aloud, yet this silent tongue binds the stars and answers every spell cast in this world. It is simply the true language of magic, woven into the fabric of creation itself. The pages that follow are our humble attempt to capture how these words might look and sound to mortal eyes and ears. May this work serve you well as you study the mysteries within."*

**Codex Arcanum** currently supports **English**, **Aelorian** (SVO), **Vaelorian** (SVO), and **Low Nerakan** (SVO) as constructed campaign languages, with dynamic support for both **English ↔ Conlang** and **direct Conlang ↔ Conlang** pivot translation. Because these languages encode pluralization, tense, comparative degree, and possession through inflectional suffixes (`-i`/`-as`/`-ot` for plural, `-an`/`-en`/`-uk` for past tense, `-el`/`-iel`/`-ek` for comparative) rather than English helper words, the translator relies on accurate Part-of-Speech and morphological tags to preserve the original meaning of spellbooks and old records.


## Project Overview
An end-to-end Natural Language Processing (NLP) and MLOps pipeline that trains, tunes, and compares **Traditional Machine Learning** and **Deep Learning** sequence-labeling models on a **25-class morphological Part-of-Speech (POS) tagging task** using the Universal Dependencies English-EWT corpus, paired with a configuration-driven **multilingual bidirectional translation engine** supporting **English**, **Aelorian**, **Vaelorian**, and **Low Nerakan**.

Standard POS taggers predict coarse categories (`NOUN`, `VERB`, `ADJ`), which is insufficient for translating between languages with different word orders and inflectional morphology. This project formulates a **25-class joint POS + morphological tagging task** (`NOUN|Number=Plur`, `VERB|Tense=Past`, `VERB|VerbForm=Part`, `VERB|Person=3|Number=Sing`, `ADJ|Degree=Cmp`, `PRON|Poss=Yes`, etc.) and evaluates how distinct machine learning paradigms handle subword morphology, out-of-vocabulary (OOV) proper nouns, and sentence-level syntactic transitions.


### Core Models Evaluated
1. **Multi-Scale Char-BiLSTM-CRF (Deep Learning Primary):** Combines word embeddings with case-sensitive multi-scale 1D Character Convolutions (`kernel_size=2, 3, 4`), a 2-layer Bidirectional LSTM, and a Conditional Random Field (CRF) layer for joint whole-sentence Viterbi sequence decoding.
2. **Random Forest (Traditional ML Primary):** A sparse `DictVectorizer` + `RandomForestClassifier` ensemble trained on 16 engineered linguistic features (1–3 char prefixes, 1–4 char suffixes, capitalization/orthography flags, and a 2-word context window).
3. **MLP Neural Network (Benchmark):** A sparse feed-forward `MLPClassifier` trained directly on ~80,000 one-hot linguistic features with early stopping.
4. **HistGradientBoosting (Benchmark):** A memory-guarded histogram gradient booster (`float32` `FeatureHasher` + `HistGradientBoostingClassifier`) demonstrating sequential boosting trade-offs on high-dimensional text features.


## Supported Languages & Grammar Profiles
| Language | Code | Word Order | Plural Suffix | Past Suffix | Future Suffix | Comp / Super Suffix |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Aelorian** | `aelorian` | SVO (Adj before) | `-i` | `-an` | `-or` | `-el` / `-is` |
| **Vaelorian** | `vaelorian` | SVO (Adj before) | `-as` | `-en` | `-ir` | `-iel` / `-ish` |
| **Low Nerakan** | `low_nerakan` | SVO (Adj before) | `-ot` | `-uk` | `-ag` | `-ek` / `-ug` |


## Model Benchmark Leaderboard (UD English-EWT Test Split)
All models were trained, tuned, and evaluated on the identical **25-class morphological POS tagging task** (`12,543` train / `2,002` dev / `2,077` held-out test sentences).

| Rank | Model Family | Architecture Type | Winning Run | Val Accuracy | Test Accuracy | Macro F1 | Cold Load (ms) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **Multi-Scale Char-BiLSTM-CRF** | Deep Sequence Model (Primary DL) | `BiLSTM_MultiCharCNN` | **94.21%** | *~94.2%* | *See metrics* | ~93 ms |
| **2** | **MLP Neural Network** | Sparse Feed-Forward NN (Benchmark) | `MLP_Tuned` | **93.80%** | *~93.8%* | *See metrics* | ~147 ms |
| **3** | **HistGradientBoosting** | Sequential Tree Booster (Benchmark) | `HGB_Hashed2048` | **93.20%+** | *~93.2%* | *See metrics* | ~391 ms |
| **4** | **Random Forest** | Bagging Tree Ensemble (Primary ML) | `RF_Tuned` | **92.63%** | *~92.6%* | *See metrics* | ~2605 ms |

*(Note: Exact held-out test metrics, Macro F1 scores, and `run_uuid` hashes are logged automatically in `artifacts/production/model_metrics.json` and `logs/experiment_runs.csv`.)*


## Repository Structure
```text
conlang-translator/
├── config.yaml                  # Central configuration (paths, splits, seeds, features, model sweeps)
├── main.ipynb                   # End-to-end EDA, training sweep, log analysis, Plotly/Seaborn visuals & demo
├── translate.py                 # Interactive CLI translator with live model switching & /diag mode
├── requirements.txt             # Pinned Python dependencies
├── README.md                    # Project documentation and experimental summary
├── data/
│   ├── english_rules.yaml       # Lemmatization & morphological synthesis rules
│   ├── lemmas.yaml              # Auto-exported English lemma lookup table from UD corpus
│   ├── languages/
│   │   ├── aelorian/            # Aelorian (SOV) YAML config, CSV lexicon, & Vulgarlang exports
│   │   ├── vaelorian/           # Vaelorian (SVO) YAML config, CSV lexicon, & Vulgarlang exports
│   │   └── low_nerakan/         # Low Nerakan (SVO) YAML config, CSV lexicon, & Vulgarlang exports
│   └── raw/                     # Cached Universal Dependencies English-EWT (.conllu) splits
├── src/
│   ├── __init__.py
│   ├── features.py              # CoNLL-U parser, 25-class target builder, & dynamic feature extractor
│   ├── engine.py                # Bidirectional & multi-conlang pivot translation engine
│   ├── utils.py                 # Config loader, deterministic seeding, UUID CSV loggers, & plot exporters
│   ├── visualizations.py        # Interactive Plotly & static figure generation helpers
│   └── models/
│       ├── __init__.py          # Dynamic plugin registry, grid search expander, & champion promotion
│       ├── base.py              # Shared tagger protocol / base interface
│       ├── bilstm_crf.py        # Multi-Scale Char-BiLSTM-CRF (PyTorch + torchcrf)
│       ├── random_forest.py     # Sparse DictVectorizer + RandomForestClassifier
│       ├── mlp_tagger.py        # Sparse Multi-Layer Perceptron benchmark
│       └── hist_gbdt.py         # Memory-guarded FeatureHasher + HistGradientBoosting benchmark
├── artifacts/
│   └── production/              # Promoted champion artifacts & model_metrics.json
├── logs/
│   ├── experiment_runs.csv      # UUID-tracked hyperparameter sweep leaderboard
│   ├── epoch_history.csv        # Per-epoch and per-step validation curves
│   └── *.log                    # Per-model training execution logs
├── figures/                     # High-resolution PNG charts, confusion matrices, & interactive HTML plots
├── images/                      # Saved output screenshots for presentation & documentation
├── presentations/               # Narrated slide deck deliverables
└── references/                  # Linguistic and dataset reference materials
```


## Quickstart & Installation
### 1. Environment Setup
```bash
git clone https://github.com/Ethereal-Mining-Company/conlang-translator.git
cd conlang-translator
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train & Evaluate Models
Run the full experiment sweep from the command line or execute **`main.ipynb`**:
```bash
python -m src.models
```
* **Automatic Dataset Ingestion:** Downloads and caches the official Universal Dependencies `en_ewt-ud-{train,dev,test}.conllu` splits automatically on first run.
* **Champion-vs.-Challenger Promotion:** Every candidate run is logged with a unique `run_uuid` in `logs/experiment_runs.csv`. Production artifacts in `artifacts/production/` are only overwritten when a challenger beats the current champion's validation score (unless `force_overwrite=True` is passed).


### 3. Launch the Interactive CLI Translator
```bash
python translate.py
```

#### Interactive CLI Commands
| Command | Description |
| :--- | :--- |
| `/to <lang>` | Switch target language (`english`, `aelorian`, `vaelorian`, `low_nerakan`) |
| `/from <lang>` | Switch source language (supports direct conlang-to-conlang pivot translation) |
| `/swap` | Swap source and target languages |
| `/model <name>` | Hot-swap active production model (`bilstm_crf`, `random_forest`, `mlp_tagger`, `hist_gbdt`) |
| `/diag` | Toggle **Multi-Model Diagnostics Mode** (compares every enabled model token-by-token with latency and side-by-side translations) |
| `/reload` | Hot-reload language YAML lexicons and grammar rules from disk without restarting |
| `/info` | Display active model UUID, hyperparameters, and validation/test metrics |
| `/menu`, `/help` | Reprint command menu and active session settings |
| `/quit` | Exit the CLI |


## Key Experimental Findings
* **Subword & Orthography Visibility Drives OOV Accuracy:** Adding case-sensitive multi-scale character convolutions (`kernel_size=2, 3, 4`) to BiLSTM-CRF boosted validation accuracy from **90.33%** (word-only) to **94.21%**, resolving unseen proper nouns (e.g., `"Rue"` → `PROPN`) that word-level embeddings previously mapped to `<UNK>`.
* **Sequence Decoding Resolves Syntactic Ambiguity:** In multi-model diagnostics (`"The wizards of Waterdeep have fled to higher ground."`), BiLSTM-CRF and MLP accurately distinguished prepositional `"to"` (`ADP`) and comparative `"higher"` (`ADJ|Degree=Cmp` → `solisel`), whereas isolated tree models struggled with context-dependent word roles.
* **Production Latency Trade-Off:** Tree ensembles (`Random Forest`) exhibit higher cold-start disk load times (~1.7–2.6s) due to serialized forest size, whereas `BiLSTM-CRF` loads in ~90ms while maintaining sub-millisecond per-sentence inference.


## Reproducibility & Experiment Tracking
* **Deterministic Seeding:** Controlled via `runtime.random_seed: 42` in `config.yaml`.
* **Full Telemetry:** Every training run logs `run_uuid`, `promoted` status, `tuned_params`, `mean_cv_accuracy`, `final_test_score`, `macro_f1`, and `latency_ms` to `logs/experiment_runs.csv`, alongside epoch/iteration learning curves in `logs/epoch_history.csv`.