# ⚙️ SMART-PDM — AI Predictive Maintenance for Home Appliances

An interactive machine-learning dashboard that detects faults and anomalies in
**household appliances** from their electrical and vibration signals — built on the
EU **SMART-PDM** research dataset.

> **🔗 Live demo:** _add your Streamlit Cloud URL here after deploying_
> (free app — it may take ~30 s to wake on the first visit)

Built by an engineer with a **mechanical / industrial-engineering** background to show
predictive-maintenance ideas end-to-end: from raw machine signals to a deployable,
explainable decision tool.

---

## What it does

The app covers **two appliance types**, each with the modelling approach that fits its data:

### 🧺 Washing machines — supervised fault classification
- **108 wash cycles** across 4 brands, labelled **Working / Heating / Bearings / Motor**.
- Random Forest on **75 tsfresh features** (Active Power · Current · Vibration).
- A **windowed model** slices each cycle's high-frequency signal into short windows to
  expand the data to **4,000+ samples**, evaluated **leak-free with GroupKFold**.

### 🧊 Refrigerators — unsupervised anomaly detection
- **1,111 cycles**, but only **24 malfunctions (~2%)** — far too few for supervised
  learning, so the model learns "normal" and flags outliers (**IsolationForest**).
- Includes a distinctive **24-hour power trace** view showing compressor duty-cycling.

## Results (headline metrics)

| Task | Metric | Score |
|---|---|---|
| Washing-machine fault classification (cycle-level) | Macro-F1 (5-fold) | see app |
| Washing-machine **windowed** model | Macro-F1 (**GroupKFold**, leak-free) | **0.82** |
| Refrigerator anomaly detection | ROC-AUC | **0.88** |

> **Honesty first:** the app openly shows *why* the leak-free GroupKFold score (0.82) is
> the one to trust versus a naive shuffled split (0.97) — demonstrating how data leakage
> inflates results. Windowing multiplies *samples*, not independent fault *events*.

## Pages

- **Overview** — project framing, datasets, and metrics for both appliances.
- **Washing Machines** → Signal Explorer · Cycle Detection · Feature Analysis ·
  Fault Classifier · Windowed Model
- **Refrigerators** → Signal Explorer (with 24 h trace) · Feature Analysis · Anomaly Detection

## Tech stack

`Python` · `Streamlit` · `scikit-learn` · `tsfresh` · `Plotly` · `pandas` / `numpy`

---

## Run it locally

```bash
# from the repository root
pip install -r requirements.txt
streamlit run 1-code/app.py
```

The repo ships a **small demo subset** of raw cycles so the Signal Explorer pages work
out of the box; the analysis pages run on the bundled precomputed feature tables.

## Repository layout

```
.
├── requirements.txt              # slim runtime deps (for the deployed app)
├── 1-code/
│   ├── app.py                    # the Streamlit application
│   ├── extract_features.py       # tsfresh feature extraction (offline)
│   ├── window_features.py        # windowed feature extraction (offline)
│   ├── classifier.py, pca.py     # standalone analysis scripts
│   └── requirements-dev.txt      # full deps for the offline pipeline
├── 2-washing_machines/           # WM metadata + feature CSVs (+ demo cycles)
└── 3-fridges/                    # fridge metadata + feature CSVs (+ demo cycles)
```

## A note on the data

The full SMART-PDM raw dataset is **~26 GB** (multi-hour signals at up to ~2 kHz) and is
**not** included here. This repo tracks only:
- the small precomputed **feature / metadata CSVs** the models use, and
- a curated **~5 MB demo subset** of raw cycles for the Signal Explorer pages.

The offline scripts (`extract_features.py`, `window_features.py`) regenerate the feature
tables from the full dataset if you have it.

## Credits

Data: **SMART-PDM** EU research project — https://smart-pdm.eu/
Feature extraction adapted from the project's reference scripts (© 2022, GPL).
