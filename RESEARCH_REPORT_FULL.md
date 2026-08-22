# Mwakatobe Research Report — Full

Date: 2026-08-12

This file collects a complete research-ready report draft for the Mwakatobe federated dataset and experiments. It includes dataset description, anonymisation/code notes, required figures/tables (with placeholders), technical validation, preliminary results, and reproducible commands. Replace placeholders with final numbers and figures as needed.

---

**Contents**

- Summary
- Figures (list + captions)
- Pipeline (concise steps)
- Dataset description & data dictionary
- Anonymisation & code (exact routines and reversibility)
- Federated training & system architecture (clients, server, Flower, privacy mechanisms)
- Tables (Dataset variables, Hospital stats, Missingness, Diagnoses, Outcomes, Ethics mapping)
- Technical validation (sanity checks, anonymisation checks)
- Preliminary results (per-hospital confusion matrices, experiment summary, FL convergence notes, basic fairness metrics)
- Reproducible run & code availability
- Appendix: code excerpts and commands

---

## Summary

This project uses a federated learning (FL) setup (Flower) to train a shared hospital model while keeping patient-level data local. Data flows: `data/raw/` → `data/processed/` → `data/cleaned/` → model training (local clients). Privacy protections: local hashing of identifiers, client-side differential privacy (DP) noise addition to parameter updates, and additive masking (SMPC-style) before upload. The server runs a FedAvg-style aggregator and only receives masked/noisy model updates.

Key deliverables for reviewers:
- Federated dataset description and per-hospital breakdown
- Full anonymisation code and description of reversibility / secret management
- Figures: network diagram, pipeline, dataset composition, variable distributions, FL workflow
- Tables: data dictionary, hospital counts, missingness, diagnoses, outcome classes, ethics mapping
- Technical validation and a simple baseline model demonstrating usability

---

## Figures (to produce and include)

- Figure 1 — Hospital network: nodes = hospitals, edges = FL communication to central server (annotate sample counts per hospital).
- Figure 2 — Data preprocessing pipeline (graphic): raw EMR → cleaning → harmonisation → encoding → anonymisation → QC → final CSV.
- Figure 3 — Dataset composition: bar chart of samples per hospital and per-outcome class.
- Figure 4 — Variable distribution: histograms / violin plots for Age and vitals (Pulse, Resp, Temp, Sys, Dia); categorical frequency tables for Sponsor/Procedures/Medications.
- Figure 5 — FL workflow: client architecture (local preprocessing, local training loop, DP/SMPC steps), server (FedAvg aggregator), communication rounds timeline.

Pipeline figure (compact step list):

- Step 1 — Cleaning: `client/data_utils.py` column detection, normalization, deduplication.
- Step 2 — Harmonisation: `client/clean_app.py` (drop identity columns, unify outcomes, create `Age_bin`).
- Step 3 — Encoding: global/cached OneHot encoders (`GLOBAL_ENCODERS`), numeric scaling.
- Step 4 — Anonymisation: `safe_hash` applied to `Id`, `Name`, `District`, `Ward`, `Diagnoses` (SHA-256 + salt).
- Step 5 — Quality control: missing-value imputation (medians/defaults), deduplication, and exported `data/cleaned/*.csv`.

---

## Dataset description & data dictionary (Table 1)

Table 1 — Dataset variables (present columns in `data/cleaned/*.csv`):

- `Age` — numeric (years)
- `Age_bin` — categorical (0-19, 20-25, 26-30, 31-35, 36-40, 41-50, 51+ , Unknown)
- `Sponsor` — categorical (GOVERNMENT / CASH / PRIVATE / Unknown)
- `Region` — categorical (geographical region; hashed in some flows)
- `Pulse`, `Resp`, `Temp`, `Sys`, `Dia` — numeric vitals (imputed by median or default)
- `Procedures`, `Medications` — clinical text/generalized fields (often 'Recorded' / 'Not recorded')
- `Outcome` — label (mapped to integers; see Table 5)

Add any derived or encoded columns (OneHot encoded columns) after preprocessing — report the final column count and names in supplementary materials. Reviewers will check number of features post-encoding; run a quick script to list columns and counts and include the number here.

Table 2 — Hospital statistics (per-hospital counts)

| Hospital | Cleaned records |
|---|---:|
| HospitalA | 10569 |
| HospitalA_mapped | 10569 |
| HospitalB | 990 |
| HospitalB_mapped | 990 |
| HospitalC | 1770 |
| HospitalC_mapped | 1770 |
| **TOTAL** | **26658** |

Note: the table shows counts of cleaned CSVs present under `data/cleaned/`.

Table 3 — Missingness summary (per hospital)

| Hospital | % missing Pulse | % missing Resp | % missing Temp | % missing Sys | % missing Dia |
|---|---:|---:|---:|---:|---:|
| HospitalA | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| HospitalA_mapped | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| HospitalB | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| HospitalB_mapped | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| HospitalC | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| HospitalC_mapped | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

Guidance: these percentages were computed with pandas `isna().mean()` on the vitals columns and reflect the cleaned CSVs in `data/cleaned/`.

Table 4 — Diagnoses (top-k frequency) — produce per-hospital frequency tables of `Diagnoses` (after anonymisation/hashing, provide grouped categories or `Recorded` counts).

Table 5 — Outcome classes mapping (global)

Provide mapping used by `get_global_outcome_classes` and `map_outcomes_to_global`. Example in this repo: `Home`=0, `Admitted`=1, `Referred`=2, Unknown=3. Confirm mapping used in your run and include counts per class.

Table 6 — Ethical compliance mapping

| Item | Status | Note |
|---|---|---|
| Institutional approval (IRB) | (yes/no) | add reference / date / protocol # |
| Data sharing agreements | (yes/no) | describe access restrictions |
| Anonymisation method documented | yes | SHA-256 hashing of identifiers + DP/SMPC for model updates |
| Re-identification risk assessment | (done/not done) | add results / k-anonymity check if available |

---

## Anonymisation & Code for anonymisation (required to share)

Include the exact code, with version and commit hash. Below is the anonymisation routine used in the repo (copy of the implementation):

```python
def safe_hash(value, salt_key='id'):
    if pd.isna(value) or str(value).strip() == '':
        return np.nan
    clean = str(value).strip()
    salt = SALTS.get(salt_key, "")
    return hashlib.sha256((salt + clean).encode('utf-8')).hexdigest()
```

Salts are defined in `client/data_utils.py` as the `SALTS` dict. For publication/shareable code:

- DO NOT publish the plaintext salts. Replace salt values with environment variables in production and in any public code sample.
- State clearly whether salts are centrally held (e.g., by the coordinating centre) or held privately per hospital. If salts are centrally held, anonymisation is reversible by the salt-holder in principle — document this.

Reversibility and risk statement:
- Hashing with salt is not reversible without the salt; however if salts are stored centrally and can be accessed, re-identification is possible. In your report state whether salts are held centrally and what access controls exist. For stronger non-reversible anonymisation consider irreversible tokenization with per-hospital secrets and/or irreversible redaction of free-text fields.

Provide the anonymisation code as part of the software availability statement (see final section).

---

## Federated training and system architecture

System overview:

- Server: Flower server (`server/server_app.py`) running a `SecureFedAvg` (FedAvg wrapper). Server config: address `127.0.0.1:9090`, `num_rounds` default 10 in example.
- Clients: `client/client_app.py` implements `FlowerHospitalClient` (NumPyClient). Each client loads local `data/cleaned/HospitalX.csv`, trains a local model for multiple local epochs, and returns NumPy model parameter arrays.
- Model: small feed-forward network (`HospitalModel`) in `client/client_app.py` / `client/model_utils.py`.

Privacy mechanisms (high-level):

- Local identifier hashing (see `safe_hash`).
- Differential privacy (DP): optional client-side clipping and Gaussian noise applied to model parameter updates (`clip_and_add_noise(params, C=5.0, sigma=DP_SIGMA)`), controlled by `USE_DP` and `DP_SIGMA` in `client/client_app.py`.
- SMPC-style masking (secure aggregation): clients generate random additive masks (seeded RNG) and apply them to updates; masks cancel on the server at aggregation time. See `generate_mask` and `apply_mask`.

Recommendation for the methods section (short):

- Describe client architecture (local preprocess -> encode -> local trainer -> DP/SMPC -> send updates) and server (FedAvg aggregation). Explain that Flower coordinates rounds and distributes global parameters.

---

## Technical validation (required)

Include the following checks and their results in the final manuscript:

1. Anonymisation evaluation
   - k-anonymity check for quasi-identifiers (age bins, region, sponsor) — compute k for each equivalence class.
   - Statement on collision risk: SHA-256 collision probability negligible; salted hash prevents simple dictionary attacks but salt secrecy matters.

2. Sanity checks
   - Physiological ranges: filter and count any values outside plausible ranges (e.g., Temp < 30 or > 43, Sys < 50 or > 250, Resp > 60).
   - Pairwise checks: e.g., Age vs. Age_bin consistency, Outcome not missing post-mapping.

3. Provenance & reproducibility
   - Include exact git commit hash and `requirements.txt` (list packages: flwr, torch, opacus, sklearn, pandas, imbalanced-learn, matplotlib, seaborn, etc.).

4. Baseline model for dataset usability
   - Run a simple baseline on a combined cleaned dataset (centralized logistic regression or small NN) and report accuracy/F1. This demonstrates the dataset supports modeling and should be included as a short table or figure.

---

## Preliminary results (populate and refine)

Aggregate experiment metrics (from `results/experiment_summary.csv`):

- Accuracy (mean): 51.386% (Std 0.371)
- Precision (mean): 50.731% (Std 0.171)
- Recall (mean): 95.647% (Std 4.240)
- Specificity (mean): 7.126% (Std 3.500)
- F1 (mean): 66.280% (Std 1.178)

Per-run results available in `results/experiment_runs.csv` include three runs (sigma=0.002 used). Attach these CSVs as supplementary data.

Per-hospital confusion matrix-derived metrics (Round 0 CSVs found in `results/confusion_matrices/`):

Table: Per-hospital performance (Round 0)

| Hospital | Accuracy (%) | Precision (%) | Recall/Sensitivity (%) | Specificity (%) | F1 (%) | FPR (%) | FNR (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| HospitalA | 99.19 | 98.69 | 99.72 | 98.67 | 99.20 | 1.33 | 0.28 |
| HospitalB | 55.87 | 92.59 | 12.76 | 98.98 | 22.45 | 1.02 | 87.24 |
| HospitalC | 40.86 | 37.79 | 28.29 | 53.43 | 32.21 | 46.57 | 71.71 |

Notes:
- HospitalA shows near-perfect per-round performance (possible label leakage or very different class balance). HospitalB and HospitalC show much lower recall for the positive class — document class imbalance and per-hospital sample size.
- Provide a small subsection interpreting these numbers and whether they arose from label imbalance, small test sets, or preprocessing issues.

Fairness metrics (suggested):

- Per-hospital disparity in Recall and FPR (report ratio of highest/lowest recall, and difference in FPR across hospitals).
- If demographic attributes exist, compute parity metrics (e.g., demographic parity difference, equal opportunity difference) and include in supplementary material.

FL convergence sample (visual):
- Use `results/epoch/epoch_accuracy/*` and `results/epoch/epoch_loss/*` PNGs — include a figure showing epoch-level accuracy per hospital across rounds.

---

## Dataset Usage Example / Minimal Notebook

Provide a small example in the repo that reads one cleaned CSV, trains a small centralized logistic regression, and prints classification metrics. Add as `examples/quick_baseline.py` and include output CSVs. This strengthens the data descriptor and demonstrates usability.

Command snippet (example):

```bash
python examples/quick_baseline.py data/cleaned/HospitalA.csv
```

Output: a small CSV with accuracy/F1 and confusion matrix saved to `results/experiments/`.

---

## Reproducible run & code availability

Commands to reproduce experiments (example):

```bash
# 1) Preprocess and clean
python client/data_utils.py
python client/clean_app.py

# 2) Start server
python server/server_app.py

# 3) Start clients (one process per hospital)
python client/client_app.py data/cleaned/HospitalA.csv
python client/client_app.py data/cleaned/HospitalB.csv
python client/client_app.py data/cleaned/HospitalC.csv
```

Software availability statement (example text to adapt):

"Preprocessing and model code are available at the project repository [link]. Raw patient-level data are not shared due to patient privacy constraints; cleaned and anonymised CSVs used in the experiments are archived at [location] under controlled access. The anonymisation code and preprocessing scripts are included in this repository. Contact the corresponding author for data access requests and IRB documentation."

---

## Appendix: quick references (code excerpts)

- `client/data_utils.py`: `safe_hash`, `parse_age_to_years`, `preprocess_single_file` — anonymisation and column harmonisation.
- `client/clean_app.py`: `clean_all_data` — imputation defaults, `Age_bin` creation, mapping outcomes and saving cleaned files.
- `client/client_app.py`: FL client (`FlowerHospitalClient`), DP noise `clip_and_add_noise`, mask generation `generate_mask`, `train_local_model`, and result plotting/summaries.

### Recommended actions before submission

1. Replace salt values in `client/data_utils.py` with environment variable reads and rotate secrets. Add a short paragraph describing key management and who holds salts.
2. Run the k-anonymity / re-identification risk checks and add numeric results.
3. Populate Table 2/3 values by running quick pandas counts over `data/cleaned/*.csv`.
4. Add baseline centralized model run and include results in Preliminary Results.
5. Create the five figures listed above (figures already partially available under `results/` — collect and annotate them for the paper).

---

If you want, I can now:

- (A) Fill Tables 2 and 3 automatically by scanning `data/cleaned/*.csv` and computing counts and missingness.
- (B) Add the small centralized baseline script `examples/quick_baseline.py` and run it to produce example metrics.
- (C) Produce the five figures as publication-ready PNGs (use existing `results/` plots and generate missing ones).

Tell me which of (A)/(B)/(C) to do next and I'll proceed.
