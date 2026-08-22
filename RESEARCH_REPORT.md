# Research Report — Data, Privacy, Model Sharing, and Results

## Dataset Overview

- Data locations: `data/raw/`, `data/processed/`, `data/cleaned/`.
- Per-hospital cleaned CSVs are under `data/cleaned/` (e.g. HospitalA.csv).
- Processed intermediate files are in `data/processed/` and final metrics/plots are written to `results/`.

## How the program handles privacy

- Identifiers: `Id` and `Name` are hashed using SHA-256 with a per-field salt. See `client/data_utils.py` (`safe_hash`) — salts are defined in the `SALTS` dict and must be replaced with environment variables in production.
- Partial generalization: some clinical fields are replaced with coarse values (e.g. `Recorded` vs `Not recorded`).
- Differential privacy (DP): client-side clipping and Gaussian noise are applied to model parameter updates when `USE_DP` is enabled (noise scale `DP_SIGMA`, implementation in `client/client_app.py` `clip_and_add_noise`). Report the `sigma` used.
- Secure aggregation / SMPC: clients generate additive masks and apply them to their updates so the server aggregates masked updates without inspecting individual updates. See `generate_mask` / `apply_mask` in `client/client_app.py` and `SecureFedAvg` in `server/server_app.py`.
- Opacus mention: the repo imports `opacus` (used for DP in PyTorch) — verify any per-example privacy accounting if enabled in experiments.
- Recommendation: rotate salts, store them as secrets, and document the threat model (honest-but-curious server vs. malicious client), plus any PSG or IRB approvals.

## How data and model updates are shared

- Federated learning with Flower (`flwr`): the server runs a FedAvg-style strategy (`server/strategy.py`) and clients use `fl.client.NumPyClient` (`client/client_app.py`).
- Clients send only model parameter updates (NumPy arrays). With DP and SMPC enabled, those updates are noisy and masked on the client side before being uploaded.
- The server uses `SecureFedAvg` and never inspects raw client updates beyond aggregation (per comments in `server/server_app.py`).
- Global encoders / outcome mapping: encoders and a global outcome-to-index map are coordinated so clients map local labels to a shared label space (`get_global_outcome_classes` and `map_outcomes_to_global` in `client/client_app.py`).

## How another hospital can train with their data

1. Prepare raw CSV following local conventions then run preprocessing:
   - Run `client/preprocess` flow: use `python client/data_utils.py` or call `preprocess_single_file` to create processed CSVs in `data/processed/`.
   - Run `python client/clean_app.py` (or import `clean_all_data`) to produce `data/cleaned/*.csv`.
2. Ensure outcomes and classes are consistent: the code maps outcomes to global indices using `get_global_outcome_classes`.
3. Start a Flower server (central coordinator):

```
python server/server_app.py
```

4. Start a hospital (client) using their cleaned CSV path:

```
python client/client_app.py data/cleaned/HospitalX.csv
```

5. Configuration knobs to report/adjust:
- `USE_DP` and `DP_SIGMA`: whether DP is on and the noise scale.
- `USE_SMPC`: whether secure aggregation (masking) is used.
- `USE_SMOTE`: whether local SMOTE oversampling is applied for class imbalance.
- Number of federated rounds (server config) and local epochs per client (client training loop).

## What to include in the research report (based on `results/`)

- Experiment summary files:
  - `results/experiment_runs.csv` — per-run metrics saved by `save_experiment_summary` / `save_experiment_results`.
  - `results/experiment_summary.csv` — aggregated mean/std across runs.
  - `results/experiments/experiment_run_<N>_sigma_<SIGMA>.csv` — per-run metrics with DP sigma noted.

- Confusion matrices:
  - `results/confusion_matrices/*.csv` — saved numeric confusion matrices per hospital and round.
  - `results/confusion_matrices/*_confusion_matrix.png` — generated png images (one per hospital) at program exit.

- Epoch / round plots:
  - `results/epoch/epoch_accuracy/` and `results/epoch/epoch_loss/` — per-hospital epoch accuracy and loss plots.
  - `results/global/` — aggregate plots: dataset sizes, federated weights, sample distribution, contribution matrix, and experiment comparison visuals.

- Recommended reported items to include in manuscript or internal report:
  - Dataset description: per-hospital sample counts (train/test), age distributions, and class balance (before/after SMOTE).
  - Privacy settings: whether DP and SMPC were active, the `DP_SIGMA` value, clipping norm `C`, and whether salts were rotated.
  - Federated configuration: number of clients, number of federated rounds, local epochs, aggregation strategy (`FedAvg`), and fraction_fit/eval settings.
  - Performance: per-hospital confusion matrices and derived metrics (Accuracy, Precision, Recall/Sensitivity, Specificity, F1, FPR, FNR). Use `results/experiments` CSVs for per-run numbers and `results/experiment_summary.csv` for aggregate statistics.
  - Training dynamics: epoch-wise accuracy and loss plots per hospital (from `results/epoch/`) and any notable divergence across sites.
  - Contribution weights: pie chart and contribution matrix showing relative dataset contributions to FedAvg (from `results/global/`).
  - Failure modes: report any hospitals with extremely small classes, skipped SMOTE, or samples mapped to `Unknown` outcomes.

## Practical checklist for submission

- Include the exact code revision (commit hash) and `requirements.txt` used for experiments.
- Attach or reference:
  - `results/experiment_summary.csv` and `results/experiments/*.csv` (raw CSVs)
  - `results/confusion_matrices/*.csv` and PNGs
  - `results/global/*.png` and `results/epoch/*/*.png` for visualizations
- Report privacy parameters and threat model explicitly:
  - DP: `sigma`, clipping norm `C`, per-client accounting (if using Opacus).
  - SMPC: mask generation method and seed strategy.
  - Hashing salts and where they are stored (do not include secrets in the paper; describe management policy).

## Reproducible run (minimal commands)

```
python client/data_utils.py
python client/clean_app.py

python server/server_app.py

python client/client_app.py data/cleaned/HospitalA.csv
python client/client_app.py data/cleaned/HospitalB.csv
python client/client_app.py data/cleaned/HospitalC.csv
```

## Notes & next steps

- Verify salts are moved to environment variables before any real-world deployment.
- If stronger privacy guarantees are required, add per-example DP accounting (Opacus) and formal secure aggregation implementation (e.g., cryptographic secure aggregation libraries) and record epsilon values.
- Consider extending the report with a short Methods subsection describing preprocessing steps (`client/data_utils.py`, `client/clean_app.py`) and model architecture (`client/model_utils.py` / `client/client_app.py`).

---

File generated from the repository codebase to help assemble your research report. Adjust wording and parameter values to match the experiments you actually ran.
