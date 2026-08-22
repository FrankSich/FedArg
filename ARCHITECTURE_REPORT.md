# Mwakatobe Federated Learning and Privacy-Preserving Healthcare Architecture

## Executive Summary

Mwakatobe is a multi-institutional healthcare machine-learning prototype in which Hospital A, Hospital B, and Hospital C train a shared predictive model without placing their patient-level CSV records on a central server. The system uses Flower to coordinate federated learning, PyTorch for local neural-network training, local preprocessing and anonymisation, weighted Federated Averaging (FedAvg), and experimental client-side parameter perturbation and masking.

The architecture is best described as a **central-coordinator federated learning system with privacy-oriented controls**. Patient rows remain within each hospital's local data boundary during training. The coordinating server distributes the current global model and receives client model parameters, sample counts, and evaluation results. The current implementation does not yet provide a formal differential-privacy guarantee or cryptographic secure aggregation; those should be treated as planned hardening activities rather than completed security properties.

## 1. System Objectives and Boundaries

### Objectives

The proposed system aims to:

- Enable several hospitals to learn a common clinical outcome model.
- Avoid centralising raw patient records.
- Harmonise differently structured hospital datasets.
- Reduce exposure of direct identifiers and detailed clinical fields.
- Address local class imbalance.
- Measure performance separately for each hospital.
- Provide a foundation for fairness and privacy analysis.

### Trust boundaries

There are three important boundaries:

1. **Hospital boundary:** each institution owns and processes its local patient data.
2. **Communication boundary:** model parameters move between clients and the Flower server.
3. **Coordinator boundary:** the server coordinates rounds and aggregates updates but is not assumed to have access to raw patient rows.

A production deployment would also need authenticated, encrypted client-server communication, access control, audit logs, secret management, and an explicit threat model. The current local configuration uses `127.0.0.1:9090`, which is suitable for experimentation but not a distributed clinical deployment.

## 2. High-Level Architecture

```mermaid
flowchart TB
    R[data/raw/HospitalA.csv<br/>HospitalB.csv<br/>HospitalC.csv]
    P[Preprocessing and harmonisation<br/>column detection, cleaning, deduplication]
    C[Cleaned local datasets<br/>data/cleaned/]

    R --> P --> C

    C --> HA[Hospital A client]
    C --> HB[Hospital B client]
    C --> HC[Hospital C client]

    subgraph Local hospital processing
        HA --> HA1[Local encoding and scaling]
        HA1 --> HA2[Local train/test split and SMOTE]
        HA2 --> HA3[PyTorch local training]
        HA3 --> HA4[Privacy transformation]
        HB --> HB1[Local encoding and scaling]
        HB1 --> HB2[Local train/test split and SMOTE]
        HB2 --> HB3[PyTorch local training]
        HB3 --> HB4[Privacy transformation]
        HC --> HC1[Local encoding and scaling]
        HC1 --> HC2[Local train/test split and SMOTE]
        HC2 --> HC3[PyTorch local training]
        HC3 --> HC4[Privacy transformation]
    end

    HA4 --> S[Flower server<br/>FedAvg-style aggregation]
    HB4 --> S
    HC4 --> S
    S --> G[Updated global model parameters]
    G --> HA3
    G --> HB3
    G --> HC3

    HA3 --> E[Local evaluation and metrics]
    HB3 --> E
    HC3 --> E
    E --> O[results/<br/>CSV summaries, confusion matrices, plots]
```

The architecture has two complementary pipelines:

- **Data pipeline:** raw hospital files are transformed into cleaned, harmonised local datasets.
- **Model pipeline:** local clients train from the current global model, transform outgoing parameters, and participate in server aggregation.

## 3. End-to-End Execution Flow

The entry point is [main.py](main.py). Its execution sequence is:

1. Disable CUDA for the experiment and force CPU execution.
2. Preprocess raw hospital files with `preprocess_all_data()`.
3. Clean and harmonise the processed files with `clean_all_data()`.
4. Discover the shared outcome-class mapping.
5. Start a Flower server subprocess.
6. Start one client thread for each cleaned hospital CSV.
7. Run the configured federated rounds.
8. Wait for all clients to finish.
9. Generate per-run results, figures, and summaries.
10. Stop the server and begin the next experiment run.

The main experiment runner repeats this process for `NUM_RUNS` runs. The generated outputs are placed under `results/`.

### Federated round sequence

```mermaid
sequenceDiagram
    participant S as Flower server
    participant A as Hospital A client
    participant B as Hospital B client
    participant C as Hospital C client

    S->>A: Send global model parameters
    S->>B: Send global model parameters
    S->>C: Send global model parameters
    A->>A: Train locally for 30 epochs
    B->>B: Train locally for 30 epochs
    C->>C: Train locally for 30 epochs
    A->>S: Parameters + local sample count
    B->>S: Parameters + local sample count
    C->>S: Parameters + local sample count
    S->>S: Weighted FedAvg aggregation
    S-->>A: Updated global model
    S-->>B: Updated global model
    S-->>C: Updated global model
    A->>S: Local evaluation metrics
    B->>S: Local evaluation metrics
    C->>S: Local evaluation metrics
```

For client $k$ with $n_k$ training examples, the intended FedAvg update is:

$$
w_{t+1} = \sum_{k=1}^{K}\frac{n_k}{\sum_{j=1}^{K}n_j}w_{t+1}^{(k)}
$$

where $w_{t+1}^{(k)}$ is the locally trained parameter set returned by client $k$.

## 4. Data Engineering and Anonymisation

### 4.1 Raw-to-cleaned transformation

The preprocessing layer in [client/data_utils.py](client/data_utils.py) detects and standardises source columns, parses ages, categorises sponsors, generalises clinical fields, hashes selected values, and removes duplicate records. The cleaning layer in [client/clean_app.py](client/clean_app.py) removes direct identity and diagnosis columns, imputes vital signs, creates age bins, and maps outcomes.

The resulting feature set used by clients includes:

- `Age`
- `Sponsor`
- `Region`
- `Pulse`
- `Resp`
- `Temp`
- `Sys`
- `Dia`
- `Procedures`
- `Medications`
- `Age_bin`

### 4.2 Outcome harmonisation

All institutions use the same label space:

| Outcome | Numeric class |
|---|---:|
| Home | 0 |
| Admitted | 1 |
| Referred | 2 |
| Missing or unknown | 3 |

The shared mapping is important because model outputs from different hospitals must represent the same clinical categories.

### 4.3 Hashing and generalisation code

The following excerpt is the project's current salted hashing routine:

```python
def safe_hash(value, salt_key='id'):
    if pd.isna(value) or str(value).strip() == '':
        return np.nan
    clean = str(value).strip()
    salt = SALTS.get(salt_key, "")
    return hashlib.sha256((salt + clean).encode('utf-8')).hexdigest()
```

Clinical detail is also coarsened where possible:

```python
def generalize_clinical(value):
    if pd.isna(value) or str(value).strip().lower() in [
        '', 'none', 'no', '-', 'null', 'not recorded'
    ]:
        return "Not recorded"
    return "Recorded"
```

This reduces direct exposure, but hashing is pseudonymisation rather than guaranteed anonymisation. Static salts remain in the repository and should be moved to protected, institution-specific environment variables or a secret manager before deployment.

## 5. Federated Client Architecture

Each hospital client is implemented by `FlowerHospitalClient` in [client/client_app.py](client/client_app.py). A client performs the following operations:

1. Read only its own hospital CSV.
2. Select the shared model features.
3. Encode categorical fields.
4. Scale numeric features.
5. Apply local imbalance handling.
6. Construct the PyTorch model.
7. Receive global parameters from the server.
8. Train locally.
9. Apply configured privacy transformations.
10. Return parameters and sample count.
11. Evaluate the received global model on its local test set.

### Model code

The active model is a compact feed-forward classifier:

```python
class HospitalModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values, dtype=torch.float32)
        return self.net(x.float())
```

During federated training, the client uses weighted cross-entropy and performs a full-batch local optimization loop:

```python
criterion = nn.CrossEntropyLoss(weight=self.class_weights)
optimizer = optim.Adam(self.model.parameters(), lr=0.001)

for epoch in range(1, 31):
    self.model.train()
    optimizer.zero_grad()

    output = self.model(self.X_train)
    loss = criterion(output, self.y_train)
    loss.backward()
    optimizer.step()
```

Local SMOTE and class weighting are intended to reduce the influence of local class imbalance. However, SMOTE should ideally be applied only to the training partition; applying it before the train/test split can allow synthetic information to influence evaluation.

## 6. Server and Aggregation Architecture

The active server is in [server/server_app.py](server/server_app.py):

```python
import flwr as fl

class SecureFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}

        aggregated_params, metrics = super().aggregate_fit(
            rnd, results, failures
        )
        return aggregated_params, metrics


if __name__ == "__main__":
    fl.server.start_server(
        server_address="127.0.0.1:9090",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=SecureFedAvg(),
    )
```

The server performs model coordination and FedAvg aggregation. It does not receive raw patient rows. The separate [server/strategy.py](server/strategy.py) defines another FedAvg configuration, but it is not imported by the active `server_app.py`; therefore, `server_app.py` is the authoritative server path for the current experiment.

## 7. Privacy-Preserving Design

Privacy is implemented as a layered design rather than a single algorithm. Data minimisation and generalisation reduce sensitive detail before training, federated learning keeps patient rows at the institution, update clipping and noise reduce the precision of released parameters, and masking is intended to hide individual client contributions during aggregation.

| Algorithm or control | Operation | Intended protection | Current status |
|---|---|---|---|
| Data minimisation | Remove fields not needed by the model | Reduces sensitive information processed | Implemented |
| Salted SHA-256 | Hash `salt + value` | Prevents direct identifier transmission | Implemented; static salts need replacement |
| Generalisation | Replace detailed fields with broad categories | Reduces re-identification risk | Partially implemented |
| Age binning | Convert exact ages into ranges | Reduces quasi-identifier precision | Implemented |
| Federated learning | Train at each hospital and exchange parameters | Prevents routine central collection of patient rows | Implemented |
| Gaussian perturbation | Clip parameters and add random noise | Reduces information exposed by updates | Prototype only |
| Additive masking | Add a mask before upload | Intended to conceal individual updates | Prototype only |
| TLS and authentication | Encrypt and authenticate network traffic | Protects updates in transit | Required for deployment |

### 7.1 Local data protection and raw-data non-sharing

The strongest currently implemented architectural property is data locality. Each hospital client reads its own CSV, performs preprocessing locally, constructs tensors locally, and trains locally. The server receives model-related values rather than the original patient table.

```mermaid
flowchart LR
    H[Hospital data store] -->|Raw rows remain inside hospital| L[Hospital client]
    L -->|Parameters, sample count, selected metrics| S[Flower server]
    S -->|Global model parameters| L
    S -.->|No raw patient CSV transfer| H
```

During a normal federated round, these remain inside the institution:

- Patient identifiers and demographic rows.
- Raw clinical measurements and original categorical text.
- Local feature matrices, labels, train/test partitions, and intermediate gradients.
- Local predictions used for hospital-level evaluation.

The following may leave the institution:

- Model parameter arrays returned by `get_parameters()` and `fit()`.
- The local training sample count used for weighted FedAvg.
- Evaluation values such as loss, accuracy, and other metrics.
- Experiment metadata and logs generated by the orchestration process.

This distinction matters because federated learning prevents direct raw-row sharing, but model updates can still leak information through membership inference, gradient analysis, or model inversion. “Raw data not shared” is therefore a data-flow property, not proof that the trained model reveals nothing about patients.

In a real deployment, each hospital should expose only an authenticated client gateway. The gateway should read the approved local dataset and communicate with the federation server, while the server should have no filesystem or database permissions inside the hospital. Outgoing payloads should be type- and size-validated, logs should not contain feature values, and raw-data access should be audited. The current local script does not enforce this isolation because the server and clients run on the same development machine.

### 7.2 Algorithm 1: salted hashing and pseudonymisation

For an input value $x$ and secret salt $s$, the client computes:

$$
h = \operatorname{SHA256}(s \mathbin{||} \operatorname{normalise}(x))
$$

where $\mathbin{||}$ means concatenation. The same salt and value produce the same digest, which supports local duplicate detection without transmitting the original value. Hashing is not encryption: low-entropy values may still be guessed if the salt is exposed or an attacker can test candidate values. The current salts are stored in source code and must be moved to protected, institution-specific secrets.

### 7.3 Algorithm 2: local generalisation and data minimisation

Before training, the pipeline reduces detail by dropping direct identifiers, converting age into ranges, grouping sponsor values, replacing selected clinical text with `Recorded` or `Not recorded`, imputing missing vitals, and deduplicating available records. This is data minimisation and generalisation, not formal $k$-anonymity. A formal privacy assessment should calculate the size of every equivalence class formed by quasi-identifiers such as age bin, region, and sponsor.

### 7.4 Algorithm 3: federated local training and FedAvg

At round $t$, the server broadcasts global parameters $w_t$. Hospital $k$ trains locally and returns $w_{t+1}^{(k)}$. The server calculates:

$$
w_{t+1} = \sum_{k=1}^{K}\frac{n_k}{\sum_{j=1}^{K}n_j}w_{t+1}^{(k)}
$$

where $n_k$ is the local training count. The server does not need patient rows for this calculation. The privacy benefit comes from moving computation to the data rather than moving data to a central store; model parameters, update sizes, and metrics can nevertheless leak information.

### 7.5 Algorithm 4: parameter clipping and Gaussian noise

The client currently applies clipping and Gaussian noise to the returned parameter arrays:

```python
def clip_and_add_noise(params, C=5.0, sigma=DP_SIGMA):
    dp_params = []

    for parameter in params:
        norm = np.linalg.norm(parameter)
        clipped = parameter * min(1.0, C / (norm + 1e-8))
        noise = np.random.normal(0, sigma, clipped.shape)
        dp_params.append(clipped + noise)

    return dp_params
```

Conceptually, this resembles:

$$
\tilde{w} = \operatorname{clip}(w, C) + \mathcal{N}(0, \sigma^2I)
$$

More explicitly, for tensor $w$ and clipping threshold $C$, the implementation computes:

$$
\operatorname{clip}(w,C) = w\min\left(1, \frac{C}{\lVert w\rVert_2 + 10^{-8}}\right)
$$

and then adds $z \sim \mathcal{N}(0,\sigma^2I)$. Clipping bounds the magnitude of the submitted tensor, while the Gaussian term makes the exact unnoised value less precise. The code uses `C = 5.0` and `DP_SIGMA = 0.002` when enabled.

This is not sufficient by itself to claim formal differential privacy. The implementation does not define sensitivity over a client update, does not use per-example gradient clipping, and does not calculate an accountant-derived $(\epsilon, \delta)$. Noise scale alone cannot establish a privacy guarantee. A defensible DP implementation would define the adjacency relation, sampling assumptions, composition across rounds, and privacy budget. Although Opacus appears in the dependencies, the active path does not use a `PrivacyEngine` or privacy accountant.

### 7.6 Algorithm 5: additive masking and secure aggregation

The current masking helpers are:

```python
def generate_mask(params, seed, scale=1e-3):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, scale, p.shape) for p in params]


def apply_mask(params, masks):
    return [p + mask for p, mask in zip(params, masks)]
```

The client applies the mask after the noise transformation:

```python
if USE_SMPC:
    round_seed = int(config.get("server_round", 0))
    masks = generate_mask(params, seed=round_seed)
    params = apply_mask(params, masks)
```

This is an experimental additive perturbation, not a complete secure-aggregation protocol. For true additive secure aggregation, clients need masks $r_k$ satisfying:

$$
\sum_{k=1}^{K}r_k = 0
$$

so that:

$$
\sum_{k=1}^{K}(w_k+r_k)=\sum_{k=1}^{K}w_k
$$

while the server cannot isolate an individual $w_k$. The current clients use the same round seed, so they generate the same mask rather than complementary masks. The average mask therefore remains in the aggregate and may corrupt the global model. A production protocol would use pairwise masks or a vetted cryptographic secure-aggregation scheme, authenticated key exchange, dropout recovery, minimum-participant thresholds, and controlled unmasking only after enough clients have contributed.

### 7.7 Privacy threat model and residual risks

The architecture primarily reduces exposure to a central coordinator that should not receive raw hospital tables. It does not by itself fully protect against a malicious client submitting poisoned updates, a curious server analysing unmasked updates, membership inference against the global model, model inversion, re-identification from rare quasi-identifier combinations, leakage through logs or temporary files, or collusion between the coordinator and hospitals.

Privacy verification should therefore inspect both code and operations: capture network payloads to confirm that no CSV rows are sent, check logs for identifiers and feature values, validate that result files contain only approved aggregates, and test the trained model for membership or inversion leakage where appropriate.

## 8. Evaluation, Results, and Fairness

The system saves numerical and visual outputs under `results/`.

### 8.1 Dataset and contribution figures

![Dataset size by hospital](results/global/dataset_sizes_by_hospital.png)

*Figure 1. Number of records represented by each participating hospital.*

![Federated weights](results/global/federated_weights.png)

*Figure 2. Relative FedAvg contribution weights based on hospital sample counts.*

![Sample distribution](results/global/sample_distribution.png)

*Figure 3. Training, testing, and total sample distribution.*

![Contribution matrix](results/global/contribution_matrix.png)

*Figure 4. Hospital contribution matrix used to visualise aggregation weights.*

### 8.2 Hospital-level confusion matrices

The project records local predictions and generates hospital-specific confusion-matrix images. The current evaluation converts the four-class outcome into a binary task: `Admitted` is positive and all other outcomes are negative.

![Hospital A confusion matrix](results/confusion_matrices/HospitalA_confusion_matrix.png)

*Figure 5. Hospital A confusion matrix.*

![Hospital B confusion matrix](results/confusion_matrices/HospitalB_confusion_matrix.png)

*Figure 6. Hospital B confusion matrix.*

![Hospital C confusion matrix](results/confusion_matrices/HospitalC_confusion_matrix.png)

*Figure 7. Hospital C confusion matrix.*

The corresponding numeric matrices are available in:

- `results/confusion_matrices/HospitalA_round_0.csv`
- `results/confusion_matrices/HospitalB_round_0.csv`
- `results/confusion_matrices/HospitalC_round_0.csv`

### 8.3 Training dynamics

![Hospital A accuracy](results/epoch/epoch_accuracy/HospitalA_epoch_accuracy_avg.png)

*Figure 8. Average epoch-wise training accuracy for Hospital A.*

![Hospital B accuracy](results/epoch/epoch_accuracy/HospitalB_epoch_accuracy_avg.png)

*Figure 9. Average epoch-wise training accuracy for Hospital B.*

![Hospital C accuracy](results/epoch/epoch_accuracy/HospitalC_epoch_accuracy_avg.png)

*Figure 10. Average epoch-wise training accuracy for Hospital C.*

![Hospital A loss](results/epoch/epoch_loss/HospitalA_epoch_loss_avg.png)

*Figure 11. Average epoch-wise training loss for Hospital A.*

![Hospital B loss](results/epoch/epoch_loss/HospitalB_epoch_loss_avg.png)

*Figure 12. Average epoch-wise training loss for Hospital B.*

![Hospital C loss](results/epoch/epoch_loss/HospitalC_epoch_loss_avg.png)

*Figure 13. Average epoch-wise training loss for Hospital C.*

### 8.4 Experiment comparison

![Experiment comparison](results/experiments/experiment_comparison.png)

*Figure 14. Comparison of recorded metrics across experiment runs.*

![Experiment summary table](results/global/experiment_summary_table.png)

*Figure 15. Rendered experiment summary table.*

The primary numerical sources are `results/experiment_runs.csv`, `results/experiment_summary.csv`, and the per-run files under `results/experiments/`.

### 8.5 Metrics

The implementation calculates or records:

- Accuracy
- Precision
- Recall or sensitivity
- Specificity
- F1 score
- False-positive rate
- False-negative rate
- Local training loss and accuracy
- Per-hospital confusion matrices

The standalone fairness utilities can support sponsor-stratified analysis and subgroup resampling. However, demographic parity, equal opportunity, subgroup recall, and hospital-level disparity metrics are not yet fully integrated into the main federated run. They should be reported separately once calculated from consistently tracked per-hospital predictions.

## 9. Security and Methodological Caveats

The following points should be stated clearly in a thesis, paper, or deployment assessment:

| Area | Current state | Required strengthening |
|---|---|---|
| Raw-data locality | Implemented in the client workflow | Add deployment controls and audit verification |
| Identifier hashing | Implemented with static salts | Use protected, rotating, institution-specific secrets |
| Data generalisation | Partially implemented | Perform formal re-identification and k-anonymity checks |
| Federated aggregation | Weighted FedAvg implemented | Secure transport, authentication, and coordinator hardening |
| Differential privacy | Parameter clipping and noise prototype | Add Opacus/accounting and report $(\epsilon, \delta)$ |
| Secure aggregation | Additive masking prototype | Implement pairwise/cryptographic secure aggregation |
| Feature encoding | Shared in-memory encoder cache | Fit and distribute deterministic schemas explicitly |
| Evaluation | Binary metrics from four-class labels | Report multiclass metrics and document binary reduction |
| Fairness | Utilities exist but are not central to the run | Add subgroup and hospital disparity reporting |
| Test validity | Local resampling may precede splitting | Split first, then apply SMOTE only to training data |

A further implementation concern is that clients run concurrently while global encoder state is shared. The first client reaching the encoder-fitting code can influence the categorical vocabulary. A production design should define a deterministic schema or federated vocabulary before training begins.

## 10. Recommended Production Architecture

A hardened deployment should extend the current prototype with:

1. **Institutional gateways:** run one isolated client service per hospital rather than sharing a filesystem process.
2. **Secure transport:** use TLS, certificate-based client authentication, and server authorization.
3. **Secret management:** remove static salts and store secrets in a managed vault.
4. **Deterministic feature schema:** distribute a versioned encoder vocabulary and feature order.
5. **Formal DP:** clip per-example gradients or well-defined client updates, use an accountant, and publish privacy budgets.
6. **Cryptographic secure aggregation:** use pairwise masks or a vetted secure-aggregation protocol with dropout handling.
7. **Robust aggregation:** consider client validation, anomaly detection, and Byzantine-resilient aggregation.
8. **Evaluation isolation:** keep an untouched local test partition and apply SMOTE only after the split.
9. **Fairness monitoring:** calculate subgroup recall, false-positive rate, demographic parity, and equal opportunity by hospital and sponsor group.
10. **Governance:** record consent, data-sharing agreements, IRB/ethics approval, model versions, audit events, and rollback procedures.

## 11. Reproducibility

The principal experiment can be run from the repository root with the project environment active:

```powershell
python main.py
```

For manual component execution:

```powershell
python client/data_utils.py
python client/clean_app.py
python server/server_app.py
python client/client_app.py data/cleaned/HospitalA.csv
python client/client_app.py data/cleaned/HospitalB.csv
python client/client_app.py data/cleaned/HospitalC.csv
```

The exact Python dependencies are listed in [requirements.txt](requirements.txt). Before reporting results, record the code revision, Python version, dependency versions, privacy flags, clipping norm, noise scale, number of rounds, local epochs, train/test policy, and the experiment seed policy.

## Conclusion

Mwakatobe establishes the main architectural pattern required for privacy-conscious multi-institutional healthcare learning: data stays at the hospital, local clients train a common model, and a coordinator aggregates model information rather than patient rows. The project also includes useful preprocessing, anonymisation, imbalance handling, hospital-level evaluation, and result visualisation.

The current system should be presented as a research prototype. Its core federated workflow is operational, while formal differential privacy and secure aggregation remain engineering and validation tasks. Making that distinction explicit strengthens the report: it separates the demonstrated data-locality benefit from privacy guarantees that require additional cryptographic implementation, accounting, and empirical validation.
