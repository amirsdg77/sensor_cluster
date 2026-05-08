# sensorcluster

Semi-supervised clustering of multivariate sensor data for **predictive-maintenance triage** and **unknown-failure-mode discovery**.

When an industrial machine fails, you usually know *that* it failed and have a thick history of sensor readings around each failure — but you rarely have the budget for a domain expert to label every event. `sensorcluster` takes a small set of expert-labeled breakdowns and a much larger set of unlabeled ones, finds the dense regions in sensor space, names the clusters that overlap with known labels, and **flags clusters that contain no labels as candidate undiscovered failure modes**.

```
            scaler -> PCA -> HDBSCAN ----+--> cluster_id, membership strength
                                         |
                                         +--> GLOSH outlier score -> novelty_score
                                         |
   40 expert labels --> majority vote ---+--> cluster name (or "UNKNOWN_k" if no labels)
```

---

## Why HDBSCAN

- **Density-based** → no spherical-cluster assumption, no need to pick `k` upfront.
- **Native noise label** (`cluster == -1`) → an honest "I don't know" instead of forced classification.
- **`approximate_predict`** for new points → cluster + calibrated membership strength in one call.
- **GLOSH outlier scores** → principled novelty score from the same density model that produced the clusters.

One model, three useful outputs (label, confidence, novelty), one artifact to deploy.

A side-by-side empirical comparison against KMeans, GMM, Bayesian GMM, DBSCAN, LabelSpreading (with honest leave-one-out CV), Spectral, Agglomerative, and IsolationForest is in [notebooks/02_algorithm_comparison.ipynb](notebooks/02_algorithm_comparison.ipynb).

---

## Quickstart

```bash
# 1. Install (uses uv)
uv sync --all-extras

# 2. Train end-to-end on the included sample dataset
uv run sensorcluster train --config configs/base.yaml
# writes artifacts/{scaler,pca,hdbscan}.joblib + cluster_label_map.json + evaluation_report.md

# 3. Serve the inference API
uv run sensorcluster serve --port 8000

# 4. Predict
curl -X POST http://localhost:8000/predict \
     -H 'content-type: application/json' \
     -d '{"sensors":[0.10,-0.42, ... ,0.05]}'
```

Or all-in-one with Docker:

```bash
docker compose --profile train up training      # produces ./artifacts
docker compose up -d api mlflow                 # API on :8000, MLflow UI on :5000
```

---

## Repository layout

```
.
├── data/raw/data_sensors.csv          # sample sensor dataset (1600 readings, 20 channels, 40 expert labels)
├── artifacts/                          # trained model + report (gitignored)
├── configs/base.yaml                   # default hyperparameters
├── src/sensorcluster/
│   ├── data/                           # schema, loader, CV splits
│   ├── features/                       # scaler, PCA, UMAP (viz)
│   ├── models/                         # HDBSCAN wrapper + bundled pipeline
│   ├── pipeline/                       # train, predict_batch, label_propagation
│   ├── evaluation/                     # metrics, CV, report generation
│   ├── visualization/                  # UMAP / t-SNE plots
│   ├── api/                            # FastAPI inference service
│   └── cli.py                          # typer entrypoint: train | predict | evaluate | serve
├── notebooks/                          # 01_eda → 05_business_story
├── tests/                              # unit + integration (pytest)
├── docker/                             # multi-stage Dockerfiles
└── docker-compose.yml                  # api + mlflow + training + jupyter
```

---

## Configuration

All hyperparameters live in [configs/base.yaml](configs/base.yaml). Override via CLI flags or `SENSORCLUSTER_*` environment variables (12-factor).

Key knobs:

| Section | Knob | Default | Effect |
|---|---|---|---|
| `pca` | `variance_target` | `0.95` | Auto-pick `n_components` to retain this fraction of variance |
| `hdbscan` | `min_cluster_size` | `15` | Smallest grouping HDBSCAN will accept as a cluster |
| `hdbscan` | `min_samples` | `5` | Conservativeness of noise labeling (lower → less noise) |
| `novelty` | `glosh_threshold` | `0.7` | GLOSH score above this → `is_novel = true` |
| `label_propagation` | `purity_warning` | `0.6` | Per-cluster purity below this triggers a warning in the report |

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health` | Liveness + model version + train timestamp |
| `POST` | `/predict` | Single sensor reading → `{label, confidence, novelty_score, cluster_id, is_novel, top_neighbors}` |
| `POST` | `/predict_batch` | Same, vectorized |
| `GET`  | `/clusters` | Label map with per-cluster purity (introspection) |
| `GET`  | `/metrics` | Prometheus exposition |

OpenAPI schema is auto-served at `/docs` (Swagger UI) and `/openapi.json` when the API is running.

---

## Results on the sample dataset

After training on `data/raw/data_sensors.csv` (1600 readings × 20 sensors, 40 expert labels across 3 known failure modes), the report at `artifacts/evaluation_report.md` summarizes:

- per-cluster purity and label assignment,
- 5-fold cross-validated ARI on the 40 labeled points,
- silhouette and Davies–Bouldin internal metrics,
- a UMAP plot with labeled points highlighted and `UNKNOWN_*` clusters circled.

The bundled sample has nearly-independent sensor channels with weak density structure — an honest dataset to probe how the system behaves when the inputs are diffuse. With `min_cluster_size=8, min_samples=3` and PCA at 95 % variance:

| Metric | Value |
|---|---|
| Clusters discovered | 2 (1 mapped to `CLASS_1`, 1 `UNKNOWN_0` candidate) |
| Noise fraction | ~82 % |
| CV mean ARI on the labeled subset | ~0.12 ± 0.18 |
| Silhouette (non-noise) | ~0.003 |
| Bootstrap stability ARI | ~0.50 |

Two important properties of this output:

1. **The system surfaces an `UNKNOWN_0` cluster** — exactly the discovery deliverable, even on a difficult sample.
2. **The report flags low quality loudly** rather than fabricating better-looking numbers. On real production sensor data with correlated channels and physical dynamics, expect substantially better silhouette and CV-ARI; the same code path applies.


---

## Development

```bash
uv sync --all-extras                                              # install with all extras
uv run ruff check . && uv run ruff format --check .               # lint + format
uv run mypy src/                                                  # static types (strict)
uv run pytest --cov=src/sensorcluster --cov-fail-under=80         # tests + coverage gate
```

A [Makefile](Makefile) exposes the most-used workflows as short aliases — `make install`, `make lint`, `make test`, `make cov`, `make train`, `make serve`, and `make demo` (install → train → serve in one shot). The same checks run on every push via [.github/workflows/ci.yml](.github/workflows/ci.yml).
