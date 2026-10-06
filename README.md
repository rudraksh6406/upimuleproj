# MuleGuard

Real-time UPI mule-account detection using a **Temporal Graph Network (TGN)**, synthetic transaction streams, an auditable policy engine, and a FastAPI/D3 live dashboard.

> **Research status:** This is an academic prototype trained on synthetic data. It must not be used to freeze real accounts without validation, calibration, fairness review, and human oversight. Files under `results/` are experiment artifacts, not claims about production performance.

## What is implemented

- Realistic UPI account/VPA generation and nine planted mule typologies
- Stateful TGN: message MLP, mean message aggregation, GRU node memory, continuous-time encoding, multi-head temporal attention, and classifier
- Rule, logistic regression, XGBoost, GCN, GraphSAGE, and TGAT baselines
- Rolling multi-edge transaction graph (repeated transfers are preserved)
- Guardian actions (`ALLOW`, `FLAG`, `RESTRICT`, `FREEZE`), explanations, SQLite audit trail, and human override API
- FastAPI REST/WebSocket backend with a D3 dashboard
- Honest measured evaluation (ROC/PR, confusion matrix, FPR, latency); no target metrics are substituted for model outputs

The detailed design and acceptance criteria are in [PRD_MuleGuard_TGNN.md](PRD_MuleGuard_TGNN.md).

## Quick start

### Docker

```bash
docker compose up --build
# open http://localhost:8000
```

### Local Python

Python 3.10+ is required (3.11 recommended).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
pytest -q
python scripts/run_demo.py --host 0.0.0.0 --port 8000
```

## Reproduce training and evaluation

The default generator CLI uses development-sized datasets. Increase account counts to the PRD scale only when sufficient compute is available.

```bash
python scripts/generate_data.py --accounts_train 2000 --accounts_val 1000 --accounts_test 1000
python scripts/train_model.py --epochs 50 --batch_size 200 --lr 0.0001 --device cuda
python scripts/evaluate.py --device cuda --temporal-epochs 10
```

Use `--device cpu` where CUDA is unavailable. Data is generated in chronological order. Training resets temporal state each epoch and detaches memory between batches for truncated backpropagation through time. Evaluation consumes the real checkpoint predictions and writes measured tables/plots to `results/`.

### Avoiding misleading results

- Tune thresholds on validation data, then freeze them before test evaluation.
- Use non-overlapping account populations and chronological splits.
- Compare like with like: node-level baseline predictions and event-level TGN predictions are not interchangeable.
- Run multiple paired seeds before using `paired_model_comparison` for a significance claim.
- Never describe PRD targets as observed metrics.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/metrics` | Live throughput/alert/latency summary |
| `GET /api/graph` | Current graph snapshot |
| `GET /api/alerts` | Recent audited alerts |
| `GET /api/accounts/{vpa}` | Account risk and transaction details |
| `POST /api/accounts/action` | Human action override |
| `WS /ws` | Live graph, alert, and metric events |
| `GET /docs` | Interactive OpenAPI documentation |

## Project layout

```text
config/                 model, data, Guardian, dashboard settings
scripts/                generation, training, evaluation, demo entry points
src/data/               generator, patterns, preprocessing, temporal loader
src/models/             TGN/TGAT and baseline implementations
src/pipeline/           rolling graph, inference, stream orchestration
src/guardian/           policy, explanations, SQLite audit
src/dashboard/          FastAPI, WebSocket, D3 frontend
src/evaluation/         metrics, plots, per-pattern and statistical analysis
tests/                  unit and integration tests
```

## Security and privacy

The generator creates synthetic identifiers; do not place real payment data in this repository. The demo's permissive CORS policy is for local development. Restrict origins, add authentication/authorization, encrypt storage, and remove ground-truth fields before any non-local deployment.

## License

No license file is currently included. All rights remain with the repository owner unless a license is added.
