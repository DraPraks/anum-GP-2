# SETAR least squares (GP1, problem 2)

Even-group implementation of a two-regime SETAR model (lag 2, threshold 0, delay 1)
fit by least squares. The unknown vector is

`x = [alpha_1, phi_1_1, phi_1_2, alpha_2, phi_2_1, phi_2_2]`.

Two solvers are written in-repo: normal equations with dense LU and partial
pivoting, and thin QR by Givens rotations. Library factorizations and solvers
(`numpy.linalg.solve`, `qr`, `lstsq`, and SciPy equivalents) are not used.

## Layout

| Path | Role |
|---|---|
| `data/stock_train.csv` | 303 Close observations |
| `data/stock_test.csv` | 103 Close observations |
| `data/fixture.py` | Printed six-price `A` and `b` |
| `src/returns.py` | Simple returns |
| `src/design.py` | Design matrix builder (train and test) |
| `src/validation.py` | Shape and sparsity checks |
| `src/lu.py` | Dense LU with partial pivoting |
| `src/normal_equations.py` | `AᵀA x = Aᵀb` |
| `src/givens.py` | Givens QR and the first-column rotation check |
| `src/metrics.py` | Residual norm, condition numbers, RMSE |
| `src/outliers.py` | Largest absolute train residuals |
| `src/plots.py` | Return overlays |
| `experiments/run_experiment.py` | Table and figures |

## Run

Interpreter: Python 3.11 or newer. Run these commands from this directory.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m experiments.run_experiment
```

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m experiments.run_experiment
```

`python -m experiments.run_experiment` rebuilds every number and figure used in
the report. It writes:

- stdout and `artifacts/comparison.txt` — FLOPs, memory, `kappa_2(A)`, `kappa_2(AᵀA)`, train residual 2-norm, train RMSE, test RMSE, runtime, and the largest training residuals
- `figures/train_overlay.png` — training `R_t` and fitted `R_t`
- `figures/train_test_overlay.png` — train concatenated with test, vertical split

Coefficients are estimated on the training design only and then frozen for the
test RMSE. The test set is not refit.
