# How to run the experiment

Run these steps from the project directory. That is the folder that contains `experiments`, `src`, `data`, and `requirements.txt`.

You need Python 3.11 or newer. These steps use a virtual environment named `.venv`.

## 1. Open a terminal in the project directory

Windows: open PowerShell and `cd` to the project directory.

macOS or Linux: open a terminal and `cd` to the project directory.

Check the interpreter:

```bash
python --version
```

If that command is missing, use `python3` in every step below, including `python3 -m venv` and `python3 -m experiments.run_experiment`.

## 2. Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

The prompt shows `(.venv)` after activation. Activate this environment again in each new terminal before running the experiment.

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs NumPy and Matplotlib.

## 4. Run the experiment

```bash
python -m experiments.run_experiment
```

The command fits the two-regime SETAR model on `data/stock_train.csv`, then scores `data/stock_test.csv` with those coefficients held fixed. It does not refit on the test prices.

## 5. Read the results

The terminal prints a comparison of the normal-equation solver and the Givens QR solver, then the five largest training residuals.

The same run writes three files:

| File | Contents |
|---|---|
| `artifacts/comparison.txt` | FLOPs, memory, condition numbers, train residual 2-norm, train RMSE, test RMSE, runtime, and the largest training residuals |
| `figures/train_overlay.png` | Training return `R_t` and the fitted return |
| `figures/train_test_overlay.png` | Training and test returns joined together, with a vertical line at the split |

Running the command again replaces those three files. The runtime column changes from run to run. The other numeric columns are properties of the fit.
