How to run it

1. Open a terminal in the project folder (the one that contains `experiments`, `src`, and `data`).
2. Use Python 3.11 or newer. If `python` is missing, use `python3` in the commands below.
3. Create and activate a virtual environment.
   - Windows (PowerShell): `python -m venv .venv` then `.venv\Scripts\Activate.ps1`
   - macOS or Linux: `python -m venv .venv` then `source .venv/bin/activate`
4. Install the dependencies: `pip install -r requirements.txt`
5. Run: `python -m experiments.run_experiment`
6. Read the table in the terminal. The same text is saved to `artifacts/comparison.txt`. The plots are `figures/train_overlay.png` and `figures/train_test_overlay.png`.
