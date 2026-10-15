# The Aqua-Epoch — Water Potability Prediction

A beginner-friendly machine-learning project using the supplied BIS water-quality dataset.

## Files
- `water_quality.csv` — dataset converted from the uploaded Excel `.xls` file.
- `train_model.py` — cleans the target, handles impossible negative pH/hardness/fluoride values as missing, compares three classifiers, evaluates them on a held-out test set, and saves the best model by positive-class F1.
- `app.py` — Streamlit web application for entering measurements and viewing the prediction.
- `requirements.txt` — Python dependencies.
- `models/` — trained model and metadata are created here after training.

## Run on Windows with VS Code

Open this folder in VS Code. In the terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If PowerShell blocks environment activation, use Command Prompt instead:
```bat
.venv\Scripts\activate.bat
```

Then open the local URL Streamlit prints in the terminal (usually `http://localhost:8501`).

## Important notes
- The target column is `Potability` (`0` or `1`).
- The file has 8,691 rows and 19 columns. One row has a missing target and is excluded from training.
- Some negative values appear in pH, Hardness, and Fluoride. These are treated as missing during training, rather than learned as valid measurements.
- The training script compares Random Forest, Extra Trees, and HistGradientBoosting using a stratified 80/20 train/test split. It selects by F1 for class `1` and saves test metrics.
- This is a challenge prototype, not a certified water-safety device. Model predictions do not replace accredited laboratory testing or microbial analysis. The dataset may contain implausible or extreme measurements; review data quality and units with your challenge documentation.
- Do not claim model accuracy before running `train_model.py` on your machine. The script prints metrics and creates `model_comparison.csv`.
