# Patient Health Risk & Length-of-Stay Prediction

A two-stage machine learning pipeline that:
1. Predicts a patient's **health risk level** (Normal / Low / Medium / High) from basic vital signs, similar to an early warning score used in clinical triage.
2. If the predicted risk is **Medium or High**, predicts the patient's likely **length of hospital stay** (in days) using a broader set of clinical and demographic features.

> ⚠️ **Disclaimer:** This project is for educational purposes only. It is trained on public/sample datasets and is **not validated for real clinical or diagnostic use**.

## How It Works

### Stage 1 — Risk Level Prediction
Given a patient's vitals, a Random Forest classifier predicts one of 4 risk levels.

**Inputs:**
| Feature | Description |
|---|---|
| Respiratory_Rate | Breaths per minute |
| Oxygen_Saturation | SpO2 (%) |
| O2_Scale | Oxygen scale used (1 or 2) |
| Systolic_BP | Systolic blood pressure |
| Heart_Rate | Beats per minute |
| Temperature | Body temperature (°C) |
| Consciousness | A = Alert, C = Confusion, P = Pain response, U = Unresponsive, V = Verbal |
| On_Oxygen | 0 = No, 1 = Yes (on supplemental oxygen) |

**Output:** `Normal`, `Low`, `Medium`, or `High` risk.

### Stage 2 — Length of Stay Prediction
If risk is Medium or High, a second regression model estimates length of hospital stay using 23 features covering pre-existing conditions (asthma, renal disease, psychological disorders, etc.), lab values (hematocrit, glucose, creatinine, BMI, etc.), and admin data (facility ID, revisit count).

**Output:** Predicted length of stay in days.

## Models Used
- `risk_model_rf.pkl` — Random Forest Classifier (risk level prediction, primary model used)
- `risk_model_lr.pkl` — Logistic Regression Classifier (alternate risk model, included for comparison)
- `base_model.pkl` — Regression model for length-of-stay prediction

## Usage

```bash
pip install -r requirements.txt
python health.py
```

The script will prompt you to enter patient vitals interactively. If the predicted risk is Medium or High, it will then prompt for the additional features needed to estimate length of stay.

**Example run:**
```
Enter R to check if patient has Health Risk: R
Enter Respiratory_Rate: 28
Enter Oxygen_Saturation: 92
Enter O2_Scale: 2
Enter Systolic_BP: 116
Enter Heart_Rate: 151
Enter Temperature: 38.5
Enter Consciousness('A', 'C', 'P', 'U', 'V'): P
Enter On_Oxygen: 1
Predicted Risk Level: High
Enter S to check how much long a patient has to Stay: S
...
Predicted Length of Stay: 7.2 days
```

## Files
```
├── health.py            # Main script — loads models and runs predictions
├── risk_model_rf.pkl    # Random Forest model for risk classification
├── risk_model_lr.pkl    # Logistic Regression model for risk classification
├── base_model.pkl       # Regression model for length-of-stay prediction
├── requirements.txt
└── README.md
```

## Notes / Limitations
- Trained on a small/sample dataset — not representative of all patient populations.
- No input validation yet on interactive prompts (e.g., invalid consciousness letters or non-numeric input will raise an error).
- Intended as a learning project to practice classification, regression, and building a simple decision pipeline — not a substitute for clinical judgment.

## Future Improvements
- Add input validation and error handling
- Deploy as a simple web app (Streamlit/Gradio) for easier demoing
- Add model evaluation metrics (accuracy, confusion matrix, RMSE) to this README
- Add a training script/notebook showing how the models were built
