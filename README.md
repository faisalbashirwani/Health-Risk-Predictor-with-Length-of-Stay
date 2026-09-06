# Patient Health Risk & Length-of-Stay Prediction

A two-stage machine learning pipeline that:
1. Predicts a patient's **health risk level** (Normal / Low / Medium / High) from basic vital signs, similar to an early warning score used in clinical triage.
2. If the predicted risk is **Medium or High**, predicts the patient's likely **length of hospital stay** (in days) using a broader set of clinical and demographic features.

>  **Disclaimer:** In README AI was used to some extend but everthing else wasn't.
>  **Disclaimer:** This project is for educational purposes only. It is trained on public/sample datasets and is **not validated for real clinical or diagnostic use**.

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
# Some samples to try on.
Respiratory_Rate,Oxygen_Saturation,O2_Scale,Systolic_BP,Heart_Rate,Temperature,Consciousness,On_Oxygen,Risk_Level
25,96,1,97,107,37.5,A,0,Medium
28,92,2,116,151,38.5,P,1,High
18,98,1,127,72,36.6,A,0,Normal
19,97,1,104,93,37.9,A,1,Low

 gender,dialysisrenalendstage,asthma,irondef,pneum,substancedependence,psychologicaldisordermajor,depress,
 psychother,fibrosisandother,malnutrition,hemo,hematocrit,neutrophils,sodium,glucose,bloodureanitro,creatinine,
 bmi,pulse,respiration,secondarydiagnosisnonicd9,discharged,facid, rcount -> lengthofstay
F,0,0,0,0,0,0,0,0,0,0,0,11.5,14.2,140.3611318,192.4769177,12,1.390722238,30.43241778,96,6.5,4,B,0, -->  3
F,0,0,0,0,0,0,0,0,0,0,0,9,4.1,136.7316918,94.07850731,8,0.943164319,28.46051612,61,6.5,1,A,5+, --> 7
F,0,0,0,0,0,0,0,0,0,0,0,8.4,8.9,133.0585135,130.5305238,12,1.065750282,28.84381191,64,6.5,2,B,1 --> 3
F,0,0,0,0,0,0,0,0,0,0,0,11.9,9.4,138.994023,163.3770276,12,0.90686182,27.95900732,76,6.5,1,,A,0 -->1
F,0,0,0,1,0,1,0,0,0,0,0,9.1,9.05,138.6348364,94.88665408,11.5,1.242854164,30.25892703,67,5.6,2,E,0 --> 4

## Notes / Limitations
- Trained on a small/sample dataset — not representative of all patient populations.
- No input validation yet on interactive prompts (e.g., invalid consciousness letters or non-numeric input will raise an error).
- Intended as a learning project to practice classification, regression, and building a simple decision pipeline — not a substitute for clinical judgment.

## Future Improvements
- Add input validation and error handling
- Deploy as a simple web app (Streamlit/Gradio) for easier demoing
- Add model evaluation metrics (accuracy, confusion matrix, RMSE) to this README
- Add a training script/notebook showing how the models were built
